"""lane_boxes/fp8_unpack.py — funct model of FP8Unpack.scala.

`FP8Unpack` is the **1 → 2 phased + queued** counterpart of `FP8Pack`.
One packed `req.valid` pulse (16 UInt16 slots, two FP8 bytes each) is
buffered into the FSM, and two `resp.valid` pulses are emitted on
consecutive cycles carrying the unpacked low half and high half as
BF16 values.

Cycle accounting (mirrors `FP8Unpack.scala:40-128`):

    cycle K    : req=packed → reqQ.enq.fire (deq still false this cycle)
    cycle K+1  : state=sIdle, reqQ.deq.valid → consume from queue,
                                                inputBuf := xVec.asUInt,
                                                expBuf   := expShift,
                                                state    := sLow
    cycle K+2  : state=sLow, io.resp.valid=1, output = convert(low half)
                                              state.next = sHigh
    cycle K+3  : state=sHigh, io.resp.valid=1, output = convert(high half)
                                               state.next = sIdle (or sLow if
                                                                    queue non-empty)

Two cycles of latency before the first output beat. The queue is
**32 deep** (`Queue(FP8UnpackReq, 32)`, default `pipe=false flow=false`,
1-cycle enq→deq latency). The state machine dequeues in both `sIdle`
and `sHigh`, so a back-to-back stream of inputs reaches an equilibrium
of one dequeue every two cycles (matching the 1-input → 2-output
expansion).

Subnormal FP8 inputs **and** the reserved NaN encoding (`exp=0xF,
mant=0x7`) flush to **signed zero** — the MXU dequant policy. The
per-byte conversion lives in `fp8_e4m3.e4m3_byte_to_bf16`, already
validated against `RVFP8UnpackTest.scala`'s golden.

Use `unpack_low_row(req)` / `unpack_high_row(req)` for pure per-half
conversion (no FSM, no queue). Use `unpack_both_rows(req)` for the
combined two-row helper. Use `step("fp8unpack", req)` for cycle-
accurate replay.

The Scala assertion `!io.req.valid || reqQ.io.enq.ready` is mirrored
as a Python `RuntimeError` raised when an enqueue arrives with a full
queue. Note that calling `step(req)` 33 times in a row does NOT trip
the overflow because the consumer also advances each cycle (one
dequeue per two cycles in steady state). To exercise the assertion
deterministically, push entries directly into `_queue` until full and
then call `step(req)`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .. import fp8_e4m3 as fp8
from ..vector_params import VectorParams


_QUEUE_DEPTH = 32


@dataclass
class FP8UnpackReq:
    xVec: list[int]                     # 16 UInt16 slots; low byte = byte 2j
    expShift: int = 0                   # int8 (post engine-side clamp)


@dataclass
class FP8UnpackResp:
    result: list[int]                   # 16 BF16 bit patterns


class FP8Unpack:
    """Mirror of `class FP8Unpack` in
    src/main/scala/atlas/vector/laneBoxes/FP8Unpack.scala.

    Visible latency = 2 (one cycle for the queue's enq→deq latency,
    one cycle for the `state := sLow` register update). The class is
    stateful: `_state`, `_input_buf`, `_exp_buf`, and `_queue` mirror
    the four pieces of register state in the Scala module.
    """

    LATENCIES: dict[str, int] = {"fp8unpack": 2}
    QUEUE_DEPTH: int = _QUEUE_DEPTH

    def __init__(self, p: VectorParams):
        self.p = p
        self._state: str = "idle"             # "idle" | "low" | "high"
        # `inputBuf = Reg(UInt(256.W))` — modelled as a flat 32-byte list.
        self._input_buf: list[int] = [0] * (2 * p.num_lanes)
        # `expBuf = Reg(SInt(8.W))` — int8 holding the post-clamp scale.
        self._exp_buf: int = 0
        # `Queue(FP8UnpackReq, 32)` with default `pipe=false flow=false`,
        # i.e. enqueue at cycle K is visible on deq starting cycle K+1.
        # We model that 1-cycle latency by appending to _queue at the END
        # of step() and consuming from _queue at the START of the next.
        self._queue: deque[FP8UnpackReq] = deque()

    def reset(self) -> None:
        self._state = "idle"
        self._input_buf = [0] * (2 * self.p.num_lanes)
        self._exp_buf = 0
        self._queue.clear()

    # ------------------------------------------------------------
    #  Pure per-half conversion helpers (no FSM, no queue).
    # ------------------------------------------------------------

    def unpack_low_row(self, req: FP8UnpackReq) -> FP8UnpackResp:
        """Convert the low half of one packed row to 16 BF16 values."""
        bytes_low, _ = self._unpack_xvec(req.xVec)
        return FP8UnpackResp(result=[
            fp8.e4m3_byte_to_bf16(b, req.expShift) for b in bytes_low
        ])

    def unpack_high_row(self, req: FP8UnpackReq) -> FP8UnpackResp:
        """Convert the high half of one packed row to 16 BF16 values."""
        _, bytes_high = self._unpack_xvec(req.xVec)
        return FP8UnpackResp(result=[
            fp8.e4m3_byte_to_bf16(b, req.expShift) for b in bytes_high
        ])

    def unpack_both_rows(
        self, req: FP8UnpackReq
    ) -> tuple[FP8UnpackResp, FP8UnpackResp]:
        """Two-row helper: convert the low and high halves of one
        packed row in one call. Bypasses the FSM entirely."""
        return self.unpack_low_row(req), self.unpack_high_row(req)

    def peek_resp(self) -> Optional[FP8UnpackResp]:
        """Return whatever `io.resp` shows right now, without
        advancing the cycle. `io.resp.valid` is combinational over
        `state`; `io.resp.bits.result` is the Mux'd low/high half
        conversion of the buffered packed row."""
        if self._state == "idle":
            return None
        bytes_view = (
            self._input_buf[: self.p.num_lanes]
            if self._state == "low"
            else self._input_buf[self.p.num_lanes :]
        )
        return FP8UnpackResp(result=[
            fp8.e4m3_byte_to_bf16(b, self._exp_buf) for b in bytes_view
        ])

    # ------------------------------------------------------------
    #  Cycle-accurate step.
    # ------------------------------------------------------------

    def step(
        self, op_name: str, req: Optional[FP8UnpackReq]
    ) -> Optional[FP8UnpackResp]:
        """Cycle-accurate step. Returns whatever `io.resp` shows at
        the START of this cycle (combinational over `_state`,
        `_input_buf`, `_exp_buf`). At the END of the cycle the FSM
        advances and any new request is enqueued (visible on the next
        cycle's dequeue, mirroring the Chisel `Queue`'s 1-cycle
        enq→deq latency).
        """
        if op_name != "fp8unpack":
            raise KeyError(
                f"FP8Unpack has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )

        # Snapshot the queue size BEFORE any cycle K update — this is
        # the value `enq.ready := !full` is computed against in Chisel.
        pre_enq_len = len(self._queue)
        if req is not None and pre_enq_len >= self.QUEUE_DEPTH:
            raise RuntimeError(
                "FP8Unpack: request queue overflow "
                "(software scheduling violated)"
            )

        # Combinational output for cycle K.
        output = self.peek_resp()

        # State transition at end-of-cycle K.
        if self._state == "idle":
            if self._queue:
                self._consume_from_queue()
                self._state = "low"
            # else: stay idle.
        elif self._state == "low":
            self._state = "high"
        else:  # self._state == "high"
            if self._queue:
                self._consume_from_queue()
                self._state = "low"
            else:
                self._state = "idle"

        # Enqueue req at end-of-cycle K (visible on deq cycle K+1).
        if req is not None:
            self._queue.append(req)

        return output

    # ------------------------------------------------------------
    #  Internal helpers.
    # ------------------------------------------------------------

    def _consume_from_queue(self) -> None:
        """Dequeue one request and load it into `_input_buf` /
        `_exp_buf`. Mirrors `inputBuf := reqQ.io.deq.bits.xVec.asUInt;
        expBuf := reqQ.io.deq.bits.expShift` in Scala."""
        next_req = self._queue.popleft()
        bytes_low, bytes_high = self._unpack_xvec(next_req.xVec)
        self._input_buf = list(bytes_low) + list(bytes_high)
        self._exp_buf = next_req.expShift

    def _unpack_xvec(
        self, x_vec: list[int]
    ) -> tuple[list[int], list[int]]:
        """Split a 16-slot packed row into 16 low-half FP8 bytes and 16
        high-half FP8 bytes. Mirrors `xVec.asUInt` followed by the
        slice patterns in `FP8Unpack.scala:60-63`.

        - xVec[0..7]  → bytes 0..15   (low half)
        - xVec[8..15] → bytes 16..31  (high half)
        - Within each slot j: bits[7:0] = byte 2j, bits[15:8] = byte 2j+1
        """
        n = self.p.num_lanes
        if len(x_vec) != n:
            raise ValueError(f"xVec must have {n} slots, got {len(x_vec)}")
        half = n // 2
        bytes_low = [0] * n
        bytes_high = [0] * n
        for j in range(half):
            slot_lo = x_vec[j] & 0xFFFF
            bytes_low[2 * j] = slot_lo & 0xFF
            bytes_low[2 * j + 1] = (slot_lo >> 8) & 0xFF
            slot_hi = x_vec[j + half] & 0xFFFF
            bytes_high[2 * j] = slot_hi & 0xFF
            bytes_high[2 * j + 1] = (slot_hi >> 8) & 0xFF
        return bytes_low, bytes_high

"""lane_boxes/col_add_vec.py — funct model of ColAddVec.scala.

The Scala module accumulates BF16 inputs through a widened recFN
adder (`AddRecFN(BF16.expWidth=8, computeSigWidth = BF16.sigWidth+16
= 24)`) and exposes the result as recoded FP, 33 bits per lane. In
hardware that's HardFloat's internal recFN; functionally it is bit-
exact equivalent to IEEE-754 binary32 with RNE rounding at every add.
The engine wrapper later converts the result to BF16 via the raw
upper-16-bit slice of the FN representation:

    fNFromRecFN(8, 24, res)(31, 16)       // engine, not lane_box

so the BF16 output is **not** re-rounded RNE-to-BF16; it is a top
slice of the FP32 bit pattern.

To avoid modelling recFN explicitly, this funct model carries the
accumulator state as raw FP32 bit patterns (uint32) — recFN ↔ FN is
lossless for finite values and AddRecFN(8, 24) is bit-exact equivalent
to IEEE-754 binary32 add (RNE), so the FP32-bit-pattern path agrees
with the RTL on every value. Convert results to BF16 at the engine
layer with `bf16_upper_half_of_fp32_bits` from `bf16_utils`.

The output register `addResultNext` is the only stateful piece: it
latches a fresh per-lane sum every cycle the request is valid AND
`isDoneReadingColSum` is false. Once the engine asserts
`isDoneReadingColSum`, the latch freezes — the engine can then read
the final accumulator value while the lane_box is still being driven
with garbage on the `aVec`/`bVec` ports.

Visible latency = 1 (the `isValidReg = RegNext(io.req.valid)` plus
the same-cycle `addResultNext` register).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .. import bf16_utils as fp
from ..vector_params import VectorParams


@dataclass
class ColAddReq:
    aVec: list[int]                          # 16 BF16 bit patterns (uint16)
    bVec: list[int] = field(default_factory=list)   # 16 FP32 bit patterns (uint32)
    isDoneReadingColSum: bool = False


@dataclass
class ColAddResp:
    result: list[int]                        # 16 FP32 bit patterns (uint32)


class ColAddVec:
    """Mirror of `class ColAddVec` in
    src/main/scala/atlas/vector/laneBoxes/ColAddVec.scala.

    Use `compute_now` for a pure `aVec + bVec` add (no latch, no
    freeze). Use `step("csum", req)` for cycle-accurate replay with
    the `addResultNext` latch and the `isDoneReadingColSum` freeze.
    """

    LATENCIES: dict[str, int] = {"csum": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        # `addResultNext` register, RegInit(0) — FP32 bit pattern of +0.0 = 0.
        self._latched: list[int] = [0] * p.num_lanes
        # `isValidReg = RegNext(io.req.valid, false.B)` — single-bit pipe.
        self._prev_valid: bool = False

    def reset(self) -> None:
        self._latched = [0] * self.p.num_lanes
        self._prev_valid = False

    def peek_result(self) -> list[int]:
        """Return a copy of the current `addResultNext` latched value
        without advancing the cycle. Mirrors the engine's
        `csum.io.req.bits.bVec := Mux(..., csum.io.resp.bits.result)`
        wiring, which is combinational — the engine reads the latched
        value at the start of cycle k and feeds it back as `bVec` for
        the same cycle. A test driver that wants to replicate the
        engine's feedback loop must call `peek_result()` first, then
        pass it as `bVec` into `step("csum", req)`."""
        return list(self._latched)

    def compute_now(self, req: ColAddReq) -> ColAddResp:
        """Pure per-lane `aVec + bVec`, with `aVec` BF16 zero-padded
        to FP32 first. Ignores `isDoneReadingColSum` (the freeze only
        affects the output latch in `step()`)."""
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes, got {len(req.aVec)}")
        if len(req.bVec) != n:
            raise ValueError(f"bVec must have {n} lanes, got {len(req.bVec)}")
        out: list[int] = []
        for i in range(n):
            a32 = fp.fp32_bits_from_bf16(req.aVec[i])
            b32 = req.bVec[i] & 0xFFFFFFFF
            out.append(fp.fp32_bits_add(a32, b32))
        return ColAddResp(result=out)

    def step(self, op_name: str, req: Optional[ColAddReq]) -> Optional[ColAddResp]:
        """Cycle-accurate step. Returns the value `addResultNext` held
        at the START of this cycle (i.e. the previous cycle's sum, or
        the frozen sum if `isDoneReadingColSum` was high), gated by
        the previous cycle's `req.valid`.

        Side effect: at the END of the cycle, `addResultNext` is
        updated unless this cycle's request has `isDoneReadingColSum`
        set, and `isValidReg` is updated to reflect this cycle's valid.
        """
        if op_name != "csum":
            raise KeyError(
                f"ColAddVec has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )

        # Compute output BEFORE updating state — output reflects the
        # PREVIOUS cycle's `addResultNext` and `isValidReg`.
        output = (
            ColAddResp(result=list(self._latched)) if self._prev_valid else None
        )

        # Update state at end-of-cycle.
        if req is None:
            self._prev_valid = False
            # Latch holds its prior value.
        else:
            self._prev_valid = True
            if not req.isDoneReadingColSum:
                self._latched = self.compute_now(req).result
            # else: freeze — `addResultNext` keeps its prior value.

        return output

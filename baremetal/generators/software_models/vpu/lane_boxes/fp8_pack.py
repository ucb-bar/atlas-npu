"""lane_boxes/fp8_pack.py — funct model of FP8Pack.scala.

`FP8Pack` is the only **2 → 1 phased** lane_box on the BF16-side of the
engine. Two consecutive `req.valid` pulses (16 BF16 lanes each) feed
the phased state machine; one `resp.valid` pulse is emitted carrying a
`Vec(16, UInt(16))` packed row (32 FP8 bytes total, two per slot).

Cycle accounting (mirrors `FP8Pack.scala:101-129`):

    cycle K   : req=row_low,  !phase  → phase=true,  lowHalf=fp8(row_low),
                                        respValidReg=false
    cycle K+1 : req=row_high, phase=true → phase=false, respValidReg=true,
                                           respBitsReg=Cat(lowHalf, fp8(row_high))
    cycle K+2 : io.resp.valid=1, io.resp.bits.result=respBitsReg

The output is **registered** (`respValidReg`, `respBitsReg`) so it
appears one cycle AFTER the second pulse, not on the same cycle. The
phase register holds across idle cycles — `req.valid=false` clears
`respValidReg` but does NOT touch `phase` or `lowHalf`.

Output layout per `FP8Pack.scala:122-125`:

    result(j)     = Cat(lowHalf (2j+1), lowHalf (2j))   for j = 0..7
    result(j + 8) = Cat(fp8Bytes(2j+1), fp8Bytes(2j))   for j = 0..7

i.e. inside each 16-bit slot, byte 2j sits in bits[7:0] and byte 2j+1
sits in bits[15:8] (low byte first within the slot).

Per-byte BF16 → E4M3 conversion lives in `fp8_e4m3.bf16_to_e4m3_byte`,
already validated against `RVFP8PackTest.goldenFp8ByteFromBf16`.

Use `convert_row(req)` for a pure per-row BF16 → FP8 conversion (16
bytes out, no latching). Use `pack_two_rows(req_low, req_high)` for
the pure 2-row math without any phase state. Use `step("fp8pack", req)`
for cycle-accurate replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .. import fp8_e4m3 as fp8
from ..vector_params import VectorParams


@dataclass
class FP8PackReq:
    xVec: list[int]                    # 16 BF16 bit patterns (uint16)
    expShift: int = 0                  # int8 (post engine-side clamp)


@dataclass
class FP8PackResp:
    result: list[int]                  # 16 UInt16 slots; low byte = byte 2j


class FP8Pack:
    """Mirror of `class FP8Pack` in
    src/main/scala/atlas/vector/laneBoxes/FP8Pack.scala.

    Visible latency = 1 (the `respValidReg` register on the second
    pulse). The class is stateful: `_phase`, `_low_half`,
    `_resp_valid`, and `_resp_bits` mirror the four Scala registers.
    """

    LATENCIES: dict[str, int] = {"fp8pack": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        # `phase = RegInit(false.B)` — false → first half, true → second.
        self._phase: bool = False
        # `lowHalf = Reg(Vec(numLanes, UInt(8.W)))` — 16 buffered FP8 bytes.
        self._low_half: list[int] = [0] * p.num_lanes
        # `respValidReg = RegInit(false.B)`.
        self._resp_valid: bool = False
        # `respBitsReg = Reg(Vec(numLanes, UInt(16.W)))`.
        self._resp_bits: list[int] = [0] * p.num_lanes

    def reset(self) -> None:
        self._phase = False
        self._low_half = [0] * self.p.num_lanes
        self._resp_valid = False
        self._resp_bits = [0] * self.p.num_lanes

    def convert_row(self, req: FP8PackReq) -> list[int]:
        """Pure per-lane BF16 → E4M3 byte conversion for one row.
        No latching, no phase, no output register. Mirrors
        `FP8Pack.scala:42-98` (the combinational `fp8Bytes` wire)."""
        n = self.p.num_lanes
        if len(req.xVec) != n:
            raise ValueError(f"xVec must have {n} lanes, got {len(req.xVec)}")
        return [fp8.bf16_to_e4m3_byte(x, req.expShift) for x in req.xVec]

    def pack_two_rows(
        self, req_low: FP8PackReq, req_high: FP8PackReq
    ) -> FP8PackResp:
        """Pure 2-row pack helper: convert both rows and assemble the
        16 UInt16 output slots, bypassing the latch entirely. Useful
        for tests that want to verify the conversion + packing math
        without driving the cycle-accurate state machine."""
        low_bytes = self.convert_row(req_low)
        high_bytes = self.convert_row(req_high)
        return FP8PackResp(result=self._assemble(low_bytes, high_bytes))

    def peek_resp(self) -> Optional[FP8PackResp]:
        """Return the current registered output without advancing the
        cycle. Mirrors a combinational read of `io.resp.valid` /
        `io.resp.bits.result`."""
        if not self._resp_valid:
            return None
        return FP8PackResp(result=list(self._resp_bits))

    def step(
        self, op_name: str, req: Optional[FP8PackReq]
    ) -> Optional[FP8PackResp]:
        """Cycle-accurate step. Returns whatever `io.resp` shows at
        the START of this cycle (i.e. the registered output that was
        latched at the end of the previous cycle).

        Side effect: at the END of the cycle, the state machine
        advances based on `req`:
          - `req is None` (req.valid=false): clear `respValidReg`,
            preserve `phase` and `lowHalf` across the idle.
          - `req` and `not phase`: convert row, latch into `lowHalf`,
            flip `phase` to true, clear `respValidReg`.
          - `req` and `phase`: convert row, pack both halves into
            `respBitsReg`, set `respValidReg=true`, flip `phase`
            back to false.
        """
        if op_name != "fp8pack":
            raise KeyError(
                f"FP8Pack has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )

        # Output reflects the PREVIOUS cycle's register state.
        output = (
            FP8PackResp(result=list(self._resp_bits))
            if self._resp_valid
            else None
        )

        # End-of-cycle state update.
        if req is None:
            # `}.otherwise { respValidReg := false.B }` — phase and
            # lowHalf are NOT touched, so a late second pulse still
            # finds the correct first-half buffer.
            self._resp_valid = False
        else:
            fp8_bytes = self.convert_row(req)
            if not self._phase:
                self._low_half = fp8_bytes
                self._phase = True
                self._resp_valid = False
            else:
                self._phase = False
                self._resp_valid = True
                self._resp_bits = self._assemble(self._low_half, fp8_bytes)

        return output

    def _assemble(self, low_bytes: list[int], high_bytes: list[int]) -> list[int]:
        """Build the 16 UInt16 output slots from 16 low-half + 16
        high-half FP8 bytes. Mirrors `FP8Pack.scala:122-125` exactly."""
        n = self.p.num_lanes
        half = n // 2
        out = [0] * n
        for j in range(half):
            out[j] = ((low_bytes[2 * j + 1] & 0xFF) << 8) | (low_bytes[2 * j] & 0xFF)
            out[j + half] = (
                ((high_bytes[2 * j + 1] & 0xFF) << 8) | (high_bytes[2 * j] & 0xFF)
            )
        return out

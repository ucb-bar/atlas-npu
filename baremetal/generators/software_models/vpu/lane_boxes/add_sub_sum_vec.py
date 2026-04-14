"""lane_boxes/add_sub_sum_vec.py — funct model of AddSubSumVec.scala.

Mirrors the laneBox at FP32 precision (computeSigWidth = BF16.sigWidth + 16
= 24, which is exactly IEEE-754 binary32). Every BF16 input is widened by
appending 16 zero bits, every adder runs in FP32, and the BF16 output is the
*raw* upper 16 bits of the FP32 result — NOT a second RNE-to-BF16 rounding
pass. This matches `fNFromRecFN(8, 24, v)(31, 16)` in the Scala module.

The rsum reduction tree pairing must match AddSubSumVec.scala bit-for-bit
because floating-point sum is order-sensitive. The Scala adders pair
(i, i+8), then (i, i+4), then (i, i+2), then (0, 1) — see the comment in
`compute_now`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .. import bf16_utils as fp
from ..vector_params import VectorParams


@dataclass
class AddSubSumReq:
    aVec: list[int]
    bVec: list[int] = field(default_factory=list)
    isSub: bool = False
    isSum: bool = False
    isDoneReadingColSum: bool = False


@dataclass
class AddSubSumResp:
    result: list[int]


class AddSubSumVec:
    """Funct mirror of `class AddSubSumVec` in
    src/main/scala/atlas/vector/laneBoxes/AddSubSumVec.scala.

    Per-op visible latencies (AddSubSumVec.scala:130-138):
        - add/sub: `isAddSubValid = isValidStage1 && !isSumStage1` → 1 cycle
        - rsum:    `isReduSumValid = isValidStage4 && isSumStage4` → 4 cycles

    `step("add", req)` and `step("rsum", req)` MUST NOT share a queue —
    submitting an add does not pop a pending rsum and vice versa.
    """

    LATENCIES: dict[str, int] = {"add": 1, "sub": 1, "rsum": 4}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: AddSubSumReq) -> AddSubSumResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes, got {len(req.aVec)}")

        if not req.isSum:
            if len(req.bVec) != n:
                raise ValueError(f"bVec must have {n} lanes for add/sub, got {len(req.bVec)}")
            f32_op = fp.fp32_bits_sub if req.isSub else fp.fp32_bits_add
            out: list[int] = []
            for i in range(n):
                a32 = fp.fp32_bits_from_bf16(req.aVec[i])
                b32 = fp.fp32_bits_from_bf16(req.bVec[i])
                s32 = f32_op(a32, b32)
                out.append(fp.bf16_upper_half_of_fp32_bits(s32))
            return AddSubSumResp(result=out)

        # Reduction sum (rsum). Pairing must match AddSubSumVec.scala:42-110:
        #   stage 0 (adders 0..7):  p0[i] = aVec[i]   + aVec[i+8]   for i in 0..7
        #   stage 1 (adders 8..11): p1[i] = p0[i]     + p0[i+4]     for i in 0..3
        #   stage 2 (adders 12..13):p2[i] = p1[i]     + p1[i+2]     for i in 0..1
        #   stage 3 (adder 14):     p3    = p2[0]     + p2[1]
        # Final scalar broadcast to all 16 lanes (see reduSumOut in Scala:134).
        widened = [fp.fp32_bits_from_bf16(x) for x in req.aVec]
        s0 = [fp.fp32_bits_add(widened[i], widened[i + 8]) for i in range(8)]
        s1 = [fp.fp32_bits_add(s0[i], s0[i + 4]) for i in range(4)]
        s2 = [fp.fp32_bits_add(s1[i], s1[i + 2]) for i in range(2)]
        s3 = fp.fp32_bits_add(s2[0], s2[1])
        bf16_result = fp.bf16_upper_half_of_fp32_bits(s3)
        return AddSubSumResp(result=[bf16_result] * n)

    def step(self, op_name: str, req: Optional[AddSubSumReq]) -> Optional[AddSubSumResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"AddSubSumVec has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

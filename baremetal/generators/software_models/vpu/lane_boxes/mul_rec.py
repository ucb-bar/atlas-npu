"""lane_boxes/mul_rec.py — funct model of MulRec.scala.

MulRec runs at BF16 precision throughout (`MulRawFN(8, 8)` + raw float
recombine via `RoundRawFNToRecFN(8, 8, 0)` with RNE). `bf16_mul` is bit-exact
for this path because BF16 inputs zero-pad cleanly into FP32, the FP32
multiply is exact (8 + 8 mantissa bits ≤ 24), and `f32_to_bf16_bits_rne`
performs the single rounding that the round-raw-to-recFN unit does.

Visible latency is 1 cycle (numIntermediateStages = 1, see MulRec.scala:60-79).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .. import bf16_utils as fp
from ..vector_params import VectorParams


@dataclass
class MulReq:
    aVec: list[int]
    bVec: list[int]


@dataclass
class MulResp:
    result: list[int]


class MulRec:
    LATENCIES: dict[str, int] = {"mul": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: MulReq) -> MulResp:
        n = self.p.num_lanes
        if len(req.aVec) != n or len(req.bVec) != n:
            raise ValueError(f"aVec/bVec must have {n} lanes")
        return MulResp(result=[fp.bf16_mul(req.aVec[i], req.bVec[i]) for i in range(n)])

    def step(self, op_name: str, req: Optional[MulReq]) -> Optional[MulResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"MulRec has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

"""lane_boxes/pair_wise_min.py — funct model of PairwiseMin.scala.

Symmetric to pair_wise_max — see that file's docstring for the
laneMask gating, freeze semantics, and signed-zero / NaN ordering
notes. Calls `compare_return_min` instead of `compare_return_max`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .. import bf16_utils as fp
from ..vector_params import VectorParams


@dataclass
class PairWiseMinReq:
    aVec: list[int]
    bVec: list[int]
    laneMask: int = 0xFFFF
    isDoneReadingColMin: bool = False


@dataclass
class PairWiseMinResp:
    result: list[int]
    laneMask: int = 0xFFFF


class PairWiseMin:
    """Mirror of `class PairWiseMin` in
    src/main/scala/atlas/vector/laneBoxes/PairwiseMin.scala.
    """

    LATENCIES: dict[str, int] = {"pairmin": 1, "cmin": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: PairWiseMinReq) -> PairWiseMinResp:
        n = self.p.num_lanes
        if len(req.aVec) != n or len(req.bVec) != n:
            raise ValueError(f"aVec/bVec must have {n} lanes")
        out: list[int] = []
        for i in range(n):
            if (req.laneMask >> i) & 1:
                out.append(fp.compare_return_min(req.aVec[i], req.bVec[i]))
            else:
                out.append(0x0000)
        return PairWiseMinResp(result=out, laneMask=req.laneMask & ((1 << n) - 1))

    def step(self, op_name: str, req: Optional[PairWiseMinReq]) -> Optional[PairWiseMinResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"PairWiseMin has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

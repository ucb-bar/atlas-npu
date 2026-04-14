"""lane_boxes/pair_wise_max.py — funct model of PairwiseMax.scala.

Per-lane bitwise max using the sign-magnitude ordering from
`VectorParam.compareReturnMax` — NOT Python's `max()`, which would
diverge on signed zeros (0x0000 vs 0x8000) and on NaN ordering.

`laneMask` follows the Scala module's gating: disabled lanes return
`0x0000` (not the input). The `isDoneReadingColMax` field is plumbed
through but ignored by `compute_now`; it only matters in cycle-accurate
simulation of the column-reduction freeze register, which the funct
model handles at the engine driver layer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .. import bf16_utils as fp
from ..vector_params import VectorParams


@dataclass
class PairWiseMaxReq:
    aVec: list[int]
    bVec: list[int]
    laneMask: int = 0xFFFF
    isDoneReadingColMax: bool = False


@dataclass
class PairWiseMaxResp:
    result: list[int]
    laneMask: int = 0xFFFF


class PairWiseMax:
    """Mirror of `class PairWiseMax` in
    src/main/scala/atlas/vector/laneBoxes/PairwiseMax.scala.

    Visible latency = 1 (single `outputVec` register on the result path).
    """

    LATENCIES: dict[str, int] = {"pairmax": 1, "cmax": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: PairWiseMaxReq) -> PairWiseMaxResp:
        n = self.p.num_lanes
        if len(req.aVec) != n or len(req.bVec) != n:
            raise ValueError(f"aVec/bVec must have {n} lanes")
        out: list[int] = []
        for i in range(n):
            if (req.laneMask >> i) & 1:
                out.append(fp.compare_return_max(req.aVec[i], req.bVec[i]))
            else:
                out.append(0x0000)
        return PairWiseMaxResp(result=out, laneMask=req.laneMask & ((1 << n) - 1))

    def step(self, op_name: str, req: Optional[PairWiseMaxReq]) -> Optional[PairWiseMaxResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"PairWiseMax has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

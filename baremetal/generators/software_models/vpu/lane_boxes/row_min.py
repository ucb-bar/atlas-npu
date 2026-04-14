"""lane_boxes/row_min.py — funct model of RowMin.scala.

Symmetric to row_max — see that file's docstring for the reduction
order, scalar broadcast, and latency notes. Calls
`compare_return_min` instead of `compare_return_max`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .. import bf16_utils as fp
from ..vector_params import VectorParams


@dataclass
class RowMinReq:
    aVec: list[int]


@dataclass
class RowMinResp:
    result: list[int]


class RowMin:
    """Mirror of `class RowMin` in
    src/main/scala/atlas/vector/laneBoxes/RowMin.scala.
    """

    LATENCIES: dict[str, int] = {"rmin": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: RowMinReq) -> RowMinResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes, got {len(req.aVec)}")
        a = req.aVec
        m8 = [fp.compare_return_min(a[2 * i], a[2 * i + 1]) for i in range(8)]
        m4 = [fp.compare_return_min(m8[2 * i], m8[2 * i + 1]) for i in range(4)]
        m2 = [fp.compare_return_min(m4[2 * i], m4[2 * i + 1]) for i in range(2)]
        m1 = fp.compare_return_min(m2[0], m2[1])
        return RowMinResp(result=[m1] * n)

    def step(self, op_name: str, req: Optional[RowMinReq]) -> Optional[RowMinResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"RowMin has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

"""lane_boxes/mov.py — funct model of Mov.scala.

Identity copy with a 1-cycle visible latency:

    resultReg := io.req.bits.aVec   when io.req.valid
    validReg  := io.req.valid
    io.resp.{valid, result} := {validReg, resultReg}

The funct model ignores `laneMask` because no engine path drives it to
anything other than 0xFFFF for Mov. If that ever changes, add a `laneMask`
field on `MovReq` and apply it here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams


@dataclass
class MovReq:
    aVec: list[int]


@dataclass
class MovResp:
    result: list[int]


class Mov:
    LATENCIES: dict[str, int] = {"mov": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: MovReq) -> MovResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes")
        return MovResp(result=[x & 0xFFFF for x in req.aVec])

    def step(self, op_name: str, req: Optional[MovReq]) -> Optional[MovResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"Mov has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

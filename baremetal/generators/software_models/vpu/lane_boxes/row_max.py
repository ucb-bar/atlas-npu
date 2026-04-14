"""lane_boxes/row_max.py — funct model of RowMax.scala.

8-4-2-1 reduction tree of bitwise max with **adjacent pairing**:
    stage 0: max8[i] = compareReturnMax(aVec[2i], aVec[2i+1])  for i in 0..7
    stage 1: max4[i] = compareReturnMax(max8[2i], max8[2i+1])  for i in 0..3
    stage 2: max2[i] = compareReturnMax(max4[2i], max4[2i+1])  for i in 0..1
    stage 3: max1    = compareReturnMax(max2[0], max2[1])

Note that this pairing is `(2i, 2i+1)` — adjacent — NOT the
`(i, i+8)` split-half pattern that AddSubSumVec.rsum uses. The two
modules picked different orders; for a max reduction the order doesn't
affect the *value* (max is associative + commutative under the bitwise
ordering), but the bit-exact unit tests still mirror the Scala order.

The scalar `max1` is broadcast across all 16 lanes
(`io.resp.bits.result := VecInit.fill(numLanes)(max1Next)`).

Visible latency = 1 (`max1Next` register).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .. import bf16_utils as fp
from ..vector_params import VectorParams


@dataclass
class RowMaxReq:
    aVec: list[int]


@dataclass
class RowMaxResp:
    result: list[int]


class RowMax:
    """Mirror of `class RowMax` in
    src/main/scala/atlas/vector/laneBoxes/RowMax.scala.
    """

    LATENCIES: dict[str, int] = {"rmax": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: RowMaxReq) -> RowMaxResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes, got {len(req.aVec)}")
        a = req.aVec
        m8 = [fp.compare_return_max(a[2 * i], a[2 * i + 1]) for i in range(8)]
        m4 = [fp.compare_return_max(m8[2 * i], m8[2 * i + 1]) for i in range(4)]
        m2 = [fp.compare_return_max(m4[2 * i], m4[2 * i + 1]) for i in range(2)]
        m1 = fp.compare_return_max(m2[0], m2[1])
        return RowMaxResp(result=[m1] * n)

    def step(self, op_name: str, req: Optional[RowMaxReq]) -> Optional[RowMaxResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"RowMax has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

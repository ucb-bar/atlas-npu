"""lane_boxes/relu.py — funct model of sp26FPUnits.Relu.

Lives in dependencies/sp26-fp-units/.../vpuFUnits/Relu.scala. Per-lane
behavior is `Mux(inVal(15), 0, inVal)` — i.e. zero out anything with the
sign bit set. Notes for parity:

  * BF16 -0 (0x8000) has sign bit 1, so it returns +0 (0x0000).
  * Negative subnormals and -inf collapse to +0.
  * NaNs whose sign bit happens to be 1 (e.g. 0xFFC0) collapse to +0.
  * Positive NaNs pass through unchanged. The Scala does not isolate NaN
    handling — it really is a pure sign-bit check.

Visible latency is 1 cycle (`reqReg`/`RegNext(io.req.valid)`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams


@dataclass
class ReluReq:
    aVec: list[int]


@dataclass
class ReluResp:
    result: list[int]


class Relu:
    LATENCIES: dict[str, int] = {"relu": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: ReluReq) -> ReluResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes")
        out = [(0 if (x & 0x8000) else (x & 0xFFFF)) for x in req.aVec]
        return ReluResp(result=out)

    def step(self, op_name: str, req: Optional[ReluReq]) -> Optional[ReluResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"Relu has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

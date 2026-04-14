"""lane_boxes/vector_load_imm.py — funct model of VectorLoadImm.scala.

Pattern fill driven by (op, imm, rowIdx). The Scala module:

    immU = imm.asUInt              // 16-bit pattern
    isRow0 = (rowIdx === 0.U)
    outVec(i) := 0.U                // default for all lanes
    switch(op) {
      vliAll: outVec(i) := immU                                  // every lane
      vliRow: when(isRow0) { outVec(i) := immU }                 // every lane on row 0
      vliCol: outVec(0) := immU                                  // lane 0 only
      vliOne: when(isRow0) { outVec(0) := immU }                 // lane 0 on row 0 only
    }

Visible latency: 1 cycle (`RegNext(io.req.valid)` + `RegEnable(outVec, ...)`).
All four sub-ops share the latch, but we keep them as separate queue keys
so cycle-accurate drivers can interleave them without crosstalk.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams
from ..vpu_op import VPUOp


_VLI_OPS = ("vliAll", "vliRow", "vliCol", "vliOne")


@dataclass
class VLIReq:
    op: str            # one of {"vliAll", "vliRow", "vliCol", "vliOne"}
    imm: int           # 16-bit pattern (signed Scala SInt, taken as unsigned bits)
    rowIdx: int = 0


@dataclass
class VLIResp:
    result: list[int]


class VectorLoadImm:
    LATENCIES: dict[str, int] = {op: 1 for op in _VLI_OPS}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: VLIReq) -> VLIResp:
        n = self.p.num_lanes
        if req.op not in self.LATENCIES:
            raise ValueError(
                f"unknown VLI op {req.op!r}; expected one of {_VLI_OPS}"
            )
        # Scala SInt(16.W).asUInt → keep low 16 bits, regardless of sign.
        imm = req.imm & 0xFFFF
        is_row0 = (req.rowIdx == 0)
        out = [0] * n
        if req.op == "vliAll":
            out = [imm] * n
        elif req.op == "vliRow":
            if is_row0:
                out = [imm] * n
        elif req.op == "vliCol":
            out[0] = imm
        elif req.op == "vliOne":
            if is_row0:
                out[0] = imm
        return VLIResp(result=out)

    def step(self, op_name: str, req: Optional[VLIReq]) -> Optional[VLIResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"VectorLoadImm has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()


# Sanity: the four VLI ops here line up with the VPUOp enum values.
assert set(_VLI_OPS) == {VPUOp.vliAll.name, VPUOp.vliRow.name, VPUOp.vliCol.name, VPUOp.vliOne.name}

"""lane_boxes/square_cube_vec.py — funct model of SquareCubeVec.scala.

Two ops on one port:
  - `square` (`isCube=False`): per-lane `aVec[i]² → BF16`
  - `cube`   (`isCube=True`):  per-lane `aVec[i]³ → BF16`

The Scala wraps a `squareCube(x, isCube)` helper from the `VectorParam`
trait (`laneBoxes/VectorParam.scala`). It forms x² or x³ exactly and
rounds once to BF16 with HardFloat, as vmul rounds: nearest even, with
subnormal inputs and results; a NaN gives 0x7FC0; ±0 and ±inf follow
the IEEE signs (square ≥ +0, cube keeps the sign of x). x² and x³ of a
BF16 x are exact in a Python float, so the model rounds that value.

Lane masking (`SquareCubeVec.scala:37-40`):
  - `laneEnable = laneMask & valid`. Disabled lanes return 0x0000.

Visible latency = 1 (`isValid = RegNext(io.req.valid)` and
`resultNext = RegNext(result)`).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .. import bf16_utils as fp
from ..vector_params import VectorParams


@dataclass
class SquareCubeReq:
    aVec: list[int]                     # 16 BF16 bit patterns
    isCube: bool = False                # False → square, True → cube
    laneMask: int = 0xFFFF              # numLanes-bit mask, 1=enabled


@dataclass
class SquareCubeResp:
    result: list[int]                   # 16 BF16 bit patterns
    laneMask: int = 0xFFFF              # passed through (RegNext'd) by the RTL


_BF16_POS_INF = 0x7F80


def square_cube_bf16(bf16: int, is_cube: bool) -> int:
    """Pure per-lane `squareCube` mirror: x² or x³ rounded once to BF16."""
    x = fp.bf16_bits_to_f32(bf16)
    v = x * x * x if is_cube else x * x
    if v != v:
        return 0x7FC0
    sign = 0x8000 if math.copysign(1.0, v) < 0 else 0
    a = abs(v)
    if a == 0.0 or math.isinf(a):
        return sign | (_BF16_POS_INF if a else 0)
    e = max(math.frexp(a)[1] - 1, -126)          # leading exponent; the quantum is 2^(e - 7)
    n = round(math.ldexp(a, 7 - e))               # exact; round() ties to even
    return sign | min(((e + 126) << 7) + n, _BF16_POS_INF)


class SquareCubeVec:
    """Mirror of `class SquareCubeVec` in
    src/main/scala/atlas/vector/laneBoxes/SquareCubeVec.scala.

    Two op names share the queue tree because the Scala module has
    one valid port and dispatches via `isCube`. We split into two
    op names so back-to-back square / cube don't collide on the
    same latency queue."""

    LATENCIES: dict[str, int] = {"square": 1, "cube": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: SquareCubeReq) -> SquareCubeResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes, got {len(req.aVec)}")
        out: list[int] = []
        for i in range(n):
            lane_en = bool((req.laneMask >> i) & 1)
            if not lane_en:
                out.append(0x0000)
            else:
                out.append(square_cube_bf16(req.aVec[i], req.isCube))
        return SquareCubeResp(result=out, laneMask=req.laneMask & 0xFFFF)

    def step(
        self, op_name: str, req: Optional[SquareCubeReq]
    ) -> Optional[SquareCubeResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"SquareCubeVec has no op {op_name!r}; "
                f"valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

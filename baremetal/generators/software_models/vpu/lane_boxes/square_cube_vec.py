"""lane_boxes/square_cube_vec.py — funct model of SquareCubeVec.scala.

Two ops on one port:
  - `square` (`isCube=False`): per-lane `aVec[i]² → BF16`
  - `cube`   (`isCube=True`):  per-lane `aVec[i]³ → BF16`

The Scala wraps a `squareCube(sign, exp, fra, isCube)` helper from the
`VectorParam` trait (`laneBoxes/VectorParam.scala:6-96`). That helper
does NOT route through HardFloat — it builds the explicit 8-bit
mantissa, multiplies it as an integer, recovers the leading-1 position
to renormalize, adjusts the exponent by hand, then assembles a BF16
word. The funct model mirrors that integer dataflow exactly so the
half-ULP rounding edge cases stay bit-equal with the RTL.

Sign convention:
  - `square` always returns sign 0 (square is non-negative).
  - `cube` preserves the input sign.

Special cases (all from `VectorParam.scala:78-92`):
  - Input zero, subnormal, or NaN → flush to signed zero. NaN is NOT
    propagated; it is treated as "junk → zero", which matches the
    Scala `when (... || isInputNaN ...) { result := zero }`.
  - Input +inf → +inf (square) or signed inf (cube).
  - Result-exponent underflow (`adjustedExp ≤ 0`) → signed zero.
  - Result-exponent overflow (`adjustedExp ≥ 255`) → signed inf.

Lane masking (`SquareCubeVec.scala:37-40`):
  - `laneEnable = laneMask & valid`. Disabled lanes return 0x0000.

Visible latency = 1 (`isValid = RegNext(io.req.valid)` and
`resultNext = RegNext(result)`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

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
_BF16_NEG_INF = 0xFF80


def square_cube_bf16(bf16: int, is_cube: bool) -> int:
    """Pure per-lane `squareCube` mirror of
    `VectorParam.scala:6-96`. Builds the explicit 8-bit mantissa,
    multiplies, finds the leading 1, normalizes, fills the exponent
    by hand, and applies the special-case lattice."""
    bf16 &= 0xFFFF
    sign = (bf16 >> 15) & 1
    exp = (bf16 >> 7) & 0xFF
    fra = bf16 & 0x7F

    is_input_zero = (exp == 0 and fra == 0)
    is_input_sub = (exp == 0 and fra != 0)
    is_input_inf = (exp == 0xFF and fra == 0)
    is_input_nan = (exp == 0xFF and fra != 0)

    real_exp = exp - 127
    mant = (1 << 7) | fra  # 8-bit `1.fra`

    if not is_cube:
        # ---- Square path ----
        prod = mant * mant                              # in [16384, 65536)
        extra_exp = 1 if (prod >> 15) & 1 else 0        # bit 15 = squareFracWidth-1
        adjusted_exp = (real_exp << 1) + extra_exp + 127

        if is_input_zero or is_input_sub or is_input_nan or adjusted_exp <= 0:
            return 0x0000                               # +0 (square is unsigned)
        if is_input_inf or adjusted_exp >= 255:
            return _BF16_POS_INF                        # +inf

        highest = prod.bit_length() - 1                 # always 14 or 15 here
        shift = highest - 7 if highest > 7 else 0
        mant_pre = prod >> shift
        frac = mant_pre & 0x7F
        return (0 << 15) | ((adjusted_exp & 0xFF) << 7) | frac

    # ---- Cube path ----
    prod = mant * mant * mant                           # in [2^21, 2^24)
    if (prod >> 23) & 1:                                # bit 23 = cubeFracWidth-1
        extra_exp = 2
    elif (prod >> 22) & 1:                              # bit 22 = cubeFracWidth-2
        extra_exp = 1
    else:
        extra_exp = 0
    adjusted_exp = real_exp * 3 + extra_exp + 127

    if is_input_zero or is_input_sub or is_input_nan or adjusted_exp <= 0:
        return (sign << 15) & 0xFFFF                    # signed zero
    if is_input_inf or adjusted_exp >= 255:
        return ((sign << 15) | _BF16_POS_INF) & 0xFFFF  # signed inf

    highest = prod.bit_length() - 1                     # always 21, 22, or 23
    shift = highest - 7 if highest > 7 else 0
    mant_pre = prod >> shift
    frac = mant_pre & 0x7F
    return ((sign << 15) | ((adjusted_exp & 0xFF) << 7) | frac) & 0xFFFF


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

"""lane_boxes/sqrt.py — funct model of `Sqrt.scala`.

Single-port, latency-1 LUT-backed square root. Wraps
`SqrtLUT(numLanes, addrBits=sigWidth-1, m=BF16T.lutValM, n=BF16T.lutValN)`
and delays the valid bit by one cycle.

Per-lane data flow (`Sqrt.scala:34-44`):
  - `lutInputExp` = `aVec[i][14:7]`
  - `oddExp`      = `!aVec[i][7]`             (LSB of biased exp == 0)
  - `lutInputMant`= `aVec[i][6:0]`
  - `lut.io.ren`  = `laneMask[i] & req.valid`

Inside `SqrtLUT.scala:25-37`:
  - `shiftedFull = ((lut[raddr] * maxVal) >> n)` where
    `maxVal = round(sqrt(max) * 2^n)`
  - `shifted    = shiftedFull[(m+n-1):0]` (truncated to m+n bits)
  - `base       = oddExp ? shifted : lut[raddr]`
  - `out        = lutFixedToBf16Sqrt(base, exp, raddr, m, n, false)`

The `oddExp` branch handles odd unbiased exponents: for `x = 1.frac *
2^k` with k odd, `sqrt(x) = sqrt(2) * sqrt(1.frac) * 2^((k-1)/2)`, so
the LUT entry (which is `sqrt(r)` for `r in [1,2)`) is multiplied by
`sqrt(2)` once before being handed to the formatter.

Note that `oddExp = !a(exponentLowBitIndex)` flags an *even* biased
exponent. Since the BF16 bias is 127 (odd), even biased exp ↔ odd
real exp, which matches the name.

Sign handling: Sqrt does NOT propagate the BF16 input sign — `neg` is
hardcoded `false.B` in the helper call. Negative finite inputs are
treated as their magnitude (the helper's special-case lattice doesn't
flush them). This matches the RTL even though IEEE sqrt(-x) → NaN.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams
from ..lut_sources.lut_params import lut_fixed_to_bf16_sqrt
from ..lut_sources.lut_tables import gen_sqrt_lut, sqrt_max_val


_SQRT_ADDR_BITS = 7
_SQRT_M = 1
_SQRT_N = 16
_SQRT_MAX = 2.0


@dataclass
class SqrtReq:
    aVec: list[int]
    laneMask: int = 0xFFFF
    roundingMode: int = 0               # ignored


@dataclass
class SqrtResp:
    result: list[int]


def sqrt_bf16(bf16: int, lut: list[int], max_val: int) -> int:
    """Pure per-lane sqrt mirror of `SqrtLUT` + `lutFixedToBf16Sqrt`."""
    bf16 &= 0xFFFF
    exp = (bf16 >> 7) & 0xFF
    mant = bf16 & 0x7F
    odd_exp = ((bf16 >> 7) & 1) == 0   # !exponentLowBit
    raw = lut[mant]
    if odd_exp:
        # `((lut[raddr] * maxVal) >> n)` truncated to (m+n) bits.
        shifted_full = (raw * max_val) >> _SQRT_N
        base = shifted_full & ((1 << (_SQRT_M + _SQRT_N)) - 1)
    else:
        base = raw
    return lut_fixed_to_bf16_sqrt(
        base, exp, mant, lut_val_m=_SQRT_M, lut_val_n=_SQRT_N, neg=0
    )


class Sqrt:
    """Mirror of `class Sqrt` in
    src/main/scala/atlas/vector/laneBoxes/Sqrt.scala. Latency 1."""

    LATENCIES: dict[str, int] = {"sqrt": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._lut = gen_sqrt_lut(
            addr_bits=_SQRT_ADDR_BITS, m=_SQRT_M, n=_SQRT_N,
            minimum=1.0, maximum=_SQRT_MAX,
        )
        self._max_val = sqrt_max_val(n=_SQRT_N, maximum=_SQRT_MAX)
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: SqrtReq) -> SqrtResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes, got {len(req.aVec)}")
        out: list[int] = []
        for i in range(n):
            lane_en = bool((req.laneMask >> i) & 1)
            if not lane_en:
                out.append(0x0000)
            else:
                out.append(sqrt_bf16(req.aVec[i], self._lut, self._max_val))
        return SqrtResp(result=out)

    def step(self, op_name: str, req: Optional[SqrtReq]) -> Optional[SqrtResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"Sqrt has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

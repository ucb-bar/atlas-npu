"""lane_boxes/log.py — funct model of `Log.scala`.

Single-port, latency-1 LUT-backed log2. Wraps
`LogLUT(numLanes, addrBits=sigWidth-1, m=9, n=BF16T.lutValN)` —
the `m=9` is **hardcoded** in `Log.scala:31`, NOT pulled from
`BF16T.lutValM=1`, because the downstream combine path needs slack:
  `realExpAsQmn = absExp << n`, and `absExp` can be up to 127, which
  requires 7 integer bits.

Per-lane data flow (`Log.scala:34-44`):
  - `lutInputExp` = `aVec[i][14:7]`
  - `lutInputMant`= `aVec[i][6:0]`        (LUT address)
  - `lut.io.ren`  = `laneMask[i] & req.valid`

Inside `LogLUT.scala:24-36`:
  - `bias = 127`
  - `isNeg = exp < bias`
  - `absExp = isNeg ? (bias - exp) : (exp - bias)`
  - `realExpAsQmn = absExp << n`           (Q(m,n), zero-frac integer)
  - `base = isNeg ? (realExpAsQmn - lut[raddr]) : (realExpAsQmn + lut[raddr])`
  - `out  = lutFixedToBf16Log(base, exp, raddr, m, n, isNeg)`

The combined value `base` represents the magnitude of `log2(x)`; the
sign of the result is delivered via `isNeg`. Note that this means the
LogLUT does NOT consume the input BF16 sign — a negative input is
treated as `log2(|x|)`, then sign-flipped only if `|x| < 1`. Matches
the RTL.

The two `realExpAsQmn ± lut[raddr]` branches model:
  - `x in [1, 2)`: `log2(x) = log2(1.frac)` → table entry, positive
  - `x in [2, 4)`: `log2(x) = 1 + log2(0.5*x)` → 1 + table, positive
  - `x in [0.5, 1)`: `log2(x) = -1 + log2(2x)` → -(1 - table)
  - etc.

Special cases (from `lutFixedToBf16Log`):
  - input NaN OR LUT raw value == 0 → signed zero
    (`x == 1.0` exactly → table value 0 → result is "-0" or "+0",
    encoded with `neg=isNeg` flag)
  - input zero / subnormal / inf / overflow → signed inf
  - normal → assembled BF16 word
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams
from ..lut_sources.lut_params import lut_fixed_to_bf16_log
from ..lut_sources.lut_tables import gen_log_lut


_LOG_ADDR_BITS = 7
_LOG_M = 9         # HARDCODED in Log.scala:31 (NOT BF16T.lutValM)
_LOG_N = 16        # BF16T.lutValN


@dataclass
class LogReq:
    aVec: list[int]
    laneMask: int = 0xFFFF


@dataclass
class LogResp:
    result: list[int]


def log_bf16(bf16: int, lut: list[int]) -> int:
    """Pure per-lane log2 mirror of `LogLUT` + `lutFixedToBf16Log`."""
    bf16 &= 0xFFFF
    exp = (bf16 >> 7) & 0xFF
    mant = bf16 & 0x7F
    bias = 127

    is_neg = exp < bias
    abs_exp = (bias - exp) if is_neg else (exp - bias)
    real_exp_qmn = abs_exp << _LOG_N
    raw = lut[mant]
    base = (real_exp_qmn - raw) if is_neg else (real_exp_qmn + raw)
    base &= (1 << (_LOG_M + _LOG_N)) - 1   # truncate to (m+n) bits

    return lut_fixed_to_bf16_log(
        base, exp, mant, lut_val_m=_LOG_M, lut_val_n=_LOG_N, neg=int(is_neg)
    )


class Log:
    """Mirror of `class Log` in
    src/main/scala/atlas/vector/laneBoxes/Log.scala. Latency 1."""

    LATENCIES: dict[str, int] = {"log": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._lut = gen_log_lut(
            addr_bits=_LOG_ADDR_BITS, m=_LOG_M, n=_LOG_N,
            minimum=1.0, maximum=2.0,
        )
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: LogReq) -> LogResp:
        n = self.p.num_lanes
        if len(req.aVec) != n:
            raise ValueError(f"aVec must have {n} lanes, got {len(req.aVec)}")
        out: list[int] = []
        for i in range(n):
            lane_en = bool((req.laneMask >> i) & 1)
            if not lane_en:
                out.append(0x0000)
            else:
                out.append(log_bf16(req.aVec[i], self._lut))
        return LogResp(result=out)

    def step(self, op_name: str, req: Optional[LogReq]) -> Optional[LogResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"Log has no op {op_name!r}; valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

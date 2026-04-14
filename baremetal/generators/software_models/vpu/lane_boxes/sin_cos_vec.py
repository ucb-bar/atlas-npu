"""lane_boxes/sin_cos_vec.py — funct model of `SinCosVec.scala`.

`SinCosVec` is a single-port, latency-1 lane_box that handles both
`sin` and `cos` via a `req.bits.cos` flag. The algorithm is the
"quadrant fold + symmetric LUT + linear interpolation" pattern from
`SinCos.scala` (the standalone fp-units module), adapted to the
register layout used inside the vector engine. The IO bundle differs:
`SinCosVec` is a `Valid` (no `tag` / `whichBank`) and there's only
one intermediate register stage.

Per-lane data flow (`SinCosVec.scala:38-115`):

  Stage 0 (combinational):
    1. `bf16ToQmnTimesTwoOverPi(x)` → 22-bit signed Q(9, 12) value.
    2. `nVec = qmnVec[qmnN-1:0]`           (12-bit fractional part)
    3. `quadrantBits = qmnVec[qmnN+1:qmnN]` (2-bit integer part)
    4. `isNeg`, `isCos` derived from quadrant + `cos` flag (table below).
    5. LUT addresses: `raddr = nVec[qmnN-1:rLowBits]` where
       `rLowBits = qmnN - lutAddrBits = 12 - 5 = 7`.
    6. Latch into `commonState[0]` (qmnN, isCos, valid) AND into
       `maskLaneNext` registers (isNeg, isCos for stage 1).
    7. The LUT module ALSO has a register stage on its outputs.

  Stage 1 (combinational, on cycle K+1):
    1. `r_lower = isCos ? (1 << rLowBits) - r[rLowBits-1:0]
                        : r[rLowBits-1:0]`
    2. `delta   = isCos ? (y0 - y1) : (y1 - y0)`  (unsigned, wraps mod 2^17)
    3. `basey   = isCos ? y1 : y0`
    4. `interp  = ((delta * r_lower) >> rLowBits) + basey`,
       truncated to `lutValM + lutValN = 17` bits.
    5. `resultBF16 = lutFixedToBf16(interp, m=1, n=16, neg=false)`
    6. `resultFinal = isNeg ? (resultBF16 ^ 0x8000) : resultBF16`

Quadrant table (matches `SinCosVec.scala:55-62`):

  cos request:                         sin request:
    Q0: isCos=true,  isNeg=false       Q0: isCos=false, isNeg=false
    Q1: isCos=false, isNeg=true        Q1: isCos=true,  isNeg=false
    Q2: isCos=true,  isNeg=true        Q2: isCos=false, isNeg=true
    Q3: isCos=false, isNeg=false       Q3: isCos=true,  isNeg=true

LUT symmetry trick (`SinCosLUT.scala:46-56`):
  - `is_sin_top_addr = (raddr == maxAddr) && !isCos` →
        `y1 := scaled = round(2^n)` instead of out-of-range `lut[32]`.
  - `is_cos_bot_addr = (raddr == 0) && isCos` →
        `y0 := scaled` instead of out-of-range `lut[maxAddr+1]`.
  - The cos branch indexes the LUT in reverse (`maxAddr - a + 1`,
    `maxAddr - a`) so that a single sin-only table feeds both ops.

Lane masking: `maskedXVec[i] = laneEnable[i] ? x : 0`. Disabled lanes
feed 0 through the pipeline, but the funct model returns 0x0000 for
those slots in `compute_now` rather than computing through. The RTL
holds the previous register value via `maskLane`/`maskLaneNext`; we
diverge from that here for simplicity (tests don't exercise it).

Visible latency = 1 cycle (the `commonState[0]` and LUT registers
both latch on the same cycle K rising edge; outputs appear on cycle
K+1 from a request driven on cycle K).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..vector_params import VectorParams
from ..lut_sources.lut_params import (
    bf16_to_qmn_times_two_over_pi,
    lut_fixed_to_bf16,
)
from ..lut_sources.lut_tables import gen_sin_cos_lut, round_half_up
import math


_QMN_M = 9
_QMN_N = 12
_LUT_ADDR_BITS = 5
_LUT_VAL_M = 1
_LUT_VAL_N = 16
_R_LOW_BITS = _QMN_N - _LUT_ADDR_BITS    # 7
_MAX_ADDR = (1 << _LUT_ADDR_BITS) - 1    # 31
_LUT_VAL_W = _LUT_VAL_M + _LUT_VAL_N     # 17
_LUT_VAL_MASK = (1 << _LUT_VAL_W) - 1


@dataclass
class SinCosVecReq:
    xVec: list[int]                     # 16 BF16 bit patterns, ideally in [0, 2pi]
    cos: bool = False                   # False → sin, True → cos
    laneMask: int = 0xFFFF


@dataclass
class SinCosVecResp:
    result: list[int]                   # 16 BF16 bit patterns


def _quadrant_table(quadrant: int, is_cos_req: bool) -> tuple[bool, bool]:
    """Return `(is_cos, is_neg)` for one lane based on quadrant bits."""
    if is_cos_req:
        is_cos = quadrant in (0, 2)
        is_neg = quadrant in (1, 2)
    else:
        is_cos = quadrant in (1, 3)
        is_neg = quadrant in (2, 3)
    return is_cos, is_neg


def _lut_pair(
    raddr: int, is_cos: bool, lut: list[int], scaled: int
) -> tuple[int, int]:
    """Compute `(y0, y1)` from the LUT for one lane.

    Mirrors `SinCosLUT.scala:46-56`. The cos branch reverses the index
    so the same sin-only LUT serves both ops; the two endpoint cases
    that would index past the LUT (`maxAddr+1`) are clamped to
    `scaled = round(2^n)` (the value for sin(pi/2) = cos(0) = 1)."""
    is_cos_bot = is_cos and (raddr == 0)
    is_sin_top = (not is_cos) and (raddr == _MAX_ADDR)

    if is_cos_bot:
        y0 = scaled
    else:
        # `raddrReal = isCos ? (maxAddr - raddr + 1) : raddr`. With
        # 5-bit addressing the `+1` wraps at 32 = 0 mod 32, but the
        # `is_cos_bot` early-out catches that case before we reach here.
        raddr_real = (_MAX_ADDR - raddr + 1) if is_cos else raddr
        y0 = lut[raddr_real & _MAX_ADDR]

    if is_sin_top:
        y1 = scaled
    else:
        raddr_nxt = (_MAX_ADDR - raddr) if is_cos else (raddr + 1)
        y1 = lut[raddr_nxt & _MAX_ADDR]

    return y0, y1


def sin_cos_bf16(
    bf16: int,
    is_cos_req: bool,
    lut: list[int],
    scaled: int,
) -> int:
    """Pure per-lane sin/cos mirror.

    Implements the entire `SinCosVec.scala` data path collapsed to a
    single function, with the register stages flattened. Caller is
    responsible for masking out disabled lanes (we treat any disabled
    lane as a fixed 0x0000 output, see module docstring).
    """
    qmn = bf16_to_qmn_times_two_over_pi(bf16, qmn_m=_QMN_M, qmn_n=_QMN_N)
    # `qmn` is a signed Python int in [-2^21, 2^21). The bottom qmnN
    # bits are the fractional position; the next 2 bits are the
    # quadrant. For inputs in [0, 2pi] post-multiply by 2/pi, `qmn`
    # is in [0, 4*2^qmnN), so quadrant ∈ {0, 1, 2, 3}.
    n_val = qmn & ((1 << _QMN_N) - 1)
    quadrant = (qmn >> _QMN_N) & 0x3

    is_cos, is_neg = _quadrant_table(quadrant, is_cos_req)

    raddr = (n_val >> _R_LOW_BITS) & _MAX_ADDR
    r_lower_raw = n_val & ((1 << _R_LOW_BITS) - 1)

    y0, y1 = _lut_pair(raddr, is_cos, lut, scaled)

    if is_cos:
        r_lower = (1 << _R_LOW_BITS) - r_lower_raw
        delta = (y0 - y1) & _LUT_VAL_MASK
        basey = y1
    else:
        r_lower = r_lower_raw
        delta = (y1 - y0) & _LUT_VAL_MASK
        basey = y0

    interp = (((delta * r_lower) >> _R_LOW_BITS) + basey) & _LUT_VAL_MASK
    bf16_pos = lut_fixed_to_bf16(
        interp, lut_val_m=_LUT_VAL_M, lut_val_n=_LUT_VAL_N, neg=0
    )
    if is_neg:
        return (bf16_pos ^ 0x8000) & 0xFFFF
    return bf16_pos & 0xFFFF


class SinCosVec:
    """Mirror of `class SinCosVec` in
    src/main/scala/atlas/vector/laneBoxes/SinCosVec.scala.

    Two op names share the queue tree because the Scala module has
    one valid port and dispatches via `cos`. Splitting into "sin"
    and "cos" prevents back-to-back requests from colliding on the
    same latency queue."""

    LATENCIES: dict[str, int] = {"sin": 1, "cos": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._lut = gen_sin_cos_lut(
            addr_bits=_LUT_ADDR_BITS, m=_LUT_VAL_M, n=_LUT_VAL_N,
            minimum=0.0, maximum=math.pi / 2,
        )
        # `scaled = BigInt(math.round(1 << n)).U((m+n).W)` — the value
        # representing sin(pi/2) = cos(0) = 1.0 in (m+n)-bit Q.
        self._scaled = round_half_up(1 << _LUT_VAL_N) & _LUT_VAL_MASK
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: SinCosVecReq) -> SinCosVecResp:
        n = self.p.num_lanes
        if len(req.xVec) != n:
            raise ValueError(f"xVec must have {n} lanes, got {len(req.xVec)}")
        out: list[int] = []
        for i in range(n):
            lane_en = bool((req.laneMask >> i) & 1)
            if not lane_en:
                out.append(0x0000)
            else:
                out.append(
                    sin_cos_bf16(req.xVec[i], req.cos, self._lut, self._scaled)
                )
        return SinCosVecResp(result=out)

    def step(
        self, op_name: str, req: Optional[SinCosVecReq]
    ) -> Optional[SinCosVecResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"SinCosVec has no op {op_name!r}; "
                f"valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

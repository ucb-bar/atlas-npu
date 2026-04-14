"""lane_boxes/tanh_rec.py — funct model of `TanhRec.scala` + per-lane
`Tanh` from `sp26-fp-units/vpuFUnits/TanhBlock.scala`.

Mirrors the two-module path in the RTL:

  `TanhRec.scala:52` instantiates a single shared
  `TanhLUT(ports=numLanes, addrBits=BF16T.lutAddrBits, m=1, n=BF16T.lutValN)`
  and per-lane `Tanh(BF16T)` modules. The per-lane `Tanh` module
  (`TanhBlock.scala`) does the fixed-point decode, LUT interpolation,
  and BF16 recode in two cycles:

    Cycle 1 — decode:
      1. BF16 → Q(3, 16) magnitude fixed-point (`xFixed`, 19 bits wide).
      2. `addr = xFixed[n+intBitsToMap-1 : n+intBitsToMap-addrBits]`
         = `xFixed[17:13]` (top 5 bits, 2 integer + top 3 fractional).
      3. `alpha = xFixed[fracBits-1:0] = xFixed[12:0]`
         (bottom 13 bits = interpolation fraction).
      4. `isSaturated = (trueExp >= 2)` — i.e., |x| >= 4.0.
      5. `safeAddr = isSaturated ? maxAddr : addr`.
      6. `isZero = rawExp == 0 && mantissa == 0`.

    Cycle 2 — interp + recode:
      1. `y0 = lut[safeAddr]`, `y1 = lut[safeAddr+1]` (clamped at maxAddr).
      2. `yDiff = max(y1 - y0, 0)` (guard against LUT-rounding noise).
      3. `interp = y0 + (yDiff * alpha) >> fracBits`, truncated to
         `n+1 = 17` bits.
      4. `magFixed = isSaturated ? oneFixed : interp` where
         `oneFixed = 1 << n = 0x10000` (= 1.0 in Q(1, 16)).
      5. Magnitude fixed-point → BF16: find leading-1, compute exponent
         as `msbPos - n + bias`, normalize, then RNE-round the mantissa
         to `sigWidth - 1 = 7` fraction bits (G/R/S = bits 8 / 7 / 6..0
         of the normalized (n+1)-wide value). Mantissa carry bumps exp.
      6. Zero / underflow forwarding: if input was ±0 (or the magnitude
         collapsed to 0), force rawExp=0 and mantissa=0; sign is
         always re-attached from the input.

Saturation / range behavior:
  - |x| >= 4.0 → ±1.0 (sign from input).
  - NaN / ±inf are caught by the saturation branch (`trueExp >= 2`),
    so they also map to ±1.0 — this matches RTL; it is NOT IEEE
    `tanh(NaN) = NaN`. The Chisel module does not special-case NaN.
  - Subnormals / tiny inputs collapse through the shift so xFixed = 0,
    addr = 0, alpha = 0 → mag_fixed = 0 → signed zero out.

Visible latency per `TanhRec.scala:61` is 1 cycle (one `commonState`
register stage). The per-lane `Tanh` uses `Pipe(valid, _, 1)` to delay
its own decode-stage data alongside the LUT's 1-cycle read, so the
end-to-end `TanhRec` call is a single register stage at the funct-model
boundary — matching the other LUT-backed lane boxes (Rcp, Sqrt, Log,
SinCosVec).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from ..lut_sources.lut_tables import gen_tanh_lut
from ..vector_params import VectorParams


# BF16 + Tanh module constants. These mirror `AtlasFPType.BF16` in
# `sp26-fp-units/common.scala:132-143` and the hardcoded field widths
# inside `TanhBlock.scala:16-25`. The per-lane `Tanh` uses
# `fptype.lutValN` as its internal Q(3, n) fraction width, `lutAddrBits`
# as the LUT address width, and `intBitsToMap = 2` (the number of
# integer bits folded into the LUT address).
_W = 16                          # BF16 word width
_SIG_W = 8                       # sigWidth (1 hidden + 7 explicit)
_EXP_W = 8
_BIAS = 127
_N = 16                          # BF16T.lutValN → xFixed fraction bits
_ADDR_BITS = 5                   # BF16T.lutAddrBits → 32-entry LUT
_INT_BITS_TO_MAP = 2             # hardcoded in TanhBlock.scala:71
_FRAC_BITS = _N + _INT_BITS_TO_MAP - _ADDR_BITS  # 13
_X_FIXED_WIDTH = _N + 3          # 19 bits
_X_FIXED_MASK = (1 << _X_FIXED_WIDTH) - 1
_MAG_FIXED_MASK = (1 << (_N + 1)) - 1   # 17-bit magnitude mask
_ONE_FIXED = 1 << _N             # 0x10000 = 1.0 in Q(1, 16)
_MAX_ADDR = (1 << _ADDR_BITS) - 1
_FRAC_W = _SIG_W - 1             # 7 BF16 mantissa bits


def tanh_bf16(bits: int, lut: list[int]) -> int:
    """Bit-exact mirror of `class Tanh` in `TanhBlock.scala`.

    `lut` must be the 32-entry table produced by `gen_tanh_lut` with
    the BF16 defaults (`addr_bits=5, m=1, n=16, min=0.0, max=4.0`) —
    i.e., identical to the `TanhLUT` instance wired up in
    `TanhRec.scala:52`.
    """
    bits &= 0xFFFF
    sign = (bits >> (_W - 1)) & 1
    raw_exp = (bits >> (_SIG_W - 1)) & ((1 << _EXP_W) - 1)
    mantissa = bits & ((1 << (_SIG_W - 1)) - 1)

    is_zero = (raw_exp == 0) and (mantissa == 0)

    hidden_bit = 0 if raw_exp == 0 else 1
    significand = (hidden_bit << (_SIG_W - 1)) | mantissa  # 8 bits

    true_exp = (1 - _BIAS) if raw_exp == 0 else (raw_exp - _BIAS)
    shift_amt = true_exp + (_N - (_SIG_W - 1))
    if shift_amt >= 0:
        x_fixed = (significand << shift_amt) & _X_FIXED_MASK
    else:
        x_fixed = (significand >> -shift_amt) & _X_FIXED_MASK

    is_saturated = (true_exp >= 2)

    addr = (x_fixed >> _FRAC_BITS) & _MAX_ADDR
    safe_addr = _MAX_ADDR if is_saturated else addr
    alpha = x_fixed & ((1 << _FRAC_BITS) - 1)

    y0 = lut[safe_addr]
    # TanhLUT.scala:41 — "raddrNxt" clips at maxAddr rather than wrapping.
    next_addr = safe_addr if safe_addr == _MAX_ADDR else safe_addr + 1
    y1 = lut[next_addr]

    y_diff = y1 - y0 if y1 >= y0 else 0
    interp = y0 + ((y_diff * alpha) >> _FRAC_BITS)

    mag_fixed = _ONE_FIXED if is_saturated else (interp & _MAG_FIXED_MASK)

    # PriorityEncoder(Reverse(magFixed)) counts leading zeros from the
    # top of the (n+1)-wide value. For the all-zero input, Chisel's
    # PriorityEncoder returns 0 by default — but that path is masked
    # out by the `mag_fixed == 0` guard on the final exponent/mantissa.
    if mag_fixed == 0:
        leading_zeros = 0
    else:
        leading_zeros = (_N + 1) - mag_fixed.bit_length()

    msb_pos = _N - leading_zeros
    exp_calc = msb_pos - _N + _BIAS
    underflow = (exp_calc <= 0)

    normalized = (mag_fixed << leading_zeros) & _MAG_FIXED_MASK

    frac_lo = _N - _FRAC_W          # 9
    frac_pre = (normalized >> frac_lo) & ((1 << _FRAC_W) - 1)
    guard_bit = (normalized >> (frac_lo - 1)) & 1            # bit 8
    round_bit = (normalized >> (frac_lo - 2)) & 1            # bit 7
    sticky_bit = 1 if (normalized & ((1 << (frac_lo - 2)) - 1)) else 0  # bits 6..0
    round_up = guard_bit and (round_bit or sticky_bit or (frac_pre & 1))

    rounded_frac_wide = frac_pre + (1 if round_up else 0)
    mantissa_carry = (rounded_frac_wide >> _FRAC_W) & 1
    if mantissa_carry:
        rounded_mantissa = 0
    else:
        rounded_mantissa = rounded_frac_wide & ((1 << _FRAC_W) - 1)

    if underflow or mag_fixed == 0:
        res_raw_exp = 0
    else:
        res_raw_exp = exp_calc & ((1 << _EXP_W) - 1)
    if mantissa_carry and not (underflow or mag_fixed == 0):
        res_raw_exp = (res_raw_exp + 1) & ((1 << _EXP_W) - 1)

    final_raw_exp = 0 if is_zero else res_raw_exp
    final_mantissa = 0 if (is_zero or underflow or mag_fixed == 0) else rounded_mantissa

    return ((sign & 1) << 15) | ((final_raw_exp & 0xFF) << 7) | (final_mantissa & 0x7F)


@dataclass
class TanhReq:
    xVec: list[int]                     # 16 BF16 bit patterns
    laneMask: int = 0xFFFF


@dataclass
class TanhResp:
    result: list[int]                   # 16 BF16 bit patterns


class TanhRec:
    """Mirror of `class TanhRec` in
    src/main/scala/atlas/vector/laneBoxes/TanhRec.scala.

    Visible latency = 1 (one `commonState` register stage — the per-lane
    `Tanh` modules and the shared `TanhLUT` each add their own internal
    cycle, but both are hidden behind the same single register boundary
    from the funct-model viewpoint)."""

    LATENCIES: dict[str, int] = {"tanh": 1}

    def __init__(self, p: VectorParams):
        self.p = p
        self._lut = gen_tanh_lut(
            addr_bits=_ADDR_BITS, m=1, n=_N, minimum=0.0, maximum=4.0,
        )
        self._queues: dict[str, deque] = {
            op: deque([None] * lat) for op, lat in self.LATENCIES.items()
        }

    def reset(self) -> None:
        for op, lat in self.LATENCIES.items():
            self._queues[op] = deque([None] * lat)

    def compute_now(self, req: TanhReq) -> TanhResp:
        n = self.p.num_lanes
        if len(req.xVec) != n:
            raise ValueError(f"xVec must have {n} lanes, got {len(req.xVec)}")
        out: list[int] = []
        for i in range(n):
            lane_en = bool((req.laneMask >> i) & 1)
            if not lane_en:
                out.append(0x0000)
            else:
                out.append(tanh_bf16(req.xVec[i], self._lut))
        return TanhResp(result=out)

    def step(self, op_name: str, req: Optional[TanhReq]) -> Optional[TanhResp]:
        if op_name not in self._queues:
            raise KeyError(
                f"TanhRec has no op {op_name!r}; "
                f"valid: {sorted(self.LATENCIES)}"
            )
        produced = self.compute_now(req) if req is not None else None
        q = self._queues[op_name]
        q.append(produced)
        return q.popleft()

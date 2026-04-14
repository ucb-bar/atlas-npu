"""lut_sources/lut_params.py — bit-exact ports of `trait LUTParams`
(`dependencies/sp26-fp-units/.../vpuLUTs/LUTParams.scala`) and the
`HasSinCosParams` helpers (`vpuFUnits/SinCos.scala`).

Each function mirrors a single Scala helper byte-for-byte:

  - `lut_fixed_to_bf16_rcp` ↔ `lutFixedToBf16Rcp`  (`LUTParams.scala:59-109`)
  - `lut_fixed_to_bf16_sqrt` ↔ `lutFixedToBf16Sqrt` (`LUTParams.scala:7-57`)
  - `lut_fixed_to_bf16_log` ↔ `lutFixedToBf16Log` (`LUTParams.scala:111-161`)
  - `lut_fixed_to_bf16`     ↔ `lutFixedToBf16`     (`SinCos.scala:111-159`)
  - `bf16_to_qmn_times_two_over_pi` ↔ `bf16ToQmnTimesTwoOverPi`
                                       (`SinCos.scala:24-109`)

The Scala helpers operate on `UInt`/`SInt` chisel signals; the Python
ports take Python ints and return Python ints, with the same width
truncations applied explicitly. Special-case ordering (NaN-before-Inf,
underflow-before-overflow) is preserved exactly.

Sign-handling note: `neg` is a separate flag, NOT folded into `x`. The
LUT modules pass the BF16-input sign as `neg`; the LUT *value* is
always unsigned-positive. This matches `Cat(neg, expUnsigned, frac)`
in the Scala.
"""

from __future__ import annotations


def _highest_bit_index(x: int, in_w: int) -> int:
    """Index of the highest 1-bit in `x` (0..in_w-1).

    Mirrors the Chisel sweep
        highestIdx := 0.U
        for (i <- 0 until inW) { when(x(i)) { highestIdx := i.U } }
    which returns 0 when `x == 0` (the loop never fires) and the
    last-written `i` otherwise. Python `int.bit_length() - 1` matches
    this for any non-zero `x`; for `x == 0` we explicitly return 0.
    """
    x &= (1 << in_w) - 1
    if x == 0:
        return 0
    return x.bit_length() - 1


def _signed_to_uint(value: int, width: int) -> int:
    """Two's-complement reinterpret a Python int as a `width`-bit unsigned.
    Mirrors `bfExpSigned.asUInt(7,0)` truncation patterns in the Scala."""
    return value & ((1 << width) - 1)


def lut_fixed_to_bf16_sqrt(
    x: int,
    exp: int,
    fra: int,
    lut_val_m: int = 1,
    lut_val_n: int = 16,
    neg: int = 0,
) -> int:
    """Mirror of `lutFixedToBf16Sqrt` (`LUTParams.scala:7-57`).

    `x` is the (possibly-shifted) LUT value as an `(lut_val_m + lut_val_n)`-
    bit unsigned. `exp` and `fra` are the BF16 input's biased exponent and
    7-bit fraction, used purely to decide the output exponent / special-case
    lattice. `neg` is the BF16 input sign (0 or 1)."""
    in_w = lut_val_m + lut_val_n
    x &= (1 << in_w) - 1
    exp &= 0xFF
    fra &= 0x7F
    neg &= 1

    bias = 127

    highest_idx = _highest_bit_index(x, in_w)
    unbiased_exp = highest_idx - lut_val_n
    # bfExpSigned = bias + ((unbiasedExp + exp - bias) >> 1)
    bf_exp_signed = bias + ((unbiased_exp + exp - bias) >> 1)
    exp_unsigned = _signed_to_uint(bf_exp_signed, 8)

    shift_amt = highest_idx - 7 if highest_idx > 7 else 0
    mant_pre = x >> shift_amt
    frac = mant_pre & 0x7F

    is_input_zero = (exp == 0 and fra == 0)
    is_input_subnormal = (exp == 0 and fra != 0)
    is_input_inf = (exp == 0xFF and fra == 0)
    is_input_nan = (exp == 0xFF and fra != 0)

    if is_input_zero or is_input_subnormal or is_input_nan:
        return (neg << 15)
    if is_input_inf or exp_unsigned >= 255:
        return ((neg << 15) | 0x7F80) & 0xFFFF
    return ((neg << 15) | (exp_unsigned << 7) | frac) & 0xFFFF


def lut_fixed_to_bf16_rcp(
    x: int,
    exp: int,
    fra: int,
    lut_val_m: int = 9,
    lut_val_n: int = 12,
    neg: int = 0,
) -> int:
    """Mirror of `lutFixedToBf16Rcp` (`LUTParams.scala:59-109`).

    Note the special-case lattice differs from sqrt: `0`/`subnormal`/
    `expUnsigned >= 255` flush to **infinity** (because 1/0 → inf), and
    `inf`/`NaN` flush to zero (1/inf → 0).
    """
    in_w = lut_val_m + lut_val_n
    x &= (1 << in_w) - 1
    exp &= 0xFF
    fra &= 0x7F
    neg &= 1

    bias = 127
    twobias = 2 * bias  # 254

    highest_idx = _highest_bit_index(x, in_w)
    unbiased_exp = highest_idx - lut_val_n
    bf_exp_signed = twobias + unbiased_exp - exp
    exp_unsigned = _signed_to_uint(bf_exp_signed, 8)

    shift_amt = highest_idx - 7 if highest_idx > 7 else 0
    mant_pre = x >> shift_amt
    frac = mant_pre & 0x7F

    is_input_zero = (exp == 0 and fra == 0)
    is_input_subnormal = (exp == 0 and fra != 0)
    is_input_inf = (exp == 0xFF and fra == 0)
    is_input_nan = (exp == 0xFF and fra != 0)

    if is_input_inf or is_input_nan:
        return (neg << 15)
    if is_input_zero or is_input_subnormal or exp_unsigned >= 255:
        return ((neg << 15) | 0x7F80) & 0xFFFF
    return ((neg << 15) | (exp_unsigned << 7) | frac) & 0xFFFF


def lut_fixed_to_bf16_log(
    x: int,
    exp: int,
    fra: int,
    lut_val_m: int = 9,
    lut_val_n: int = 12,
    neg: int = 0,
) -> int:
    """Mirror of `lutFixedToBf16Log` (`LUTParams.scala:111-161`).

    Special-case lattice:
      - input NaN OR `x == 0` (LUT raw value is exactly 0) → signed zero.
        The `x == 0` case kicks in when the BF16 input was exactly 1.0
        (so `log(1) == 0` and the combined LUT value is 0 too).
      - input zero / subnormal / inf / `expUnsigned >= 255` → signed inf.
        Note that input zero → -inf is encoded as `neg=1, expUnsigned=0xFF`,
        i.e. the caller passes `isNeg = (exp < bias)` and the output
        becomes `Cat(1, 0xFF, 0) = 0xFF80`.
      - normal → `Cat(neg, expUnsigned, frac)`.
    """
    in_w = lut_val_m + lut_val_n
    x &= (1 << in_w) - 1
    exp &= 0xFF
    fra &= 0x7F
    neg &= 1

    bias = 127

    highest_idx = _highest_bit_index(x, in_w)
    unbiased_exp = highest_idx - lut_val_n
    bf_exp_signed = unbiased_exp + bias
    exp_unsigned = _signed_to_uint(bf_exp_signed, 8)

    shift_amt = highest_idx - 7 if highest_idx > 7 else 0
    mant_pre = x >> shift_amt
    frac = mant_pre & 0x7F

    is_zero_lut = (x == 0)
    is_input_zero = (exp == 0 and fra == 0)
    is_input_subnormal = (exp == 0 and fra != 0)
    is_input_inf = (exp == 0xFF and fra == 0)
    is_input_nan = (exp == 0xFF and fra != 0)

    if is_input_nan or is_zero_lut:
        return (neg << 15)
    if is_input_zero or is_input_subnormal or is_input_inf or exp_unsigned >= 255:
        return ((neg << 15) | 0x7F80) & 0xFFFF
    return ((neg << 15) | (exp_unsigned << 7) | frac) & 0xFFFF


def lut_fixed_to_bf16(
    x: int,
    lut_val_m: int = 1,
    lut_val_n: int = 16,
    neg: int = 0,
) -> int:
    """Mirror of `lutFixedToBf16` (`SinCos.scala:111-159`).

    Used by SinCos for its post-interpolation conversion. Unlike the
    sqrt/rcp/log variants this one has NO BF16 input metadata — it
    works purely off the LUT-fixed value's bit pattern. The special
    cases collapse to:
      - `x == 0` → signed zero
      - `bfExpSigned <= 0` (subnormal / underflow) → signed zero
      - `bfExpSigned >= 255` (overflow) → signed inf
      - normal → `Cat(neg, expUnsigned, frac)`
    """
    in_w = lut_val_m + lut_val_n
    x &= (1 << in_w) - 1
    neg &= 1

    bias = 127
    is_zero = (x == 0)

    highest_idx = _highest_bit_index(x, in_w)
    unbiased_exp = highest_idx - lut_val_n
    bf_exp_signed = unbiased_exp + bias

    shift_amt = highest_idx - 7 if highest_idx > 7 else 0
    mant_pre = x >> shift_amt
    mant8 = mant_pre & 0xFF
    frac = mant8 & 0x7F

    if is_zero:
        return (neg << 15)
    if bf_exp_signed <= 0:
        return (neg << 15)
    if bf_exp_signed >= 255:
        return ((neg << 15) | 0x7F80) & 0xFFFF
    return ((neg << 15) | ((bf_exp_signed & 0xFF) << 7) | frac) & 0xFFFF


# ----------------------------------------------------------------------
#  bf16 → Q(m,n) × (2/π)  (used by SinCosVec, mirrors HasSinCosParams)
# ----------------------------------------------------------------------

import math


def bf16_to_qmn_times_two_over_pi(
    x: int, qmn_m: int = 9, qmn_n: int = 12
) -> int:
    """Mirror of `bf16ToQmnTimesTwoOverPi` (`SinCos.scala:24-109`).

    Returns a signed Python int representing the Q(qmn_m, qmn_n)
    fixed-point product `bf16(x) * (2/pi)`, saturated to ±maxVal.

    Step-by-step:
      1. Decode BF16 → integer mantissa with implicit leading 1.
      2. Shift left/right by `(exp - bias) + qmn_n - 7` to land in
         Q(qmn_m, qmn_n).
      3. Apply sign and saturate to [minVal, maxVal] (= [-2^(m+n),
         2^(m+n)-1]).
      4. Multiply by `round((2/pi) * 2^qmn_n)` (HALF_UP → matches
         BigDecimal's setScale semantics for positive constants).
      5. Right-shift by qmn_n with arithmetic-shift semantics; saturate
         again. Final return is a Python int in [-2^(m+n), 2^(m+n)-1].
    """
    x &= 0xFFFF
    sign = (x >> 15) & 1
    exp = (x >> 7) & 0xFF
    frac = x & 0x7F

    bias = 127
    out_w = qmn_m + qmn_n + 1
    max_val = (1 << (qmn_m + qmn_n)) - 1
    min_val = -(1 << (qmn_m + qmn_n))

    is_zero = (exp == 0 and frac == 0)
    is_sub = (exp == 0 and frac != 0)
    is_inf = (exp == 0xFF and frac == 0)
    is_nan = (exp == 0xFF and frac != 0)

    # Step 1: integer mantissa.
    mant = (1 << 7) | frac  # 8-bit `1.frac`, scaled by 2^7.

    # Step 2: shift to Q(qmn_m, qmn_n).
    shift = (exp - bias) + qmn_n - 7
    if shift >= 0:
        # `shiftedUnsigned := (mant << sh)(qmn_m + qmn_n + 7, 0)` —
        # truncation to (qmn_m + qmn_n + 8) bits is via Chisel slice.
        shifted_unsigned = (mant << shift) & ((1 << (qmn_m + qmn_n + 8)) - 1)
    else:
        shifted_unsigned = mant >> (-shift)
    shifted_signed = shifted_unsigned  # always non-negative (mant >= 0)

    if is_zero or is_sub or is_nan:
        qmn_val = 0
    elif is_inf:
        qmn_val = min_val if sign else max_val
    else:
        signed_val = -shifted_signed if sign else shifted_signed
        if signed_val > max_val:
            qmn_val = max_val
        elif signed_val < min_val:
            qmn_val = min_val
        else:
            # `signedVal.asTypeOf(SInt(outW.W))` → fit in outW signed bits.
            # Already in range, so pass through.
            qmn_val = signed_val

    # Step 4: multiply by 2/pi in Q(qmn_m, qmn_n).
    # BigDecimal HALF_UP → Python `round_half_up` for positive constant.
    two_over_pi_scaled = int(
        math.floor((2.0 / math.pi) * (1 << qmn_n) + 0.5)
    )
    # Both operands are signed; product fits in 2*out_w bits.
    product = qmn_val * two_over_pi_scaled

    # Arithmetic shift right by qmn_n.
    if product < 0:
        # Python `>>` already uses arithmetic shift on negatives.
        scaled_product = -((-product) >> qmn_n)
        # `(product >> qmn_n).asSInt` rounds toward -inf for negatives,
        # which is identical to Python's `>>` semantics. Use that
        # directly to avoid drift on exact-multiple cases.
        scaled_product = product >> qmn_n
    else:
        scaled_product = product >> qmn_n

    # Step 5: saturate result.
    if scaled_product > max_val:
        result = max_val
    elif scaled_product < min_val:
        result = min_val
    else:
        result = scaled_product
    return result

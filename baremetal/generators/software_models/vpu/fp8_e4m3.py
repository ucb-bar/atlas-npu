"""FP8 E4M3 <-> BF16 bit-exact primitives.

Mirrors src/main/scala/atlas/vector/laneBoxes/FP8Pack.scala and
FP8Unpack.scala. Validated against RVFP8PackTest.scala's goldenFp8ByteFromBf16
and RVFP8UnpackTest.scala's equivalent golden.

Reserved NaN handling:
  - Pack: if the post-rounding E4M3 result would land on 0x7F / 0xFF
    (exp=0xF mant=0x7), clamp to 0x7E / 0xFE.
  - Unpack: any FP8 byte with (exp=0xF mant=0x7) flushes to signed zero.
"""

from __future__ import annotations

E4M3_MAX_POS = 0x7E
E4M3_MAX_NEG = 0xFE


def _round_right_shift_4_rne(mant8: int) -> int:
    """Round an 8-bit significand down to 4 bits using round-to-nearest-even.

    Mirrors FP8Pack.scala's roundRightShift4RNE: the 8-bit `1.mmmmmmm` value
    has its top 4 bits kept (`1.mmm`), bit 4 is the guard, bits 0..3 form
    guard/sticky, and we increment when guard=1 and (sticky!=0 or lsb==1).
    """
    trunc = (mant8 >> 4) & 0xF
    guard = (mant8 >> 3) & 0x1
    sticky = mant8 & 0x7
    lsb = trunc & 0x1
    inc = 1 if (guard == 1 and (sticky != 0 or lsb == 1)) else 0
    return trunc + inc


def _pack_e4m3(sign: int, exp: int, mant: int) -> int:
    return ((sign & 1) << 7) | ((exp & 0xF) << 3) | (mant & 0x7)


def bf16_to_e4m3_byte(bf16: int, exp_shift: int) -> int:
    """Convert a single BF16 bit pattern to a single E4M3 FP8 byte.

    Mirrors FP8Pack.scala and RVFP8PackTest.goldenFp8ByteFromBf16.

    - Subnormal / NaN inputs -> 0x00 (plus sign dropped for the zero case).
    - Inf inputs -> signed max finite (0x7E / 0xFE).
    - Normal inputs are exponent-shifted (`unbExp - exp_shift`), rounded, and
      checked for overflow / underflow / reserved-NaN clamp. Values below
      E4M3 min-normal (unbExp < -6 after the shift) flush to zero.
    """
    sign = (bf16 >> 15) & 0x1
    exp_bf = (bf16 >> 7) & 0xFF
    mant_bf = bf16 & 0x7F

    is_zero = exp_bf == 0 and mant_bf == 0
    is_sub = exp_bf == 0 and mant_bf != 0
    is_inf = exp_bf == 0xFF and mant_bf == 0
    is_nan = exp_bf == 0xFF and mant_bf != 0

    if is_zero or is_sub or is_nan:
        return 0
    if is_inf:
        return E4M3_MAX_NEG if sign == 1 else E4M3_MAX_POS

    unb_exp = exp_bf - 127
    exp_adjusted = unb_exp - exp_shift

    mant8 = (1 << 7) | mant_bf
    rounded_sig = _round_right_shift_4_rne(mant8)
    norm_carry = rounded_sig == 16

    if norm_carry:
        final_exp_adjusted = exp_adjusted + 1
        mant_fp8 = 0
    else:
        final_exp_adjusted = exp_adjusted
        mant_fp8 = (rounded_sig - 8) & 0x7

    if final_exp_adjusted > 8:
        return E4M3_MAX_NEG if sign == 1 else E4M3_MAX_POS
    if final_exp_adjusted < -6:
        return 0

    exp_fp8 = final_exp_adjusted + 7
    packed = _pack_e4m3(sign, exp_fp8, mant_fp8)
    if (packed & 0x7F) == 0x7F:
        return E4M3_MAX_NEG if sign == 1 else E4M3_MAX_POS
    return packed


def e4m3_byte_to_bf16(fp8: int, exp_shift: int) -> int:
    """Convert a single E4M3 FP8 byte to a BF16 bit pattern.

    Mirrors FP8Unpack.scala:69-100.

    - Zero / subnormal / reserved NaN -> signed zero (sign preserved).
    - Normal: `unbExpBF16 = (expFP8 - 7) + exp_shift`, clamp on BF16 range.
    - Overflow (>= 255 biased) -> signed max finite BF16 (0x7F7F / 0xFF7F).
    - Underflow (<= 0 biased) -> signed zero.
    """
    fp8 &= 0xFF
    sign = (fp8 >> 7) & 1
    exp_fp8 = (fp8 >> 3) & 0xF
    mant_fp8 = fp8 & 0x7

    is_zero = exp_fp8 == 0 and mant_fp8 == 0
    is_sub = exp_fp8 == 0 and mant_fp8 != 0
    is_nan = exp_fp8 == 0xF and mant_fp8 == 0x7

    if is_zero or is_sub or is_nan:
        return (sign << 15) & 0xFFFF

    unb_exp_fp8 = exp_fp8 - 7
    unb_exp_bf16 = unb_exp_fp8 + exp_shift
    exp_bf16_wide = unb_exp_bf16 + 127

    if exp_bf16_wide <= 0:
        return (sign << 15) & 0xFFFF
    if exp_bf16_wide >= 255:
        return 0xFF7F if sign else 0x7F7F

    frac_bf16 = (mant_fp8 & 0x7) << 4
    return ((sign & 1) << 15) | ((exp_bf16_wide & 0xFF) << 7) | (frac_bf16 & 0x7F)


def e8m0_to_scale_exp_clamped(scale_e8m0: int) -> int:
    """Mirror VectorEngine.scala:44-51's e8m0ToScaleExpClamped.

    Takes the raw 8-bit E8M0 scale, subtracts bias 127, and clamps to int8
    [-128, 127] before handing it to FP8Pack / FP8Unpack.
    """
    scale_e8m0 &= 0xFF
    scale_exp_wide = scale_e8m0 - 127
    if scale_exp_wide > 127:
        return 127
    if scale_exp_wide < -128:
        return -128
    return scale_exp_wide

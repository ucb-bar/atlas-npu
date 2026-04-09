from __future__ import annotations

import math

from .fp_formats import (
    BF16,
    E4M3ProdFmt,
    E4M3_MAX_NEG,
    E4M3_MAX_POS,
    OutputFmtSel,
    encode_e4m3_normal,
    f32_to_bf16_bits_rne,
    round_right_shift4_rne,
    sanitize_bf16,
    wrap_signed,
)


_E4M3_PROD_EXP_WIDTH = E4M3ProdFmt.expWidth
_E4M3_PROD_BIAS = E4M3ProdFmt.bias
_E4M3_PROD_SIG_WIDTH = E4M3ProdFmt.sigWidth
_E4M3_PROD_EXP_MAX = (1 << _E4M3_PROD_EXP_WIDTH) - 1
_BF16_BIAS = BF16.ieeeBias

# Flat LUT indexed by (a << 8) | b
_E4M3_MUL_TO_PROD_LUT = [0] * 65536


def pack_e4m3_prod(sign: int, exp_unb: int, mant7: int) -> int:
    exp_field = exp_unb + _E4M3_PROD_BIAS
    if exp_field <= 0:
        return 0
    if exp_field >= _E4M3_PROD_EXP_MAX:
        exp_field = _E4M3_PROD_EXP_MAX
    return ((sign & 1) << 12) | ((exp_field & 0x1F) << 7) | (mant7 & 0x7F)


def _build_e4m3_mul_to_prod_lut() -> None:
    idx = 0
    for a_bits in range(256):
        a_sign = (a_bits >> 7) & 1
        a_exp = (a_bits >> 3) & 0xF
        a_man = a_bits & 0x7

        a_zero = (a_exp == 0)

        for b_bits in range(256):
            b_sign = (b_bits >> 7) & 1
            b_exp = (b_bits >> 3) & 0xF
            b_man = b_bits & 0x7

            out_sign = a_sign ^ b_sign

            if a_zero or b_exp == 0:
                _E4M3_MUL_TO_PROD_LUT[idx] = out_sign << 12
                idx += 1
                continue

            a_sig = 8 | a_man
            b_sig = 8 | b_man
            prod_sig = a_sig * b_sig

            need_shift = (prod_sig >> 7) & 1
            out_man = (prod_sig & 0x7F) if need_shift else ((prod_sig & 0x3F) << 1)
            out_exp = a_exp + b_exp + need_shift - 1

            _E4M3_MUL_TO_PROD_LUT[idx] = (out_sign << 12) | ((out_exp & 0x1F) << 7) | out_man
            idx += 1


_build_e4m3_mul_to_prod_lut()


def e4m3_mul_to_prod(a_bits: int, b_bits: int) -> int:
    return _E4M3_MUL_TO_PROD_LUT[((a_bits & 0xFF) << 8) | (b_bits & 0xFF)]


def ieee_to_aligned_int(ieee_bits: int, fmt, anchor_exp: int, int_width: int) -> int:
    mant_bits = fmt.mantissaBits
    exp_width = fmt.expWidth
    exp_mask = (1 << exp_width) - 1

    sign = (ieee_bits >> (fmt.ieeeWidth - 1)) & 1
    exp_field = (ieee_bits >> mant_bits) & exp_mask
    frac = ieee_bits & ((1 << mant_bits) - 1)

    if exp_field == 0 and frac == 0:
        return 0

    unb_exp = exp_field - fmt.ieeeBias
    full_sig = ((exp_field != 0) << mant_bits) | frac
    shift_right = anchor_exp - unb_exp - (int_width - 1 - mant_bits)

    if shift_right >= int_width:
        magnitude = 0
    elif shift_right >= 0:
        magnitude = full_sig >> shift_right
    elif shift_right > -int_width:
        magnitude = full_sig << (-shift_right)
    else:
        magnitude = 0

    magnitude &= (1 << int_width) - 1
    return wrap_signed(-magnitude if sign else magnitude, int_width)


def e4m3_prod_to_aligned_int(prod_bits: int, anchor_exp: int, int_width: int) -> int:
    exp_bits = (prod_bits >> 7) & 0x1F
    if exp_bits == 0:
        return 0

    sign = (prod_bits >> 12) & 1
    man = prod_bits & 0x7F

    sig = (1 << 7) | man
    unb_exp = exp_bits - _E4M3_PROD_BIAS
    rshift = anchor_exp - unb_exp

    left_pad = int_width - _E4M3_PROD_SIG_WIDTH
    sig_wide = sig << left_pad if left_pad > 0 else sig

    if rshift < 0:
        shifted = sig_wide << (-rshift)
    else:
        shifted = sig_wide >> rshift

    magnitude = shifted & ((1 << int_width) - 1)
    return wrap_signed(-magnitude if sign else magnitude, int_width)


def aligned_int_to_bf16(int_in: int, anchor_exp: int, int_width: int) -> int:
    int_in = wrap_signed(int_in, int_width)
    if int_in == 0:
        return 0
    return f32_to_bf16_bits_rne(math.ldexp(float(int_in), anchor_exp - (int_width - 1)))


def bf16_scale_to_e4m3(bf16_bits: int, scale_exp: int) -> int:
    bf16_bits &= 0xFFFF

    sign = (bf16_bits >> 15) & 1
    exp_bf16 = (bf16_bits >> 7) & 0xFF
    frac_bf16 = bf16_bits & 0x7F

    if exp_bf16 == 0:
        return 0

    if exp_bf16 == 0xFF:
        if frac_bf16 != 0:
            return 0
        return E4M3_MAX_NEG if sign else E4M3_MAX_POS

    scaled_unb_exp = (exp_bf16 - _BF16_BIAS) + scale_exp
    mant8 = 0x80 | frac_bf16
    rounded_norm = round_right_shift4_rne(mant8)

    if rounded_norm == 16:
        final_unb_exp = scaled_unb_exp + 1
        norm_mant = 0
    else:
        final_unb_exp = scaled_unb_exp
        norm_mant = (rounded_norm - 8) & 0x7

    if final_unb_exp > 8:
        return E4M3_MAX_NEG if sign else E4M3_MAX_POS
    if final_unb_exp >= -6:
        return encode_e4m3_normal(sign, final_unb_exp, norm_mant)
    return 0


def output_conv_stage(bf16_bits: int, out_fmt_sel, scale_exp: int) -> int:
    bf16_sanitized = sanitize_bf16(bf16_bits)
    if out_fmt_sel is OutputFmtSel.OutBF16:
        return bf16_sanitized
    return bf16_scale_to_e4m3(bf16_sanitized, scale_exp) & 0xFF
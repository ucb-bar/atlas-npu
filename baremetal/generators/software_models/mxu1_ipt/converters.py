"""mxu0's FP8 pop: BF16ScaleToE4M3 (FPUtils.bf16ToE4m3) on one BF16 value."""
from __future__ import annotations

from .fp_formats import (
    BF16,
    E4M3_MAX_NEG,
    E4M3_MAX_POS,
    encode_e4m3_normal,
    round_right_shift4_rne,
)


_BF16_BIAS = BF16.ieeeBias


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

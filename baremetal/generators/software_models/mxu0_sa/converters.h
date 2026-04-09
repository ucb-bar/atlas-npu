#ifndef CONVERTERS_H
#define CONVERTERS_H

/* converters.h — C translation of the Python accumulator / format-conversion
 * module.  Depends on fp_formats.h (same translation unit set).
 *
 */

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include "fp_formats.h" /* AtlasFPType, E4M3ProdFmt, OutputFmtSel, helpers */

/* ---------------------------------------------------------------------------
 * Module-level constants derived from E4M3ProdFmt / BF16
 * (hardcoded as integer literals — struct fields are not C constant-exprs)
 * ------------------------------------------------------------------------- */

#define _E4M3_PROD_EXP_WIDTH 5 /* E4M3ProdFmt.expWidth      */
#define _E4M3_PROD_BIAS 13     /* E4M3ProdFmt.bias           */
#define _E4M3_PROD_SIG_WIDTH 8 /* 1 + E4M3ProdFmt.mantissaBits */
#define _E4M3_PROD_EXP_MAX 31  /* (1 << 5) - 1               */
#define _BF16_BIAS 127         /* BF16.ieeeBias              */

/* ---------------------------------------------------------------------------
 * Flat 65536-entry multiply-product LUT, indexed by (a_bits<<8)|b_bits.
 * Call atlas_acc_init_lut() once (after atlas_fp_init_lut()) before any
 * call to e4m3_mul_to_prod().
 * ------------------------------------------------------------------------- */

static uint16_t _E4M3_MUL_TO_PROD_LUT[65536];

/* ---------------------------------------------------------------------------
 * pack_e4m3_prod
 *   Packs (sign, unbiased exponent, 7-bit mantissa) into the 13-bit
 *   E4M3 product format.
 * ------------------------------------------------------------------------- */

static inline uint16_t pack_e4m3_prod(int sign, int exp_unb, int mant7)
{
    int exp_field = exp_unb + _E4M3_PROD_BIAS;
    if (exp_field <= 0)
        return (uint16_t)0;
    if (exp_field >= _E4M3_PROD_EXP_MAX)
        exp_field = _E4M3_PROD_EXP_MAX;
    return (uint16_t)(((sign & 1) << 12) |
                      ((exp_field & 0x1F) << 7) |
                      (mant7 & 0x7F));
}

/* ---------------------------------------------------------------------------
 * _build_e4m3_mul_to_prod_lut  (internal — called by atlas_acc_init_lut)
 * ------------------------------------------------------------------------- */

static inline void _build_e4m3_mul_to_prod_lut(void)
{
    int idx = 0;
    for (int a_bits = 0; a_bits < 256; a_bits++)
    {
        int a_sign = (a_bits >> 7) & 1;
        int a_exp = (a_bits >> 3) & 0xF;
        int a_man = a_bits & 0x7;
        int a_zero = (a_exp == 0);

        for (int b_bits = 0; b_bits < 256; b_bits++)
        {
            int b_sign = (b_bits >> 7) & 1;
            int b_exp = (b_bits >> 3) & 0xF;
            int b_man = b_bits & 0x7;

            int out_sign = a_sign ^ b_sign;

            if (a_zero || b_exp == 0)
            {
                _E4M3_MUL_TO_PROD_LUT[idx++] = (uint16_t)(out_sign << 12);
                continue;
            }

            int a_sig = 8 | a_man;
            int b_sig = 8 | b_man;
            int prod_sig = a_sig * b_sig;

            int need_shift = (prod_sig >> 7) & 1;
            int out_man = need_shift ? (prod_sig & 0x7F)
                                     : ((prod_sig & 0x3F) << 1);
            int out_exp = a_exp + b_exp + need_shift - 1;

            _E4M3_MUL_TO_PROD_LUT[idx++] =
                (uint16_t)(((out_sign & 1) << 12) |
                           ((out_exp & 0x1F) << 7) |
                           (out_man & 0x7F));
        }
    }
}

/* ---------------------------------------------------------------------------
 * atlas_acc_init_lut — call once at program start (after atlas_fp_init_lut)
 * ------------------------------------------------------------------------- */

static inline void atlas_acc_init_lut(void)
{
    _build_e4m3_mul_to_prod_lut();
}

/* ---------------------------------------------------------------------------
 * e4m3_mul_to_prod  — O(1) LUT lookup
 * ------------------------------------------------------------------------- */

static inline uint16_t e4m3_mul_to_prod(uint8_t a_bits, uint8_t b_bits)
{
    return _E4M3_MUL_TO_PROD_LUT[((uint16_t)a_bits << 8) | b_bits];
}

/* ---------------------------------------------------------------------------
 * ieee_to_aligned_int
 *   Converts an IEEE-format value (BF16 or E4M3) to a fixed-point integer
 *   aligned so that the binary point sits at position anchor_exp.
 *
 *   fmt        : pointer to an AtlasFPType descriptor
 *   ieee_bits  : raw bit pattern of the value
 *   anchor_exp : target binary-point exponent
 *   int_width  : result bit-width (signed two's-complement)
 *
 *   FIX: Python computes magnitude as an arbitrary-precision integer, masks
 *   it, and only then negates (also arbitrary-precision) before wrap_signed.
 *   The original C negated a signed int32_t which is UB when magnitude ==
 *   INT32_MIN.  We now mirror Python exactly: mask first (into uint32_t),
 *   then negate in uint32_t two's-complement arithmetic, then wrap_signed.
 * ------------------------------------------------------------------------- */

static inline int32_t ieee_to_aligned_int(uint32_t ieee_bits,
                                          const AtlasFPType *fmt,
                                          int anchor_exp,
                                          int int_width)
{
    int mant_bits = fmt->mantissaBits;
    int exp_width = fmt->expWidth;
    uint32_t exp_mask = ((uint32_t)1 << exp_width) - 1u;

    int sign = (int)((ieee_bits >> (fmt->ieeeWidth - 1)) & 1u);
    int exp_field = (int)((ieee_bits >> mant_bits) & exp_mask);
    uint32_t frac = ieee_bits & (((uint32_t)1 << mant_bits) - 1u);

    if (exp_field == 0 && frac == 0)
        return 0;

    int unb_exp = exp_field - fmt->ieeeBias;
    uint32_t full_sig = ((exp_field != 0) ? ((uint32_t)1 << mant_bits) : 0u) | frac;

    int shift_right = anchor_exp - unb_exp - (int_width - 1 - mant_bits);

    /* Compute shifted magnitude as uint32_t (no mask yet — mirrors Python's
       unbounded shift before the single & at the end). */
    uint32_t shifted;
    if (shift_right >= int_width)
    {
        shifted = 0;
    }
    else if (shift_right >= 0)
    {
        shifted = full_sig >> shift_right;
    }
    else if (shift_right > -int_width)
    {
        shifted = full_sig << (-shift_right);
    }
    else
    {
        shifted = 0;
    }

    /* Apply mask — mirrors Python's  magnitude &= (1 << int_width) - 1  */
    uint32_t mask = (int_width < 32) ? (((uint32_t)1 << int_width) - 1u)
                                     : 0xFFFFFFFFu;
    uint32_t magnitude = shifted & mask;

    /* Negate in unsigned arithmetic to avoid signed-overflow UB, then
       re-mask so the result stays within int_width bits — exactly what
       Python's arbitrary-precision  -magnitude & mask  does.             */
    uint32_t result = sign ? ((-magnitude) & mask) : magnitude;

    return wrap_signed((int32_t)result, int_width);
}

/* ---------------------------------------------------------------------------
 * e4m3_prod_to_aligned_int
 *   Converts a 13-bit E4M3 product-format value to aligned fixed-point.
 *
 *   FIX: Python uses arbitrary-precision integers so the left-shift path
 *   (rshift < 0) never overflows.  The original C used uint32_t for
 *   sig_wide, which silently truncates when (-rshift) pushes bits past
 *   bit 31.  We now use uint64_t for the intermediate shift, then mask
 *   down to int_width bits — matching Python's behaviour exactly.
 *
 *   Same negation fix as ieee_to_aligned_int: negate unsigned, re-mask,
 *   then wrap_signed.
 * ------------------------------------------------------------------------- */

static inline int32_t e4m3_prod_to_aligned_int(uint16_t prod_bits,
                                               int anchor_exp,
                                               int int_width)
{
    int exp_bits = (prod_bits >> 7) & 0x1F;
    if (exp_bits == 0)
        return 0;

    int sign = (prod_bits >> 12) & 1;
    int man = prod_bits & 0x7F;

    uint64_t sig = (1u << 7) | (uint32_t)man; /* widen early */
    int unb_exp = exp_bits - _E4M3_PROD_BIAS;
    int rshift = anchor_exp - unb_exp;

    int left_pad = int_width - _E4M3_PROD_SIG_WIDTH;
    uint64_t sig_wide = (left_pad > 0) ? (sig << left_pad) : sig;

    /* Shift in 64-bit to avoid truncation on large left shifts — mirrors
       Python's arbitrary-precision integers.                               */
    uint64_t shifted;
    if (rshift < 0)
        shifted = sig_wide << (-rshift);
    else
        shifted = sig_wide >> rshift;

    /* Mask to int_width bits (same as Python's  & ((1 << int_width) - 1)  */
    uint64_t mask64 = (int_width < 64) ? (((uint64_t)1 << int_width) - 1u)
                                       : UINT64_MAX;
    uint32_t magnitude = (uint32_t)(shifted & mask64);

    /* Unsigned negate + re-mask avoids signed-overflow UB.                */
    uint32_t mask32 = (int_width < 32) ? (((uint32_t)1 << int_width) - 1u)
                                       : 0xFFFFFFFFu;
    uint32_t result = sign ? ((-magnitude) & mask32) : magnitude;

    return wrap_signed((int32_t)result, int_width);
}

/* ---------------------------------------------------------------------------
 * aligned_int_to_bf16
 *   Converts a signed fixed-point integer (with implicit binary point at
 *   anchor_exp) to BF16 bits using RNE rounding.
 * ------------------------------------------------------------------------- */

static inline uint16_t aligned_int_to_bf16(int32_t int_in,
                                           int anchor_exp,
                                           int int_width)
{
    int_in = wrap_signed(int_in, int_width);
    if (int_in == 0)
        return 0;
    /* ldexpf: int_in * 2^(anchor_exp - (int_width-1)) */
    float f = ldexpf((float)int_in, anchor_exp - (int_width - 1));
    return f32_to_bf16_bits_rne(f);
}

/* ---------------------------------------------------------------------------
 * bf16_scale_to_e4m3
 *   Scales a BF16 value by 2^scale_exp and converts to E4M3.
 * ------------------------------------------------------------------------- */

static inline uint8_t bf16_scale_to_e4m3(uint16_t bf16_bits, int scale_exp)
{
    bf16_bits &= 0xFFFF;

    int sign = (bf16_bits >> 15) & 1;
    int exp_bf16 = (bf16_bits >> 7) & 0xFF;
    int frac_bf16 = bf16_bits & 0x7F;

    if (exp_bf16 == 0)
        return 0;

    if (exp_bf16 == 0xFF)
    {
        if (frac_bf16 != 0)
            return 0;
        return sign ? E4M3_MAX_NEG : E4M3_MAX_POS;
    }

    int scaled_unb_exp = (exp_bf16 - _BF16_BIAS) + scale_exp;
    int mant8 = 0x80 | frac_bf16;
    int rounded_norm = round_right_shift4_rne(mant8);

    int final_unb_exp, norm_mant;
    if (rounded_norm == 16)
    {
        final_unb_exp = scaled_unb_exp + 1;
        norm_mant = 0;
    }
    else
    {
        final_unb_exp = scaled_unb_exp;
        norm_mant = (rounded_norm - 8) & 0x7;
    }

    if (final_unb_exp > 8)
        return sign ? E4M3_MAX_NEG : E4M3_MAX_POS;
    if (final_unb_exp >= -6)
        return encode_e4m3_normal(sign, final_unb_exp, norm_mant);
    return 0;
}

/* ---------------------------------------------------------------------------
 * output_conv_stage
 *   Final output conversion: sanitize BF16 then either keep as BF16 or
 *   convert to E4M3.
 * ------------------------------------------------------------------------- */

static inline uint32_t output_conv_stage(uint16_t bf16_bits,
                                         OutputFmtSel out_fmt_sel,
                                         int scale_exp)
{
    uint16_t sanitized = sanitize_bf16(bf16_bits);
    if (out_fmt_sel == OutputFmtSel_OutBF16)
        return (uint32_t)sanitized;
    return (uint32_t)(bf16_scale_to_e4m3(sanitized, scale_exp) & 0xFF);
}

#endif /* CONVERTERS_H */
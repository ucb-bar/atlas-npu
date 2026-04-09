#ifndef FP_FORMATS_H
#define FP_FORMATS_H

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Type descriptors
 * ------------------------------------------------------------------------- */

typedef struct
{
    const char *name;
    int ieeeWidth;
    int expWidth;
    int mantissaBits;
    int ieeeBias;
} AtlasFPType;

/* sigWidth = 1 + mantissaBits */
static inline int AtlasFPType_sigWidth(const AtlasFPType *t)
{
    return 1 + t->mantissaBits;
}

static const AtlasFPType E4M3 = {"E4M3", 8, 4, 3, 7};
static const AtlasFPType BF16 = {"BF16", 16, 8, 7, 127};

/* ---------------------------------------------------------------------------
 * Product-format descriptor
 * ------------------------------------------------------------------------- */

typedef struct
{
    int width;        /* 13 */
    int expWidth;     /* 5  */
    int mantissaBits; /* 7  */
    int bias;         /* 13 */
} E4M3ProdFmtDesc;

/* sigWidth = 1 + mantissaBits */
static inline int E4M3ProdFmtDesc_sigWidth(const E4M3ProdFmtDesc *d)
{
    return 1 + d->mantissaBits;
}

static const E4M3ProdFmtDesc E4M3ProdFmt = {13, 5, 7, 13};

/* ---------------------------------------------------------------------------
 * Enums
 * ------------------------------------------------------------------------- */

typedef enum
{
    AddendSel_UseAct = 0,
    AddendSel_UseBias = 1,
    AddendSel_UsePsum = 2,
} AddendSel;

typedef enum
{
    OutputFmtSel_OutBF16 = 0,
    OutputFmtSel_OutE4M3 = 1,
} OutputFmtSel;

/* ---------------------------------------------------------------------------
 * Constants
 * ------------------------------------------------------------------------- */

#define BF16_MAX_POS UINT16_C(0x7F7F)
#define BF16_MAX_NEG UINT16_C(0xFF7F)
#define E4M3_MAX_POS UINT8_C(0x7E)
#define E4M3_MAX_NEG UINT8_C(0xFE)

#define F32_SIGN_MASK UINT32_C(0x80000000)
#define F32_EXP_MASK UINT32_C(0x7F800000)
#define F32_FRAC_MASK UINT32_C(0x007FFFFF)

#define _E4M3_BIAS 7

/* ---------------------------------------------------------------------------
 * DecodedFloat
 * ------------------------------------------------------------------------- */

typedef struct
{
    int sign;
    int exp_field;
    int frac;
    bool is_zero;
    bool is_sub;
    bool is_inf;
    bool is_nan;
    /* unb_exp and sig are valid only when the corresponding Python field
       is not None (i.e., is_zero=false, is_inf=false, is_nan=false for
       unb_exp; same conditions for sig).  Use has_unb_exp / has_sig to
       guard access. */
    bool has_unb_exp;
    bool has_sig;
    int unb_exp;
    float sig;
    float value;
} DecodedFloat;

/* ---------------------------------------------------------------------------
 * Bit-cast helpers (avoid UB via memcpy)
 * ------------------------------------------------------------------------- */

static inline float u32_to_float(uint32_t x)
{
    float f;
    memcpy(&f, &x, sizeof f);
    return f;
}

static inline uint32_t float_to_u32(float x)
{
    uint32_t u;
    memcpy(&u, &x, sizeof u);
    return u;
}

/* ---------------------------------------------------------------------------
 * Utility: sign-extend an n-bit value
 * ------------------------------------------------------------------------- */

static inline int32_t sign_extend(int32_t value, int bits)
{
    int32_t sign_bit = 1 << (bits - 1);
    return (value & (sign_bit - 1)) - (value & sign_bit);
}

/* ---------------------------------------------------------------------------
 * Utility: clamp a signed value to fit in 'bits' bits
 * ------------------------------------------------------------------------- */

static inline int32_t clamp_signed(int32_t value, int bits)
{
    int32_t lo = -(1 << (bits - 1));
    int32_t hi = (1 << (bits - 1)) - 1;
    if (value < lo)
        return lo;
    if (value > hi)
        return hi;
    return value;
}

/* ---------------------------------------------------------------------------
 * Utility: wrap (two's-complement truncate) a signed value to 'bits' bits
 * ------------------------------------------------------------------------- */

static inline int32_t wrap_signed(int32_t value, int bits)
{
    if (bits == 32)
        return value;
    uint32_t mask = ((uint32_t)1 << bits) - 1u;
    uint32_t uv = (uint32_t)value & mask;
    uint32_t sign = (uint32_t)1 << (bits - 1);
    return (uv & sign) ? (int32_t)(uv - ((uint32_t)1 << bits))
                       : (int32_t)uv;
}

/* ---------------------------------------------------------------------------
 * E4M3 decode (single value, no LUT)
 * ------------------------------------------------------------------------- */

static inline DecodedFloat decode_e4m3_uncached(uint8_t bits)
{
    DecodedFloat d;
    d.sign = (bits >> 7) & 1;
    d.exp_field = (bits >> 3) & 0xF;
    d.frac = bits & 0x7;

    if (d.exp_field == 0)
    {
        if (d.frac == 0)
        {
            d.is_zero = true;
            d.is_sub = false;
            d.is_inf = false;
            d.is_nan = false;
            d.has_unb_exp = false;
            d.has_sig = false;
            d.unb_exp = 0;
            d.sig = 0.0f;
            d.value = d.sign ? -0.0f : 0.0f;
            return d;
        }
        d.is_zero = false;
        d.is_sub = true;
        d.is_inf = false;
        d.is_nan = false;
        d.has_unb_exp = true;
        d.has_sig = true;
        d.unb_exp = 1 - _E4M3_BIAS;
        d.sig = d.frac * 0.125f;
        {
            float mag = d.sig * powf(2.0f, (float)d.unb_exp);
            d.value = d.sign ? -mag : mag;
        }
        return d;
    }

    if (d.exp_field == 0xF)
    {
        d.is_zero = false;
        d.is_sub = false;
        d.has_unb_exp = false;
        d.has_sig = false;
        d.unb_exp = 0;
        d.sig = 0.0f;
        if (d.frac == 0)
        {
            d.is_inf = true;
            d.is_nan = false;
            d.value = d.sign ? -INFINITY : INFINITY;
        }
        else
        {
            d.is_inf = false;
            d.is_nan = true;
            d.value = NAN;
        }
        return d;
    }

    /* Normal number */
    d.is_zero = false;
    d.is_sub = false;
    d.is_inf = false;
    d.is_nan = false;
    d.has_unb_exp = true;
    d.has_sig = true;
    d.unb_exp = d.exp_field - _E4M3_BIAS;
    d.sig = 1.0f + d.frac * 0.125f;
    {
        float mag = d.sig * powf(2.0f, (float)d.unb_exp);
        d.value = d.sign ? -mag : mag;
    }
    return d;
}

/* ---------------------------------------------------------------------------
 * E4M3 decode LUT (256 entries, initialised once at program start via
 * atlas_fp_init_lut()).  After init, call decode_e4m3() for O(1) lookup.
 * ------------------------------------------------------------------------- */

static DecodedFloat _E4M3_DECODE_LUT[256];

static inline void atlas_fp_init_lut(void)
{
    for (int i = 0; i < 256; i++)
        _E4M3_DECODE_LUT[i] = decode_e4m3_uncached((uint8_t)i);
}

static inline DecodedFloat decode_e4m3(uint8_t bits)
{
    return _E4M3_DECODE_LUT[bits];
}

/* ---------------------------------------------------------------------------
 * E4M3 encode (normal numbers only)
 * ------------------------------------------------------------------------- */

static inline uint8_t encode_e4m3_normal(int sign, int unb_exp, int mant3)
{
    return (uint8_t)(((sign & 1) << 7) |
                     (((unb_exp + _E4M3_BIAS) & 0xF) << 3) |
                     (mant3 & 0x7));
}

/* ---------------------------------------------------------------------------
 * BF16 ↔ F32
 * ------------------------------------------------------------------------- */

/* Round float to BF16 bits using round-to-nearest-even. */
static inline uint16_t f32_to_bf16_bits_rne(float x)
{
    uint32_t u = float_to_u32(x);
    uint32_t exp = (u >> 23) & 0xFFu;

    if (exp == 0xFFu)
    {
        uint32_t frac = u & 0x7FFFFFu;
        return frac ? (uint16_t)0x7FC0 : (uint16_t)((u >> 16) & 0xFFFF);
    }

    uint32_t upper = u >> 16;
    uint32_t rounded = u + 0x7FFFu + (upper & 1u);
    return (uint16_t)((rounded >> 16) & 0xFFFF);
}

static inline float bf16_bits_to_f32(uint16_t bits)
{
    return u32_to_float((uint32_t)bits << 16);
}

/* ---------------------------------------------------------------------------
 * Round-right-shift by 4 with round-to-nearest-even
 * ------------------------------------------------------------------------- */

static inline int round_right_shift4_rne(int x)
{
    int trunc = (x >> 4) & 0xF;
    int guard = (x >> 3) & 1;
    int sticky = ((x & 0x7) != 0) ? 1 : 0;
    return trunc + (guard & (sticky | (trunc & 1)));
}

/* ---------------------------------------------------------------------------
 * Sanitize BF16 bits:
 *   - subnormals → flush to zero
 *   - infinities → clamp to max finite
 *   - NaN        → flush to zero
 * ------------------------------------------------------------------------- */

static inline uint16_t sanitize_bf16(uint16_t bits)
{
    uint32_t exp = (bits >> 7) & 0xFFu;

    if (exp == 0)
    {
        uint32_t frac = bits & 0x7Fu;
        return frac ? (uint16_t)0 : bits;
    }

    if (exp == 0xFFu)
    {
        uint32_t frac = bits & 0x7Fu;
        if (frac != 0)
            return (uint16_t)0;
        return (bits & 0x8000u) ? BF16_MAX_NEG : BF16_MAX_POS;
    }

    return bits;
}

#endif /* FP_FORMATS_H */
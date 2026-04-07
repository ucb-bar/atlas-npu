"""Numba-JIT hot path for InnerProductTreesModel.

Replaces the Python scalar loops in compute_now / compute_lane with compiled
machine code.  The batch kernel processes a full activation batch in one call,
parallelising over batch elements with prange.

Constants are hardcoded for the current InnerProductTreeParams:
  vecLen=32, numLanes=32, accumIntWidth=0
  → anchorHeadroom=7, intWidth=32, sentinel=-1024
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .converters import _E4M3_MUL_TO_PROD_LUT as _lut_list

# Global LUT captured by all @njit functions (built once at import).
MUL_LUT: np.ndarray = np.array(_lut_list, dtype=np.int32)  # shape (65536,)


# ---------------------------------------------------------------------------
# Inner helpers – all @njit so Numba inlines them into compute_lanes_batch.
# intWidth = 32 throughout (was 30 in the previous 32×16 config).
# ---------------------------------------------------------------------------

@njit(cache=True, inline="always")
def _wrap32(v: np.int64) -> np.int64:
    """Wrap to signed 32-bit (intWidth=32)."""
    v = v & np.int64(0xFFFFFFFF)
    if v & np.int64(0x80000000):
        v -= np.int64(0x100000000)
    return v


@njit(cache=True, inline="always")
def _prod_to_aligned(prod_bits: np.int32, anchor: np.int32) -> np.int64:
    """e4m3_prod_to_aligned_int hardcoded to int_width=32."""
    exp_bits = (prod_bits >> np.int32(7)) & np.int32(0x1F)
    if exp_bits == np.int32(0):
        return np.int64(0)
    sign     = (prod_bits >> np.int32(12)) & np.int32(1)
    man      = prod_bits & np.int32(0x7F)
    # left_pad = int_width - sig_width = 32 - 8 = 24
    sig_wide = (np.int64(0x80) | np.int64(man)) << np.int64(24)
    rshift   = np.int32(anchor) - (exp_bits - np.int32(13))  # anchor - unb_exp
    shifted  = sig_wide >> np.int64(rshift)
    mag      = shifted & np.int64(0xFFFFFFFF)
    return _wrap32(-mag if sign else mag)


@njit(cache=True, inline="always")
def _bias_to_aligned(bias_bits: np.int32, anchor: np.int32) -> np.int64:
    """ieee_to_aligned_int for E4M3 bias (mant=3, ieeeBias=7), int_width=32.

    shift_right = anchor - unb_exp - (int_width - 1 - mant_bits)
                = anchor - unb_exp - 28
    """
    sign      = (bias_bits >> np.int32(7)) & np.int32(1)
    exp_field = (bias_bits >> np.int32(3)) & np.int32(0xF)
    frac      = bias_bits & np.int32(0x7)
    if exp_field == np.int32(0) and frac == np.int32(0):
        return np.int64(0)
    unb_exp  = exp_field - np.int32(7)
    full_sig = np.int64(frac) if exp_field == np.int32(0) else (np.int64(0x8) | np.int64(frac))
    rshift   = np.int32(anchor) - unb_exp - np.int32(28)
    if rshift >= np.int32(32):
        mag = np.int64(0)
    elif rshift >= np.int32(0):
        mag = full_sig >> np.int64(rshift)
    elif rshift > np.int32(-32):
        mag = full_sig << np.int64(-rshift)
    else:
        mag = np.int64(0)
    mag &= np.int64(0xFFFFFFFF)
    return _wrap32(-mag if sign else mag)


@njit(cache=True, inline="always")
def _psum_to_aligned(psum_bits: np.int32, anchor: np.int32) -> np.int64:
    """ieee_to_aligned_int for BF16 psum (mant=7, ieeeBias=127), int_width=32.

    shift_right = anchor - unb_exp - (int_width - 1 - mant_bits)
                = anchor - unb_exp - 24
    """
    sign      = (psum_bits >> np.int32(15)) & np.int32(1)
    exp_field = (psum_bits >> np.int32(7)) & np.int32(0xFF)
    frac      = psum_bits & np.int32(0x7F)
    if exp_field == np.int32(0) and frac == np.int32(0):
        return np.int64(0)
    unb_exp  = exp_field - np.int32(127)
    full_sig = np.int64(frac) if exp_field == np.int32(0) else (np.int64(0x80) | np.int64(frac))
    rshift   = np.int32(anchor) - unb_exp - np.int32(24)
    if rshift >= np.int32(32):
        mag = np.int64(0)
    elif rshift >= np.int32(0):
        mag = full_sig >> np.int64(rshift)
    elif rshift > np.int32(-32):
        mag = full_sig << np.int64(-rshift)
    else:
        mag = np.int64(0)
    mag &= np.int64(0xFFFFFFFF)
    return _wrap32(-mag if sign else mag)


@njit(cache=True, inline="always")
def _f64_to_bf16_rne(f64_val: float) -> np.int32:
    """Convert float64 directly to BF16 with single RNE rounding.

    Avoids the double-rounding of float64→float32→BF16.
    The RTL uses HardFloat INToRecFN which rounds once from the
    full-precision integer to BF16.
    """
    tmp = np.empty(1, dtype=np.float64)
    tmp[0] = f64_val
    d = tmp.view(np.uint64)[0]
    sign = np.int32((d >> np.uint64(63)) & np.uint64(1))
    d_exp = np.int32((d >> np.uint64(52)) & np.uint64(0x7FF))
    d_frac = np.int64(d & np.uint64(0xFFFFFFFFFFFFF))

    if d_exp == np.int32(0x7FF):
        if d_frac != np.int64(0):
            return np.int32(0x7FC0)
        return np.int32(0xFF80) if sign else np.int32(0x7F80)
    if d_exp == np.int32(0):
        return np.int32(0)

    unb_exp = d_exp - np.int32(1023)
    full_sig = (np.int64(1) << np.int64(52)) | d_frac  # 53-bit significand

    # Round 53-bit significand to 8-bit (1 implicit + 7 mantissa).
    # Drop the bottom 45 bits with RNE.
    shift = np.int64(45)
    top8 = np.int32(full_sig >> shift)                       # [128, 255]
    guard = np.int32((full_sig >> (shift - np.int64(1))) & np.int64(1))
    sticky_mask = (np.int64(1) << (shift - np.int64(1))) - np.int64(1)
    sticky = np.int32(1) if (full_sig & sticky_mask) != np.int64(0) else np.int32(0)
    lsb = top8 & np.int32(1)
    top8 = top8 + (guard & (sticky | lsb))                  # RNE

    if top8 >= np.int32(256):       # carry from rounding
        top8 = top8 >> np.int32(1)
        unb_exp = unb_exp + np.int32(1)

    biased_exp = unb_exp + np.int32(127)
    if biased_exp >= np.int32(0xFF):
        return np.int32(0xFF7F) if sign else np.int32(0x7F7F)
    if biased_exp <= np.int32(0):
        return np.int32(0)

    return (sign << np.int32(15)) | (biased_exp << np.int32(7)) | (top8 & np.int32(0x7F))


@njit(cache=True, inline="always")
def _aligned_to_bf16(int_in: np.int64, anchor: np.int32) -> np.int32:
    """aligned_int_to_bf16 hardcoded to int_width=32.

    Uses float64 as an exact intermediate (32-bit int fits in 52-bit
    mantissa), then a single RNE round to BF16 — no float32 detour.
    """
    val = _wrap32(int_in)
    if val == np.int64(0):
        return np.int32(0)
    f = math.ldexp(float(val), int(anchor) - 31)  # anchor - (int_width - 1)
    return _f64_to_bf16_rne(f)


@njit(cache=True, inline="always")
def _sanitize_bf16(bits: np.int32) -> np.int32:
    bits = bits & np.int32(0xFFFF)
    exp  = (bits >> np.int32(7)) & np.int32(0xFF)
    if exp == np.int32(0):
        return bits if (bits & np.int32(0x7F)) == np.int32(0) else np.int32(0)
    if exp == np.int32(0xFF):
        if bits & np.int32(0x7F):
            return np.int32(0)
        return np.int32(0xFF7F) if (bits & np.int32(0x8000)) else np.int32(0x7F7F)
    return bits


@njit(cache=True, inline="always")
def _rshift4_rne(x: np.int32) -> np.int32:
    trunc  = (x >> np.int32(4)) & np.int32(0xF)
    guard  = (x >> np.int32(3)) & np.int32(1)
    sticky = np.int32(1) if (x & np.int32(0x7)) != np.int32(0) else np.int32(0)
    return trunc + (guard & (sticky | (trunc & np.int32(1))))


@njit(cache=True, inline="always")
def _bf16_to_e4m3(bf16_bits: np.int32, scale_exp: np.int32) -> np.int32:
    bf16_bits = bf16_bits & np.int32(0xFFFF)
    sign      = (bf16_bits >> np.int32(15)) & np.int32(1)
    exp_bf16  = (bf16_bits >> np.int32(7)) & np.int32(0xFF)
    frac_bf16 = bf16_bits & np.int32(0x7F)
    if exp_bf16 == np.int32(0):
        return np.int32(0)
    if exp_bf16 == np.int32(0xFF):
        if frac_bf16:
            return np.int32(0)
        return np.int32(0xFE) if sign else np.int32(0x7E)
    scaled_unb = exp_bf16 - np.int32(127) + scale_exp
    rn         = _rshift4_rne(np.int32(0x80) | frac_bf16)
    if rn == np.int32(16):
        final_exp = scaled_unb + np.int32(1)
        norm_mant = np.int32(0)
    else:
        final_exp = scaled_unb
        norm_mant = (rn - np.int32(8)) & np.int32(0x7)
    if final_exp > np.int32(8):
        return np.int32(0xFE) if sign else np.int32(0x7E)
    if final_exp >= np.int32(-6):
        return (
            ((sign & np.int32(1)) << np.int32(7))
            | (((final_exp + np.int32(7)) & np.int32(0xF)) << np.int32(3))
            | (norm_mant & np.int32(0x7))
        )
    return np.int32(0)


@njit(cache=True, inline="always")
def _out_stage(bf16_bits: np.int32, out_fmt_sel: np.int32, scale_exp: np.int32) -> np.int32:
    s = _sanitize_bf16(bf16_bits)
    if out_fmt_sel == np.int32(0):   # OutBF16
        return s
    return _bf16_to_e4m3(s, scale_exp) & np.int32(0xFF)


# ---------------------------------------------------------------------------
# Main exported kernel
# ---------------------------------------------------------------------------

@njit(cache=True, parallel=True)
def compute_lanes_batch(
    acts:          np.ndarray,  # (batch, vec_len)  uint8
    wbuf0:         np.ndarray,  # (num_lanes, vec_len) uint8
    wbuf1:         np.ndarray,  # (num_lanes, vec_len) uint8
    bias:          np.ndarray,  # (num_lanes,) uint8
    psums:         np.ndarray,  # (batch, num_lanes) int32  – previous k_tile output
    scale_exp:     np.ndarray,  # (num_lanes,) int32
    buf_read_sel:  bool,
    addend_sel:    np.int32,    # 0=UseAct, 1=UseBias, 2=UsePsum
    out_fmt_sel:   np.int32,    # 0=OutBF16, 1=OutE4M3
    lut:           np.ndarray,  # (65536,) int32
) -> np.ndarray:                # (batch, num_lanes) int32
    """Process all batch elements for one (k_tile, out_tile) step."""
    batch   = acts.shape[0]
    n_lanes = wbuf0.shape[0]
    vec_len = acts.shape[1]

    SENTINEL  = np.int32(-1024)
    PROD_BIAS = np.int32(13)
    ANCHOR_HR = np.int32(7)

    out = np.empty((batch, n_lanes), dtype=np.int32)

    for b_idx in prange(batch):
        act_row  = acts[b_idx]
        psum_row = psums[b_idx]

        for lane_idx in range(n_lanes):
            weights = wbuf1[lane_idx] if buf_read_sel else wbuf0[lane_idx]

            # ── Pass 1: compute + cache E4M3 products, find max exp ──────
            prods  = np.empty(vec_len, dtype=np.int32)
            max_pe = SENTINEL
            for i in range(vec_len):
                a    = np.int32(act_row[i])   & np.int32(0xFF)
                w    = np.int32(weights[i])   & np.int32(0xFF)
                prod = lut[(a << np.int32(8)) | w]
                prods[i] = prod
                exp_bits = (prod >> np.int32(7)) & np.int32(0x1F)
                pe = SENTINEL if exp_bits == np.int32(0) else (exp_bits - PROD_BIAS)
                if pe > max_pe:
                    max_pe = pe

            # ── Addend exponent ──────────────────────────────────────────
            addend_exp = SENTINEL
            b_bits = np.int32(bias[lane_idx]) & np.int32(0xFF)
            p_bits = psum_row[lane_idx]       & np.int32(0xFFFF)

            if addend_sel == np.int32(1):    # UseBias
                bf = (b_bits >> np.int32(3)) & np.int32(0xF)
                if ((b_bits >> np.int32(3)) & np.int32(0x1F)) != np.int32(0):
                    addend_exp = bf - np.int32(7)
            elif addend_sel == np.int32(2):  # UsePsum
                pef = (p_bits >> np.int32(7)) & np.int32(0xFF)
                pfr =  p_bits & np.int32(0x7F)
                if not (pef == np.int32(0) and pfr == np.int32(0)):
                    addend_exp = pef - np.int32(127)

            anchor = (max_pe if max_pe >= addend_exp else addend_exp) + ANCHOR_HR

            # ── Pass 2: accumulate from cached products ───────────────────
            # RTL uses treeReduce with +& (width-extending add); intermediate
            # sums can exceed intWidth.  Truncation happens once at the end.
            prod_sum = np.int64(0)
            for i in range(vec_len):
                prod_sum = prod_sum + _prod_to_aligned(prods[i], anchor)

            # ── Addend ───────────────────────────────────────────────────
            if addend_sel == np.int32(1):
                addend_int = _bias_to_aligned(b_bits, anchor)
            elif addend_sel == np.int32(2):
                addend_int = _psum_to_aligned(p_bits, anchor)
            else:
                addend_int = np.int64(0)

            total_int = _wrap32(prod_sum + addend_int)
            bf16_bits = _aligned_to_bf16(total_int, anchor)
            out[b_idx, lane_idx] = (
                _out_stage(bf16_bits, out_fmt_sel, np.int32(scale_exp[lane_idx]))
                & np.int32(0xFFFF)
            )

    return out


def warmup() -> None:
    """Trigger JIT compilation with dummy data.  Call once at startup."""
    dummy_acts  = np.zeros((1, 32), dtype=np.uint8)
    dummy_wbuf  = np.zeros((32, 32), dtype=np.uint8)
    dummy_bias  = np.zeros(32, dtype=np.uint8)
    dummy_psums = np.zeros((1, 32), dtype=np.int32)
    dummy_sexp  = np.zeros(32, dtype=np.int32)
    compute_lanes_batch(
        dummy_acts, dummy_wbuf, dummy_wbuf,
        dummy_bias, dummy_psums, dummy_sexp,
        False, np.int32(0), np.int32(1), MUL_LUT,
    )

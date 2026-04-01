#!/usr/bin/env python3
"""
gen_mxu_qk_scores.py

Generate chained transformer-style test vectors for:
    Q = X @ WQ
    K = X @ WK
    scores = Q @ K^T

Q and K are popped as FP8 using per-tensor E8M0 scales loaded by SELI.

Shapes:
    X      : 32 x 64
    WQ     : 64 x 64
    WK     : 64 x 64
    Q      : 32 x 64   (FP8 intermediate, checked)
    K      : 32 x 64   (FP8 intermediate, checked)
    scores : 32 x 32   (BF16 final, checked)
"""

import os
import sys
import math
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import (
    rand_matrix_fp8_safe,
    quantize_fp8,
    quantize_bf16,
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    float_to_bf16_bits,
    bf16_bits_to_float,
    fp8_e4m3_bits_to_float,
    preloads_from_words_packed,
    checks_from_words_packed,
    emit_test_data,
)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 900

# Byte offsets
X_BASE      = 0x0000
WQ_BASE     = 0x0800
WK_BASE     = 0x1800
Q_BASE      = 0x2800
K_BASE      = 0x3000
SCORES_BASE = 0x4000

# Per-tensor E8M0 scales used by SELI in assembly.
# 127 => scaleExp = 0 => multiply by 2^0 = 1
Q_SCALE_E8M0 = 127
K_SCALE_E8M0 = 127


def beat_offset(byte_off: int) -> int:
    assert byte_off % 32 == 0
    return byte_off // 32


def emit_fp8_preload(preloads, byte_off, tile):
    preloads += preloads_from_words_packed(beat_offset(byte_off), matrix_to_fp8_words(tile))


def emit_fp8_check(checks, byte_off, tile):
    checks += checks_from_words_packed(beat_offset(byte_off), matrix_to_fp8_words(tile))


def emit_bf16_split_check(checks, byte_off_lo, byte_off_hi, tile32):
    left = tile32[:, :16]
    right = tile32[:, 16:]
    checks += checks_from_words_packed(beat_offset(byte_off_lo), matrix_to_bf16_words(left))
    checks += checks_from_words_packed(beat_offset(byte_off_hi), matrix_to_bf16_words(right))


# ------------------------------------------------------------------
# FP8 conversion model (matches hardware RTL exactly)
# ------------------------------------------------------------------

def round_right_shift_4_rne(x: int) -> int:
    """
    Match hardware roundRightShift4RNE:
      trunc  = x >> 4
      guard  = bit 3
      sticky = OR(bits 2:0)
      inc    = guard && (sticky || lsb(trunc))
    """
    trunc = (x >> 4) & 0xF
    guard = (x >> 3) & 0x1
    sticky = 1 if (x & 0x7) != 0 else 0
    lsb = trunc & 0x1
    inc = 1 if (guard and (sticky or lsb)) else 0
    return trunc + inc  # 0..16


def bf16_to_e4m3_scaled_bits(bf16_bits: int, scale_e8m0: int) -> int:
    """
    Exact software mirror of BF16ScaleToE4M3 RTL.
    Returns raw 8-bit E4M3 bits.

    Scale convention: scaledUnbExp = unbExp + scaleExp  (additive, per RTL)
    Subnormals: flushed to zero (per RTL)
    """
    sign = (bf16_bits >> 15) & 0x1
    exp_bf16 = (bf16_bits >> 7) & 0xFF
    frac_bf16 = bf16_bits & 0x7F

    is_zero = (exp_bf16 == 0) and (frac_bf16 == 0)
    is_sub = (exp_bf16 == 0) and (frac_bf16 != 0)
    is_inf = (exp_bf16 == 0xFF) and (frac_bf16 == 0)
    is_nan = (exp_bf16 == 0xFF) and (frac_bf16 != 0)

    E4M3_MAX_POS = 0x7E
    E4M3_MAX_NEG = 0xFE

    if is_zero or is_sub or is_nan:
        return 0
    if is_inf:
        return E4M3_MAX_NEG if sign else E4M3_MAX_POS

    unb_exp = exp_bf16 - 127
    scale_exp = scale_e8m0 - 127
    scaled_unb_exp = unb_exp + scale_exp   # additive, per RTL

    mant8 = (1 << 7) | frac_bf16
    rounded_norm = round_right_shift_4_rne(mant8)  # 8..16 typically
    norm_carry = (rounded_norm == 16)

    final_unb_exp = scaled_unb_exp + (1 if norm_carry else 0)

    rounded_minus_8 = rounded_norm - 8
    norm_mant = 0 if norm_carry else (rounded_minus_8 & 0x7)

    if final_unb_exp > 8:
        return E4M3_MAX_NEG if sign else E4M3_MAX_POS
    elif final_unb_exp >= -6:
        exp_e4 = final_unb_exp + 7
        return ((sign & 0x1) << 7) | ((exp_e4 & 0xF) << 3) | (norm_mant & 0x7)
    else:
        # Subnormal: flushed to zero (per RTL)
        return 0


def bf16_to_e4m3_scaled_float(x: float, scale_e8m0: int) -> float:
    bits = float_to_bf16_bits(float(x))
    e4m3_bits = bf16_to_e4m3_scaled_bits(bits, scale_e8m0)
    return fp8_e4m3_bits_to_float(e4m3_bits)


def bf16_tile_to_scaled_fp8(tile_bf16: np.ndarray, scale_e8m0: int) -> np.ndarray:
    out = np.zeros_like(tile_bf16, dtype=np.float32)
    for r in range(tile_bf16.shape[0]):
        for c in range(tile_bf16.shape[1]):
            out[r, c] = bf16_to_e4m3_scaled_float(tile_bf16[r, c], scale_e8m0)
    return out


# ------------------------------------------------------------------
# MXU accumulation model (matches hardware accbuf behavior)
#
# Key insight: the accbuf stores BF16.  Between VMATMUL (fresh write)
# and VMATMUL.ACC (accumulate), the partial sum is truncated to BF16.
# The IPT accumulates products in FP32.
#
# VMATMUL:     result = BF16(IPT_FP32(act0 @ wt0))
# VMATMUL.ACC: result = BF16(IPT_FP32(act1 @ wt1) + FP32(accbuf))
# ------------------------------------------------------------------

def ipt_matmul_fp32(act: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """Inner product tree matmul in FP32 (matches hardware IPT precision)."""
    return act.astype(np.float32) @ wt.astype(np.float32)


def mxu_to_fp8_tile(act0, act1, wt0, wt1, scale_e8m0):
    """
    Model VMATMUL + VMATMUL.ACC → VMATPOP.FP8 sequence.

    Phase 1 (VMATMUL):     partial1 = FP32(act0 @ wt0) → BF16 → accbuf
    Phase 2 (VMATMUL.ACC): partial2 = FP32(act1 @ wt1)
                           acc = FP32(partial2 + BF16(partial1)) → BF16 → accbuf
    Phase 3 (VMATPOP.FP8): BF16 → E4M3 with E8M0 scale
    """
    # Phase 1: VMATMUL — fresh write to accbuf
    partial1_fp32 = ipt_matmul_fp32(act0, wt0)
    partial1_bf16 = quantize_bf16(partial1_fp32).astype(np.float32)

    # Phase 2: VMATMUL.ACC — accumulate with existing accbuf value
    partial2_fp32 = ipt_matmul_fp32(act1, wt1)
    acc_fp32 = partial2_fp32 + partial1_bf16   # FP32 addition
    acc_bf16 = quantize_bf16(acc_fp32).astype(np.float32)

    # Phase 3: VMATPOP.FP8 — convert BF16 to E4M3
    return bf16_tile_to_scaled_fp8(acc_bf16, scale_e8m0).astype(np.float32)


def mxu_to_bf16_tile(act0, act1, wt0, wt1):
    """
    Model VMATMUL + VMATMUL.ACC → VMATPOP.BF16 sequence.
    Same accumulation model but pop as BF16 instead of FP8.
    """
    partial1_fp32 = ipt_matmul_fp32(act0, wt0)
    partial1_bf16 = quantize_bf16(partial1_fp32).astype(np.float32)

    partial2_fp32 = ipt_matmul_fp32(act1, wt1)
    acc_fp32 = partial2_fp32 + partial1_bf16
    return quantize_bf16(acc_fp32).astype(np.float32)


# ------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------

X  = quantize_fp8(rand_matrix_fp8_safe(32, 64, seed=SEED + 0)).astype(np.float32)
WQ = quantize_fp8(rand_matrix_fp8_safe(64, 64, seed=SEED + 1)).astype(np.float32)
WK = quantize_fp8(rand_matrix_fp8_safe(64, 64, seed=SEED + 2)).astype(np.float32)

X_d0 = X[:, 0:32]
X_d1 = X[:, 32:64]

WQ_n0_d0 = WQ[0:32,  0:32]
WQ_n0_d1 = WQ[32:64, 0:32]
WQ_n1_d0 = WQ[0:32,  32:64]
WQ_n1_d1 = WQ[32:64, 32:64]

WK_n0_d0 = WK[0:32,  0:32]
WK_n0_d1 = WK[32:64, 0:32]
WK_n1_d0 = WK[0:32,  32:64]
WK_n1_d1 = WK[32:64, 32:64]

# ------------------------------------------------------------------
# Reference intermediates
# ------------------------------------------------------------------

Q_n0 = mxu_to_fp8_tile(X_d0, X_d1, WQ_n0_d0, WQ_n0_d1, Q_SCALE_E8M0)
Q_n1 = mxu_to_fp8_tile(X_d0, X_d1, WQ_n1_d0, WQ_n1_d1, Q_SCALE_E8M0)

K_n0 = mxu_to_fp8_tile(X_d0, X_d1, WK_n0_d0, WK_n0_d1, K_SCALE_E8M0)
K_n1 = mxu_to_fp8_tile(X_d0, X_d1, WK_n1_d0, WK_n1_d1, K_SCALE_E8M0)

# ------------------------------------------------------------------
# Final scores = Q @ K^T
#
# Assembly pushes K directly (no VTRPOSE); MXU internally transposes
# pushed weights, so it computes Q @ K^T.
#
# Uses same accumulation model: VMATMUL then VMATMUL.ACC with
# intermediate BF16 truncation in accbuf.
# ------------------------------------------------------------------

scores = mxu_to_bf16_tile(Q_n0, Q_n1, K_n0.T, K_n1.T)

# ------------------------------------------------------------------
# Emit preloads/checks
# ------------------------------------------------------------------

preloads = []
checks = []

emit_fp8_preload(preloads, X_BASE + 0x0000, X_d0)
emit_fp8_preload(preloads, X_BASE + 0x0400, X_d1)

emit_fp8_preload(preloads, WQ_BASE + 0x0000, WQ_n0_d0)
emit_fp8_preload(preloads, WQ_BASE + 0x0400, WQ_n0_d1)
emit_fp8_preload(preloads, WQ_BASE + 0x0800, WQ_n1_d0)
emit_fp8_preload(preloads, WQ_BASE + 0x0C00, WQ_n1_d1)

emit_fp8_preload(preloads, WK_BASE + 0x0000, WK_n0_d0)
emit_fp8_preload(preloads, WK_BASE + 0x0400, WK_n0_d1)
emit_fp8_preload(preloads, WK_BASE + 0x0800, WK_n1_d0)
emit_fp8_preload(preloads, WK_BASE + 0x0C00, WK_n1_d1)

# FP8 intermediate checks (exact)
emit_fp8_check(checks, Q_BASE + 0x0000, Q_n0)
emit_fp8_check(checks, Q_BASE + 0x0400, Q_n1)

emit_fp8_check(checks, K_BASE + 0x0000, K_n0)
emit_fp8_check(checks, K_BASE + 0x0400, K_n1)

# BF16 scores check (exact)
emit_bf16_split_check(checks, SCORES_BASE + 0x0000, SCORES_BASE + 0x0400, scores)

emit_test_data(preloads, checks, timeout=300000)

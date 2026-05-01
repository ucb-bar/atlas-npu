#!/usr/bin/env python3
"""
gen_mxu1_qk_scores.py

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
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import (
    rand_matrix_fp8_safe,
    quantize_fp8,
    float_to_bf16_bits,
    fp8_e4m3_bits_to_float,
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    preloads_from_words_packed,
    checks_from_words_packed,
    emit_test_data,
)
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 900

# Byte offsets
X_BASE      = 0x0000
WQ_BASE     = 0x0800
WK_BASE     = 0x1800
Q_BASE      = 0x2800
K_BASE      = 0x3000
SCORES_BASE = 0x4000
ZERO_BIAS_BASE = 0x4800

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
# BF16 → E4M3 conversion (models VMATPOP.FP8 stage)
#
# The MXU1 accumulates in BF16 (accbuf), then VMATPOP.FP8 converts
# the BF16 result to E4M3 with an E8M0 scale.  This is a separate
# stage from the accumulation, so we model it separately.
# ------------------------------------------------------------------

def round_right_shift_4_rne(x: int) -> int:
    """Match hardware roundRightShift4RNE."""
    trunc = (x >> 4) & 0xF
    guard = (x >> 3) & 0x1
    sticky = 1 if (x & 0x7) != 0 else 0
    lsb = trunc & 0x1
    inc = 1 if (guard and (sticky or lsb)) else 0
    return trunc + inc


def bf16_to_e4m3_scaled_bits(bf16_bits: int, scale_e8m0: int) -> int:
    """
    Exact software mirror of BF16ScaleToE4M3 RTL.
    Returns raw 8-bit E4M3 bits.
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
    scaled_unb_exp = unb_exp + scale_exp

    mant8 = (1 << 7) | frac_bf16
    rounded_norm = round_right_shift_4_rne(mant8)
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
# Instantiate IPT functional model (BF16 output only)
#
# The hardware flow for FP8 intermediates is:
#   VMATMUL → accbuf (BF16) → VMATMUL.ACC → accbuf (BF16) → VMATPOP.FP8
#
# IPTLinearRTLFunction with OutE4M3 is NOT correct for multi-k-tile
# accumulation because it stores E4M3 in the psum between k-tiles,
# but the hardware accbuf always holds BF16.  So we always accumulate
# in BF16 and apply the FP8 pop conversion separately.
# ------------------------------------------------------------------

ipt_bf16 = IPTLinearRTLFunction(
    vec_len=32, num_lanes=32, pipeline_depth=1,
    out_fmt_sel=OutputFmtSel.OutBF16,
)

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
ZERO_BIAS = np.zeros((32, 32), dtype=np.float32)

# ------------------------------------------------------------------
# Reference intermediates via IPT functional model
#
# IPTLinearRTLFunction computes F.linear: y = x @ w^T
# So to compute X @ WQ, pass w = WQ^T (shape 64x64).
#
# Step 1: Accumulate in BF16 (models VMATMUL + VMATMUL.ACC → accbuf)
# Step 2: Convert BF16 → E4M3 with scale (models VMATPOP.FP8)
# ------------------------------------------------------------------

X_t  = torch.from_numpy(X)
WQ_t = torch.from_numpy(WQ.T.copy())  # (64, 64) — F.linear: X @ WQ^T^T = X @ WQ
WK_t = torch.from_numpy(WK.T.copy())  # (64, 64)

# Step 1: BF16 accumulation
Q_bf16 = ipt_bf16(X_t, WQ_t, scale_exp=0).numpy().astype(np.float32)   # 32 x 64
K_bf16 = ipt_bf16(X_t, WK_t, scale_exp=0).numpy().astype(np.float32)   # 32 x 64

# Step 2: BF16 → E4M3 pop (VMATPOP.FP8)
Q_full = bf16_tile_to_scaled_fp8(Q_bf16, Q_SCALE_E8M0)   # 32 x 64, FP8-valued floats
K_full = bf16_tile_to_scaled_fp8(K_bf16, K_SCALE_E8M0)   # 32 x 64, FP8-valued floats

Q_n0 = Q_full[:, :32].astype(np.float32)
Q_n1 = Q_full[:, 32:].astype(np.float32)

K_n0 = K_full[:, :32].astype(np.float32)
K_n1 = K_full[:, 32:].astype(np.float32)

# ------------------------------------------------------------------
# Final scores = Q @ K^T
#
# F.linear: y = x @ w^T.  Pass w = K (32x64) → Q @ K^T.
#
# This is a single output tile (32x32) with 2 k-tiles, popped as BF16.
# No FP8 conversion needed.
# ------------------------------------------------------------------

Q_t = torch.from_numpy(Q_full.astype(np.float32))
K_t = torch.from_numpy(K_full.astype(np.float32))

scores = ipt_bf16(Q_t, K_t, scale_exp=0).numpy().astype(np.float32)  # 32 x 32

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
emit_fp8_preload(preloads, ZERO_BIAS_BASE, ZERO_BIAS)

# FP8 intermediate checks (exact)
emit_fp8_check(checks, Q_BASE + 0x0000, Q_n0)
emit_fp8_check(checks, Q_BASE + 0x0400, Q_n1)

emit_fp8_check(checks, K_BASE + 0x0000, K_n0)
emit_fp8_check(checks, K_BASE + 0x0400, K_n1)

# BF16 scores check (exact)
emit_bf16_split_check(checks, SCORES_BASE + 0x0000, SCORES_BASE + 0x0400, scores)

emit_test_data(preloads, checks, timeout=300000)

#!/usr/bin/env python3
"""
gen_mxu_chain_vpu_reduce.py

Generate test vectors for mxu_chain_vpu_reduce.S.

Attention-like pipeline:
    Phase A: MXU1 → Q = X @ WQ  (FP8 pop)
    Phase B: MXU1 → K = X @ WK  (FP8 pop)
    Phase C: MXU1 → S = Q @ K^T (BF16 pop)
    Phase D: VPU  → biased = S + bias_vpu

Also: VPU vmin(A, B) runs during Phase A→B transition.

Usage:
    python3 gen_mxu_chain_vpu_reduce.py [seed]
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (
    rand_matrix_fp8_safe,
    quantize_fp8,
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    preloads_from_words_packed,
    checks_from_words_packed,
    emit_test_data,
)
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    float_to_bf16,
    pack_u16_le,
    run_binary_rows,
)
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction, e4m3_words
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 333
TILE = 32
SCALE_E8M0 = 127  # identity scale (2^0 = 1)

# DRAM byte offsets
X_BASE         = 0x0000
WQ_BASE        = 0x0400
WK_BASE        = 0x0800
BIAS_FP8_BASE  = 0x0C00
A_VPU_B0_BASE  = 0x1000
A_VPU_B1_BASE  = 0x1400
B_VPU_B0_BASE  = 0x1800
B_VPU_B1_BASE  = 0x1C00
BIAS_VPU_B0    = 0x2000
BIAS_VPU_B1    = 0x2400
Q_SCRATCH      = 0x2800
K_SCRATCH      = 0x2C00
SCORES_LO      = 0x3000
SCORES_HI      = 0x3400
VMIN_B0        = 0x3800
VMIN_B1        = 0x3C00
BIASED_B0      = 0x4000
BIASED_B1      = 0x4400


def beat_offset(byte_off: int) -> int:
    assert byte_off % 32 == 0
    return byte_off // 32


def emit_fp8_preload(preloads, byte_off, tile):
    preloads += preloads_from_words_packed(
        beat_offset(byte_off), matrix_to_fp8_words(tile)
    )


def emit_fp8_check(checks, byte_off, tile):
    checks += checks_from_words_packed(
        beat_offset(byte_off), e4m3_words(tile)
    )


def emit_bf16_split_check(checks, byte_off_lo, byte_off_hi, tile32x32):
    left = tile32x32[:, :16]
    right = tile32x32[:, 16:]
    checks += checks_from_words_packed(
        beat_offset(byte_off_lo), matrix_to_bf16_words(left)
    )
    checks += checks_from_words_packed(
        beat_offset(byte_off_hi), matrix_to_bf16_words(right)
    )


def float_matrix_to_bf16_rows(mat):
    rows = []
    for r in range(mat.shape[0]):
        rows.append([float_to_bf16(float(mat[r, c])) for c in range(mat.shape[1])])
    return rows


def emit_bf16_rows_preload(preloads, byte_off, rows):
    for r, row in enumerate(rows):
        preloads.append({
            "word_offset": beat_offset(byte_off) + r,
            "data": pack_u16_le(row),
        })


def emit_bf16_rows_check(checks, byte_off_b0, byte_off_b1, rows_b0, rows_b1):
    for r, row in enumerate(rows_b0):
        checks.append({
            "word_offset": beat_offset(byte_off_b0) + r,
            "expected": pack_u16_le(row),
        })
    for r, row in enumerate(rows_b1):
        checks.append({
            "word_offset": beat_offset(byte_off_b1) + r,
            "expected": pack_u16_le(row),
        })


# ------------------------------------------------------------------
# Functional models
# ------------------------------------------------------------------

ipt_bf16 = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutBF16)

ipt_fp8 = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutE4M3)

# ------------------------------------------------------------------
# Generate inputs
# ------------------------------------------------------------------

X  = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 0)).astype(np.float32)
WQ = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 1)).astype(np.float32)
WK = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 2)).astype(np.float32)

# VPU inputs
A_vpu = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 3).astype(np.float32)
B_vpu = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 4).astype(np.float32)
bias_vpu = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 5).astype(np.float32)

A_vpu_b0_rows = float_matrix_to_bf16_rows(A_vpu[:, :16])
A_vpu_b1_rows = float_matrix_to_bf16_rows(A_vpu[:, 16:])
B_vpu_b0_rows = float_matrix_to_bf16_rows(B_vpu[:, :16])
B_vpu_b1_rows = float_matrix_to_bf16_rows(B_vpu[:, 16:])
bias_vpu_b0_rows = float_matrix_to_bf16_rows(bias_vpu[:, :16])
bias_vpu_b1_rows = float_matrix_to_bf16_rows(bias_vpu[:, 16:])

zero_bias = np.zeros((TILE, TILE), dtype=np.float32)

# ------------------------------------------------------------------
# Phase A: MXU1 → Q = X @ WQ (FP32 acc → FP8 pop)
#
# MXU1 (IPT) uses F.linear: y = x @ w^T. Pass w = WQ^T.
# Accumulate in FP32, then pop as FP8 with identity scale.
# ------------------------------------------------------------------

X_t  = torch.from_numpy(X)
WQ_t = torch.from_numpy(WQ.T.copy())
b0_t = torch.zeros(TILE)

Q_fp8 = ipt_fp8(X_t, WQ_t, scale_exp=SCALE_E8M0 - 127).numpy()

# ------------------------------------------------------------------
# Phase B: MXU1 → K = X @ WK (FP32 acc → FP8 pop)
# ------------------------------------------------------------------

WK_t = torch.from_numpy(WK.T.copy())

K_fp8 = ipt_fp8(X_t, WK_t, scale_exp=SCALE_E8M0 - 127).numpy()

# ------------------------------------------------------------------
# Phase C: MXU1 → S = Q @ K^T (BF16 pop)
#
# K is pushed directly as weights (no transpose) so MXU1 computes
# Q @ K^T. F.linear: y = x @ w^T, pass w = K (32×32) → Q @ K^T.
# ------------------------------------------------------------------

Q_t = torch.from_numpy(Q_fp8.astype(np.float32))
K_t = torch.from_numpy(K_fp8.astype(np.float32))

scores = ipt_bf16(Q_t, K_t, b_q=b0_t, scale_exp=0).numpy().astype(np.float32)

# ------------------------------------------------------------------
# Phase D: VPU → biased = vadd(scores, bias_vpu)
# ------------------------------------------------------------------

scores_b0_rows = float_matrix_to_bf16_rows(scores[:, :16])
scores_b1_rows = float_matrix_to_bf16_rows(scores[:, 16:])

biased_b0_rows = run_binary_rows("add", scores_b0_rows, bias_vpu_b0_rows)
biased_b1_rows = run_binary_rows("add", scores_b1_rows, bias_vpu_b1_rows)

# ------------------------------------------------------------------
# VPU-1: vmin(A, B) — runs during Phase A→B transition
# ------------------------------------------------------------------

vmin_b0_rows = run_binary_rows("pairmin", A_vpu_b0_rows, B_vpu_b0_rows)
vmin_b1_rows = run_binary_rows("pairmin", A_vpu_b1_rows, B_vpu_b1_rows)

# ------------------------------------------------------------------
# Emit preloads
# ------------------------------------------------------------------

preloads = []

# FP8 inputs
emit_fp8_preload(preloads, X_BASE, X)
emit_fp8_preload(preloads, WQ_BASE, WQ)
emit_fp8_preload(preloads, WK_BASE, WK)
emit_fp8_preload(preloads, BIAS_FP8_BASE, zero_bias)

# VPU BF16 inputs
emit_bf16_rows_preload(preloads, A_VPU_B0_BASE, A_vpu_b0_rows)
emit_bf16_rows_preload(preloads, A_VPU_B1_BASE, A_vpu_b1_rows)
emit_bf16_rows_preload(preloads, B_VPU_B0_BASE, B_vpu_b0_rows)
emit_bf16_rows_preload(preloads, B_VPU_B1_BASE, B_vpu_b1_rows)
emit_bf16_rows_preload(preloads, BIAS_VPU_B0, bias_vpu_b0_rows)
emit_bf16_rows_preload(preloads, BIAS_VPU_B1, bias_vpu_b1_rows)

# ------------------------------------------------------------------
# Emit checks
# ------------------------------------------------------------------

checks = []

# Q FP8 intermediate (stored at scratch for Phase C reload)
emit_fp8_check(checks, Q_SCRATCH, Q_fp8)

# K FP8 intermediate
emit_fp8_check(checks, K_SCRATCH, K_fp8)

# BF16 scores
emit_bf16_split_check(checks, SCORES_LO, SCORES_HI, scores)

# VPU vmin
emit_bf16_rows_check(checks, VMIN_B0, VMIN_B1, vmin_b0_rows, vmin_b1_rows)

# VPU biased scores
emit_bf16_rows_check(checks, BIASED_B0, BIASED_B1, biased_b0_rows, biased_b1_rows)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

sys.stderr.write(
    f"[gen_mxu_chain_vpu_reduce] seed={SEED}\n"
    f"  Q_fp8 range: [{Q_fp8.min():.4f}, {Q_fp8.max():.4f}]\n"
    f"  K_fp8 range: [{K_fp8.min():.4f}, {K_fp8.max():.4f}]\n"
    f"  scores range: [{scores.min():.4f}, {scores.max():.4f}]\n"
    f"  preloads={len(preloads)} checks={len(checks)}\n"
)

emit_test_data(preloads, checks, timeout=1000000)

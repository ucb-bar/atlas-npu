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
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    preloads_from_words_packed,
    checks_from_words_packed,
    emit_test_data,
)
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction, e4m3_words
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
    checks += checks_from_words_packed(beat_offset(byte_off), e4m3_words(tile))


def emit_bf16_split_check(checks, byte_off_lo, byte_off_hi, tile32):
    left = tile32[:, :16]
    right = tile32[:, 16:]
    checks += checks_from_words_packed(beat_offset(byte_off_lo), matrix_to_bf16_words(left))
    checks += checks_from_words_packed(beat_offset(byte_off_hi), matrix_to_bf16_words(right))


# ------------------------------------------------------------------
# IPT functional model. The FP8 intermediates take
#   VMATMUL → accbuf (FP32) → VMATMUL.ACC → accbuf (FP32) → VMATPOP.FP8,
# which OutE4M3 models; the scores pop as BF16.
# ------------------------------------------------------------------

ipt_bf16 = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutBF16)
ipt_fp8 = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutE4M3)

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
# The FP32 accumulator holds both k-tiles; VMATPOP.FP8 converts it with the scale.
# ------------------------------------------------------------------

X_t  = torch.from_numpy(X)
WQ_t = torch.from_numpy(WQ.T.copy())  # (64, 64) — F.linear: X @ WQ^T^T = X @ WQ
WK_t = torch.from_numpy(WK.T.copy())  # (64, 64)

Q_full = ipt_fp8(X_t, WQ_t, scale_exp=Q_SCALE_E8M0 - 127).numpy()   # 32 x 64, FP8-valued floats
K_full = ipt_fp8(X_t, WK_t, scale_exp=K_SCALE_E8M0 - 127).numpy()   # 32 x 64, FP8-valued floats

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

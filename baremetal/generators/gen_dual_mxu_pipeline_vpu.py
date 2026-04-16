#!/usr/bin/env python3
"""
gen_dual_mxu_pipeline_vpu.py

Generate test vectors for dual_mxu_pipeline_vpu.S.

Both MXUs perform 2-k-tile (K=64) matrix multiplications:
    MXU0 (SA):  C0 = X @ W0   (32×64 @ 64×32 → 32×32 BF16)
    MXU1 (IPT): C1 = X @ W1   (32×64 @ 64×32 → 32×32 BF16)

VPU computes:
    D = vsub(A, B)     (BF16 element-wise)
    E = vmax(A, B)     (BF16 element-wise)
    F = vsub(C0, C1)   (cross-unit post-processing)

NOTE on VPU bank allocation in assembly:
    VSUB.BF16  2, 12, 14  → result in m2,m3
    VMAX.BF16  4, 12, 14  → result in m4,m5

    The VPU ops read from separate source registers per bank:
      vsub reads bank0 pair: m11 (A_b0), m13 (B_b0) → writes m2,m3
      vmax reads bank1 pair: m12 (A_b1), m14 (B_b1) → writes m4,m5

    This is a deliberate split: each VPU instruction processes one
    bank-pair of the full 32×32 BF16 tensor. The assembly stores
    each result at the corresponding DRAM bank offset.

Usage:
    python3 gen_dual_mxu_pipeline_vpu.py [seed]
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
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TILE = 32
K_FULL = 64  # 2 k-tiles

# DRAM byte offsets (must match assembly)
X_D0_BASE      = 0x0000
X_D1_BASE      = 0x0400
W0_A_BASE      = 0x0800
W0_B_BASE      = 0x0C00
W1_A_BASE      = 0x1000
W1_B_BASE      = 0x1400
BIAS_BASE      = 0x1800
A_VPU_B0_BASE  = 0x1C00
A_VPU_B1_BASE  = 0x2000
B_VPU_B0_BASE  = 0x2400
B_VPU_B1_BASE  = 0x2800
C0_LO_BASE     = 0x2C00
C0_HI_BASE     = 0x3000
C1_LO_BASE     = 0x3400
C1_HI_BASE     = 0x3800
SUB_B0_BASE    = 0x3C00
SUB_B1_BASE    = 0x4000
MAX_B0_BASE    = 0x4400
MAX_B1_BASE    = 0x4800
DIFF_B0_BASE   = 0x4C00
DIFF_B1_BASE   = 0x5000


def beat_offset(byte_off: int) -> int:
    assert byte_off % 32 == 0
    return byte_off // 32


def emit_fp8_preload(preloads, byte_off, tile):
    preloads += preloads_from_words_packed(
        beat_offset(byte_off), matrix_to_fp8_words(tile)
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


def float_matrix_to_bf16_rows(mat: np.ndarray) -> list[list[int]]:
    rows = []
    for r in range(mat.shape[0]):
        row = [float_to_bf16(float(mat[r, c])) for c in range(mat.shape[1])]
        rows.append(row)
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

sa_bf16 = SARTLLinearFunction(
    rows=32, cols=32,
    out_fmt_sel=OutputFmtSel.OutBF16,
)

ipt_bf16 = IPTLinearRTLFunction(
    vec_len=32, num_lanes=32, pipeline_depth=1,
    out_fmt_sel=OutputFmtSel.OutBF16,
)

# ------------------------------------------------------------------
# Generate inputs
# ------------------------------------------------------------------

# Shared activations X (32×64 FP8)
X = quantize_fp8(rand_matrix_fp8_safe(TILE, K_FULL, seed=SEED + 0)).astype(np.float32)
X_d0 = X[:, :32]   # slab 0
X_d1 = X[:, 32:]   # slab 1

# MXU0 weights W0 (64×32 FP8, stored as 2 k-tiles)
W0 = quantize_fp8(rand_matrix_fp8_safe(K_FULL, TILE, seed=SEED + 1)).astype(np.float32)
W0_A = W0[:32, :]   # k-tile 0
W0_B = W0[32:, :]   # k-tile 1

# MXU1 weights W1 (64×32 FP8)
W1 = quantize_fp8(rand_matrix_fp8_safe(K_FULL, TILE, seed=SEED + 2)).astype(np.float32)
W1_A = W1[:32, :]
W1_B = W1[32:, :]

# VPU inputs (BF16, 32×32)
A_vpu_float = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 3).astype(np.float32)
B_vpu_float = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 4).astype(np.float32)

A_vpu_b0_rows = float_matrix_to_bf16_rows(A_vpu_float[:, :16])
A_vpu_b1_rows = float_matrix_to_bf16_rows(A_vpu_float[:, 16:])
B_vpu_b0_rows = float_matrix_to_bf16_rows(B_vpu_float[:, :16])
B_vpu_b1_rows = float_matrix_to_bf16_rows(B_vpu_float[:, 16:])

# Zero bias
zero_bias = np.zeros((TILE, TILE), dtype=np.float32)

# ------------------------------------------------------------------
# Reference: MXU0 (SA) — C0 = X @ W0 with zero bias
#
# SA F.linear: y = x @ w^T.  We want X @ W0 = X @ (W0^T)^T.
# So pass w = W0^T (shape 32×64).
# ------------------------------------------------------------------

X_t  = torch.from_numpy(X)
W0_t = torch.from_numpy(W0.T.copy())   # (32, 64) — F.linear convention
b0_t = torch.zeros(TILE)

C0 = sa_bf16(X_t, W0_t, b_q=b0_t, scale_exp=0).numpy().astype(np.float32)

# ------------------------------------------------------------------
# Reference: MXU1 (IPT) — C1 = X @ W1
# ------------------------------------------------------------------

W1_t = torch.from_numpy(W1.T.copy())   # (32, 64)

C1 = ipt_bf16(X_t, W1_t, scale_exp=0).numpy().astype(np.float32)

# ------------------------------------------------------------------
# Reference: VPU ops
#
# The assembly issues BF16 register-pair ops:
#   VSUB.BF16 2, 12, 14
#   VMAX.BF16 4, 12, 14
#
# Each instruction takes a full register pair as input and produces a
# full register pair as output. The VPU software model below still
# computes one physical bank (32 rows × 16 lanes) at a time, so we
# evaluate the low and high halves separately and stitch them back
# together in the checks.
# ------------------------------------------------------------------

sub_b0_rows = run_binary_rows("sub", A_vpu_b0_rows, B_vpu_b0_rows)
sub_b1_rows = run_binary_rows("sub", A_vpu_b1_rows, B_vpu_b1_rows)

max_b0_rows = run_binary_rows("pairmax", A_vpu_b0_rows, B_vpu_b0_rows)
max_b1_rows = run_binary_rows("pairmax", A_vpu_b1_rows, B_vpu_b1_rows)

# ------------------------------------------------------------------
# Reference: Cross-unit VPU — F = vsub(C0, C1)
# ------------------------------------------------------------------

C0_b0_rows = float_matrix_to_bf16_rows(C0[:, :16])
C0_b1_rows = float_matrix_to_bf16_rows(C0[:, 16:])
C1_b0_rows = float_matrix_to_bf16_rows(C1[:, :16])
C1_b1_rows = float_matrix_to_bf16_rows(C1[:, 16:])

diff_b0_rows = run_binary_rows("sub", C0_b0_rows, C1_b0_rows)
diff_b1_rows = run_binary_rows("sub", C0_b1_rows, C1_b1_rows)

# ------------------------------------------------------------------
# Emit preloads
# ------------------------------------------------------------------

preloads = []

# Activations (2 slabs of 32×32 FP8)
emit_fp8_preload(preloads, X_D0_BASE, X_d0)
emit_fp8_preload(preloads, X_D1_BASE, X_d1)

# MXU0 weights (2 k-tiles)
emit_fp8_preload(preloads, W0_A_BASE, W0_A)
emit_fp8_preload(preloads, W0_B_BASE, W0_B)

# MXU1 weights (2 k-tiles)
emit_fp8_preload(preloads, W1_A_BASE, W1_A)
emit_fp8_preload(preloads, W1_B_BASE, W1_B)

# Zero bias
emit_fp8_preload(preloads, BIAS_BASE, zero_bias)

# VPU BF16 inputs
emit_bf16_rows_preload(preloads, A_VPU_B0_BASE, A_vpu_b0_rows)
emit_bf16_rows_preload(preloads, A_VPU_B1_BASE, A_vpu_b1_rows)
emit_bf16_rows_preload(preloads, B_VPU_B0_BASE, B_vpu_b0_rows)
emit_bf16_rows_preload(preloads, B_VPU_B1_BASE, B_vpu_b1_rows)

# ------------------------------------------------------------------
# Emit checks
# ------------------------------------------------------------------

checks = []

# MXU0 result (BF16 split)
emit_bf16_split_check(checks, C0_LO_BASE, C0_HI_BASE, C0)

# MXU1 result (BF16 split)
emit_bf16_split_check(checks, C1_LO_BASE, C1_HI_BASE, C1)

# VPU register-pair semantics: each BF16 instruction reads two
# consecutive register pairs and writes two consecutive registers.
#
# VSUB.BF16 2, 12, 14:
#   src A = (m12, m13) = (A_b0, A_b1)
#   src B = (m14, m15) = (B_b0, B_b1)
#   dst   = (m2, m3)  = full 32×32 vsub result
#
# VMAX.BF16 4, 12, 14:  (same source pairs — sources untouched by VSUB)
#   src A = (m12, m13)
#   src B = (m14, m15)
#   dst   = (m4, m5)  = full 32×32 vmax result

# VSUB: full tensor A - B
sub_b0_rows = run_binary_rows("sub", A_vpu_b0_rows, B_vpu_b0_rows)
sub_b1_rows = run_binary_rows("sub", A_vpu_b1_rows, B_vpu_b1_rows)

# VMAX: full tensor max(A, B)
max_b0_rows = run_binary_rows("pairmax", A_vpu_b0_rows, B_vpu_b0_rows)
max_b1_rows = run_binary_rows("pairmax", A_vpu_b1_rows, B_vpu_b1_rows)

emit_bf16_rows_check(checks, SUB_B0_BASE, SUB_B1_BASE, sub_b0_rows, sub_b1_rows)
emit_bf16_rows_check(checks, MAX_B0_BASE, MAX_B1_BASE, max_b0_rows, max_b1_rows)

# Cross-unit diff
emit_bf16_rows_check(checks, DIFF_B0_BASE, DIFF_B1_BASE, diff_b0_rows, diff_b1_rows)

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

sys.stderr.write(
    f"[gen_dual_mxu_pipeline_vpu] seed={SEED} K_FULL={K_FULL}\n"
    f"  C0 shape={C0.shape} C1 shape={C1.shape}\n"
    f"  preloads={len(preloads)} checks={len(checks)}\n"
)

emit_test_data(preloads, checks, timeout=800000)

#!/usr/bin/env python3
"""
gen_vmem_contention_stress.py

Generate test vectors for vmem_contention_stress.S.

Computations verified despite VMEM bank-arbitration backpressure:
    MXU0 (SA):  C0 = X0 @ W0   (32×32 BF16)
    MXU1 (IPT): C1 = X1 @ W1   (32×32 BF16)
    VPU:        D  = vadd(A, B) (32×32 BF16)
    MXU0 (SA):  C2 = X2 @ W2   (32×32 BF16, post-contention)

All MXU results use zero bias (FP8 0x00 = 0.0).

Usage:
    python3 gen_vmem_contention_stress.py [seed]
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
    float_to_bf16,
    pack_u16_le,
    run_binary_rows,
)
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
TILE = 32

# ── DRAM byte offsets (must match assembly) ──
X0_BASE       = 0x0000
W0_BASE       = 0x0400
X1_BASE       = 0x0800
W1_BASE       = 0x0C00
BIAS_BASE     = 0x1000
A_VPU_B0_BASE = 0x1400
A_VPU_B1_BASE = 0x1800
B_VPU_B0_BASE = 0x1C00
B_VPU_B1_BASE = 0x2000
X2_BASE       = 0x2400
W2_BASE       = 0x2800
# outputs
C0_LO_BASE    = 0x2C00
C0_HI_BASE    = 0x3000
C1_LO_BASE    = 0x3400
C1_HI_BASE    = 0x3800
ADD_B0_BASE   = 0x3C00
ADD_B1_BASE   = 0x4000
C2_LO_BASE    = 0x4400
C2_HI_BASE    = 0x4800


def beat_offset(byte_off):
    assert byte_off % 32 == 0
    return byte_off // 32


def emit_fp8_preload(preloads, byte_off, tile):
    preloads += preloads_from_words_packed(
        beat_offset(byte_off), matrix_to_fp8_words(tile)
    )


def emit_bf16_split_check(checks, byte_off_lo, byte_off_hi, tile32x32):
    """Check a 32×32 BF16 tile stored as two 32×16 banks (lo/hi lanes)."""
    left = tile32x32[:, :16]
    right = tile32x32[:, 16:]
    checks += checks_from_words_packed(
        beat_offset(byte_off_lo), matrix_to_bf16_words(left)
    )
    checks += checks_from_words_packed(
        beat_offset(byte_off_hi), matrix_to_bf16_words(right)
    )


def float_matrix_to_bf16_rows(mat):
    """Convert float matrix (R×C) to list of rows of BF16 bit values."""
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


# ── Functional models ──

sa_bf16 = SARTLLinearFunction(
    rows=32, cols=32,
    out_fmt_sel=OutputFmtSel.OutBF16,
)

ipt_bf16 = IPTLinearRTLFunction(
    vec_len=32, num_lanes=32, pipeline_depth=1,
    out_fmt_sel=OutputFmtSel.OutBF16,
)

# ── Generate inputs ──

X0 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 0)).astype(np.float32)
W0 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 1)).astype(np.float32)
X1 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 2)).astype(np.float32)
W1 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 3)).astype(np.float32)
X2 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 4)).astype(np.float32)
W2 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 5)).astype(np.float32)

A_vpu = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 6).astype(np.float32)
B_vpu = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 7).astype(np.float32)

zero_bias = np.zeros((TILE, TILE), dtype=np.float32)
b0_t = torch.zeros(TILE)

# VPU inputs as BF16 rows (per-bank: 32×16)
A_vpu_b0_rows = float_matrix_to_bf16_rows(A_vpu[:, :16])
A_vpu_b1_rows = float_matrix_to_bf16_rows(A_vpu[:, 16:])
B_vpu_b0_rows = float_matrix_to_bf16_rows(B_vpu[:, :16])
B_vpu_b1_rows = float_matrix_to_bf16_rows(B_vpu[:, 16:])

# ── Reference: C0 = X0 @ W0 (MXU0 SA) ──

C0 = sa_bf16(
    torch.from_numpy(X0),
    torch.from_numpy(W0.T.copy()),  # F.linear convention: y = x @ w^T
    b_q=b0_t, scale_exp=0,
).numpy().astype(np.float32)

# ── Reference: C1 = X1 @ W1 (MXU1 IPT) ──

C1 = ipt_bf16(
    torch.from_numpy(X1),
    torch.from_numpy(W1.T.copy()),
    scale_exp=0,
).numpy().astype(np.float32)

# ── Reference: D = vadd(A_vpu, B_vpu) (VPU) ──
#
# VPU VADD.BF16 operates on register pairs.  The assembly issues:
#   VADD.BF16 12, 8, 10
# which reads (m8,m9) = A pair, (m10,m11) = B pair, writes (m12,m13).
# Bank 0 (m8 vs m10) and bank 1 (m9 vs m11) processed together.

add_b0_rows = run_binary_rows("add", A_vpu_b0_rows, B_vpu_b0_rows)
add_b1_rows = run_binary_rows("add", A_vpu_b1_rows, B_vpu_b1_rows)

# ── Reference: C2 = X2 @ W2 (MXU0 SA, post-contention) ──

C2 = sa_bf16(
    torch.from_numpy(X2),
    torch.from_numpy(W2.T.copy()),
    b_q=b0_t, scale_exp=0,
).numpy().astype(np.float32)

# ── Emit preloads ──

preloads = []

emit_fp8_preload(preloads, X0_BASE, X0)
emit_fp8_preload(preloads, W0_BASE, W0)
emit_fp8_preload(preloads, X1_BASE, X1)
emit_fp8_preload(preloads, W1_BASE, W1)
emit_fp8_preload(preloads, BIAS_BASE, zero_bias)
emit_fp8_preload(preloads, X2_BASE, X2)
emit_fp8_preload(preloads, W2_BASE, W2)

emit_bf16_rows_preload(preloads, A_VPU_B0_BASE, A_vpu_b0_rows)
emit_bf16_rows_preload(preloads, A_VPU_B1_BASE, A_vpu_b1_rows)
emit_bf16_rows_preload(preloads, B_VPU_B0_BASE, B_vpu_b0_rows)
emit_bf16_rows_preload(preloads, B_VPU_B1_BASE, B_vpu_b1_rows)

# ── Emit checks ──

checks = []

# C0 (MXU0 result, stored via VMEM→DMA in Phase 7 under read-port backpressure)
emit_bf16_split_check(checks, C0_LO_BASE, C0_HI_BASE, C0)

# C1 (MXU1 result)
emit_bf16_split_check(checks, C1_LO_BASE, C1_HI_BASE, C1)

# VPU add (stored via VMEM→DMA after write-port contention phase)
emit_bf16_rows_check(checks, ADD_B0_BASE, ADD_B1_BASE, add_b0_rows, add_b1_rows)

# C2 (MXU0 second compute, data loaded during contention)
emit_bf16_split_check(checks, C2_LO_BASE, C2_HI_BASE, C2)

# ── Summary ──

sys.stderr.write(
    f"[gen_vmem_contention_stress] seed={SEED}\n"
    f"  C0 range: [{C0.min():.4f}, {C0.max():.4f}]\n"
    f"  C1 range: [{C1.min():.4f}, {C1.max():.4f}]\n"
    f"  C2 range: [{C2.min():.4f}, {C2.max():.4f}]\n"
    f"  preloads={len(preloads)} checks={len(checks)}\n"
)

emit_test_data(preloads, checks, timeout=1500000)

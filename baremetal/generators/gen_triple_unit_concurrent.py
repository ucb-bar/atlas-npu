#!/usr/bin/env python3
"""
gen_triple_unit_concurrent.py

Generate test vectors for triple_unit_concurrent.S which exercises
concurrent MXU0 + MXU1 + VPU execution.

Computations:
    MXU0 (SA):  C0 = X0 @ W0          (32×32 BF16)
    MXU1 (IPT): C1 = X1 @ W1          (32×32 BF16)
    VPU:        D  = vadd(A, B)        (32×32 BF16)
                E  = vmul(A, B)        (32×32 BF16)
                F  = vadd(C0, C1)      (cross-unit post-process)

Usage:
    python3 gen_triple_unit_concurrent.py [seed]
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

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 777

# DRAM byte offsets (must match assembly)
X0_BASE       = 0x0000
W0_BASE       = 0x0400
X1_BASE       = 0x0800
W1_BASE       = 0x0C00
BIAS_BASE     = 0x1000
A_VPU_B0_BASE = 0x1400
A_VPU_B1_BASE = 0x1800
B_VPU_B0_BASE = 0x1C00
B_VPU_B1_BASE = 0x2000
C0_LO_BASE    = 0x2400
C0_HI_BASE    = 0x2800
C1_LO_BASE    = 0x2C00
C1_HI_BASE    = 0x3000
ADD_B0_BASE   = 0x3400
ADD_B1_BASE   = 0x3800
MUL_B0_BASE   = 0x3C00
MUL_B1_BASE   = 0x4000
SUM_B0_BASE   = 0x4400
SUM_B1_BASE   = 0x4800

TILE = 32


def beat_offset(byte_off: int) -> int:
    assert byte_off % 32 == 0
    return byte_off // 32


def emit_fp8_preload(preloads, byte_off, tile):
    preloads += preloads_from_words_packed(
        beat_offset(byte_off), matrix_to_fp8_words(tile)
    )


def emit_bf16_split_check(checks, byte_off_lo, byte_off_hi, tile32x32):
    """Check a 32×32 BF16 tile stored as two 32×16 banks."""
    left = tile32x32[:, :16]
    right = tile32x32[:, 16:]
    checks += checks_from_words_packed(
        beat_offset(byte_off_lo), matrix_to_bf16_words(left)
    )
    checks += checks_from_words_packed(
        beat_offset(byte_off_hi), matrix_to_bf16_words(right)
    )


def emit_bf16_bank_preload(preloads, byte_off, tile32x16):
    """Preload one 32×16 BF16 bank."""
    preloads += preloads_from_words_packed(
        beat_offset(byte_off), matrix_to_bf16_words(tile32x16)
    )


def float_matrix_to_bf16_rows(mat: np.ndarray) -> list[list[int]]:
    """Convert a float matrix (R×C) into a list of R rows of BF16 bit values.
    C must equal BF16_PER_BEAT (16 lanes per bank)."""
    rows = []
    for r in range(mat.shape[0]):
        row = [float_to_bf16(float(mat[r, c])) for c in range(mat.shape[1])]
        rows.append(row)
    return rows


def bf16_rows_to_float_matrix(rows: list[list[int]]) -> np.ndarray:
    """Convert BF16 bit rows back to float matrix for downstream use."""
    import struct
    result = []
    for row in rows:
        frow = []
        for bits in row:
            # BF16 → float32: shift left 16, interpret as float32
            f32_bits = (bits & 0xFFFF) << 16
            frow.append(struct.unpack('f', struct.pack('I', f32_bits))[0])
        result.append(frow)
    return np.array(result, dtype=np.float32)


def emit_bf16_rows_check(checks, byte_off_b0, byte_off_b1,
                          rows_b0: list[list[int]], rows_b1: list[list[int]]):
    """Emit checks for VPU-style BF16 output stored as 2 banks of rows."""
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


def emit_bf16_rows_preload(preloads, byte_off, rows: list[list[int]]):
    """Emit preloads for VPU-style BF16 bank."""
    for r, row in enumerate(rows):
        preloads.append({
            "word_offset": beat_offset(byte_off) + r,
            "data": pack_u16_le(row),
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

np.random.seed(SEED)

# MXU inputs (FP8)
X0 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 0)).astype(np.float32)
W0 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 1)).astype(np.float32)
X1 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 2)).astype(np.float32)
W1 = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 3)).astype(np.float32)

# VPU inputs (BF16) — generate as float, convert to BF16 bits
A_vpu_float = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 4).astype(np.float32)
B_vpu_float = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 5).astype(np.float32)

# Convert VPU inputs to BF16 bit representations (per-bank: 32×16)
A_vpu_b0_rows = float_matrix_to_bf16_rows(A_vpu_float[:, :16])
A_vpu_b1_rows = float_matrix_to_bf16_rows(A_vpu_float[:, 16:])
B_vpu_b0_rows = float_matrix_to_bf16_rows(B_vpu_float[:, :16])
B_vpu_b1_rows = float_matrix_to_bf16_rows(B_vpu_float[:, 16:])

# Zero bias tile (all zeros FP8)
zero_bias = np.zeros((TILE, TILE), dtype=np.float32)

# ------------------------------------------------------------------
# Reference: MXU0 (SA) — C0 = X0 @ W0
# ------------------------------------------------------------------

X0_t = torch.from_numpy(X0)
W0_t = torch.from_numpy(W0.T.copy())  # F.linear: y = x @ w^T → pass W0^T
b0_t = torch.zeros(TILE)

C0 = sa_bf16(X0_t, W0_t, b_q=b0_t, scale_exp=0).numpy().astype(np.float32)

# ------------------------------------------------------------------
# Reference: MXU1 (IPT) — C1 = X1 @ W1
# ------------------------------------------------------------------

X1_t = torch.from_numpy(X1)
W1_t = torch.from_numpy(W1.T.copy())  # F.linear convention

C1 = ipt_bf16(X1_t, W1_t, scale_exp=0).numpy().astype(np.float32)

# ------------------------------------------------------------------
# Reference: VPU ops — vadd(A, B) and vmul(A, B)
# Each bank processed independently through VPU model.
# ------------------------------------------------------------------

add_b0_rows = run_binary_rows("add", A_vpu_b0_rows, B_vpu_b0_rows)
add_b1_rows = run_binary_rows("add", A_vpu_b1_rows, B_vpu_b1_rows)

mul_b0_rows = run_binary_rows("mul", A_vpu_b0_rows, B_vpu_b0_rows)
mul_b1_rows = run_binary_rows("mul", A_vpu_b1_rows, B_vpu_b1_rows)

# ------------------------------------------------------------------
# Reference: Cross-unit VPU post-process — F = vadd(C0, C1)
#
# C0 and C1 are BF16 floats from MXU pops. Convert to BF16 bit rows
# per bank, then run through VPU add model.
# ------------------------------------------------------------------

C0_b0_rows = float_matrix_to_bf16_rows(C0[:, :16])
C0_b1_rows = float_matrix_to_bf16_rows(C0[:, 16:])
C1_b0_rows = float_matrix_to_bf16_rows(C1[:, :16])
C1_b1_rows = float_matrix_to_bf16_rows(C1[:, 16:])

sum_b0_rows = run_binary_rows("add", C0_b0_rows, C1_b0_rows)
sum_b1_rows = run_binary_rows("add", C0_b1_rows, C1_b1_rows)

# ------------------------------------------------------------------
# Emit preloads and checks
# ------------------------------------------------------------------

preloads = []
checks = []

# FP8 inputs
emit_fp8_preload(preloads, X0_BASE, X0)
emit_fp8_preload(preloads, W0_BASE, W0)
emit_fp8_preload(preloads, X1_BASE, X1)
emit_fp8_preload(preloads, W1_BASE, W1)

# Zero bias (all zeros — FP8 0x00 = 0.0)
emit_fp8_preload(preloads, BIAS_BASE, zero_bias)

# VPU BF16 inputs (2 banks each)
emit_bf16_rows_preload(preloads, A_VPU_B0_BASE, A_vpu_b0_rows)
emit_bf16_rows_preload(preloads, A_VPU_B1_BASE, A_vpu_b1_rows)
emit_bf16_rows_preload(preloads, B_VPU_B0_BASE, B_vpu_b0_rows)
emit_bf16_rows_preload(preloads, B_VPU_B1_BASE, B_vpu_b1_rows)

# MXU0 result check (BF16 split into lo/hi banks)
emit_bf16_split_check(checks, C0_LO_BASE, C0_HI_BASE, C0)

# MXU1 result check
emit_bf16_split_check(checks, C1_LO_BASE, C1_HI_BASE, C1)

# VPU add result check
emit_bf16_rows_check(checks, ADD_B0_BASE, ADD_B1_BASE, add_b0_rows, add_b1_rows)

# VPU mul result check
emit_bf16_rows_check(checks, MUL_B0_BASE, MUL_B1_BASE, mul_b0_rows, mul_b1_rows)

# Cross-unit sum result check
emit_bf16_rows_check(checks, SUM_B0_BASE, SUM_B1_BASE, sum_b0_rows, sum_b1_rows)

emit_test_data(preloads, checks, timeout=500000)

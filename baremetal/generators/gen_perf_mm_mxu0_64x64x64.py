#!/usr/bin/env python3
"""
gen_perf_mm_mxu0_64x64x64.py

Preload + golden generator for perf_mm_mxu0_64x64x64.S — a single-MXU
(M=64, K=64, N=64) tiled FP8 matmul performance test that uses only
MXU0. Pass/fail is determined by comparing the DRAM output against the
golden SA model; utilization is reported (including a cycle-bracketed
`util_timed` from a scalar CSRR snapshot) but not gated.

The 64x64 × 64x64 matmul is tiled into 32x32 blocks:
    C00 = A00·B00 + A01·B10
    C01 = A00·B01 + A01·B11
    C10 = A10·B00 + A11·B10
    C11 = A10·B01 + A11·B11

Weights are pre-transposed in the generator (B_ij_T = B_ij.T) and
preloaded directly into the layout VMATPUSH.W expects, so the assembly
skips the XLU transpose step entirely.

DRAM layout (must match perf_mm_mxu0_64x64x64.S):
    0x0000  A00      (1024B FP8, 32×32)
    0x0400  A01
    0x0800  A10
    0x0C00  A11
    0x1000  B00_T
    0x1400  B01_T
    0x1800  B10_T
    0x1C00  B11_T
    0x2000  C00      (1024B FP8, 32×32 — output, checked)
    0x2400  C01
    0x2800  C10
    0x2C00  C11

Golden model (mirrors gen_mxu0_qk_scores.py — a known-passing test):
    1. SA model in OutBF16 mode: hardware accumulates in BF16 between
       K-tiles (VMATMUL → accbuf → VMATMUL.ACC → accbuf), and the C
       model's internal K-tiling uses SA_UsePsum with a BF16 bit
       pattern. OutE4M3 would feed FP8 bytes back as psum, which get
       reinterpreted as BF16 garbage — correct multi-K-tile math
       REQUIRES OutBF16.
    2. bf16_to_e4m3_scaled_bits applied element-wise to model
       VMATPOP.FP8 — this is the exact per-element BF16ScaleToE4M3
       conversion the RTL uses (distinct from quantize_fp8's generic
       FP32 → FP8 rounding). SCALE_E8M0 = 127 matches SELI 0, 0x7F.

Usage:
    python3 gen_perf_mm_mxu0_64x64x64.py [seed]
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
    preloads_from_words_packed,
    checks_from_words_packed,
    emit_test_data,
)
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TILE = 32
TIMEOUT = 400000
SCALE_E8M0 = 127  # E8M0=127 → scale_exp=0 → unit scale (matches SELI 0, 0x7F)

# DRAM beat offsets (1 beat = 32 B). Each 32×32 FP8 tile is 1024 B = 32 beats.
A_BASES = [0x0000 // 32, 0x0400 // 32, 0x0800 // 32, 0x0C00 // 32]   # A00 A01 A10 A11
B_T_BASES = [0x1000 // 32, 0x1400 // 32, 0x1800 // 32, 0x1C00 // 32] # B00_T B01_T B10_T B11_T
C_BASES = [0x2000 // 32, 0x2400 // 32, 0x2800 // 32, 0x2C00 // 32]   # C00 C01 C10 C11


def round_right_shift_4_rne(x: int) -> int:
    """Match hardware roundRightShift4RNE."""
    trunc = (x >> 4) & 0xF
    guard = (x >> 3) & 0x1
    sticky = 1 if (x & 0x7) != 0 else 0
    lsb = trunc & 0x1
    inc = 1 if (guard and (sticky or lsb)) else 0
    return trunc + inc


def bf16_to_e4m3_scaled_bits(bf16_bits: int, scale_e8m0: int) -> int:
    """Exact software mirror of BF16ScaleToE4M3 RTL. Returns raw 8-bit E4M3 bits."""
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


def tile_fp8(seed_offset):
    return quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + seed_offset)).astype(np.float32)


def main():
    sa_bf16 = SARTLLinearFunction(
        rows=TILE, cols=TILE, out_fmt_sel=OutputFmtSel.OutBF16
    )

    # Four activation tiles (A_ik, i=row, k=inner) and four weight tiles (B_kj).
    A = [[tile_fp8(10 * i + k) for k in range(2)] for i in range(2)]
    B = [[tile_fp8(20 + 10 * k + j) for j in range(2)] for k in range(2)]

    # Pre-transposed weights (what DRAM holds; what VMATPUSH.W consumes).
    B_T = [[B[k][j].T.copy() for j in range(2)] for k in range(2)]

    # Golden C_ij = sum_k (A_ik · B_kj). Concatenate along K so the SA
    # model's internal K-tiling drives accumulation via SA_UsePsum —
    # mirroring hardware exactly: K=0 writes a BF16 acc-buffer value,
    # K=1 reads it back as psum and accumulates at PE internal precision.
    # Then apply bf16_to_e4m3_scaled_bits element-wise to mirror VMATPOP.FP8.
    C = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            a_full = np.concatenate([A[i][0], A[i][1]], axis=1)      # [32, 64]
            w_full = np.concatenate([B_T[0][j], B_T[1][j]], axis=1)  # [32, 64]
            a_t = torch.from_numpy(a_full)
            w_t = torch.from_numpy(w_full.copy())
            bf16_tile = sa_bf16(a_t, w_t, scale_exp=0).numpy().astype(np.float32)
            C[i][j] = bf16_tile_to_scaled_fp8(bf16_tile, SCALE_E8M0)

    # Preloads: activations, then pre-transposed weights.
    preloads = []
    A_flat = [A[0][0], A[0][1], A[1][0], A[1][1]]
    for base, tile in zip(A_BASES, A_flat):
        preloads += preloads_from_words_packed(base, matrix_to_fp8_words(tile))
    B_T_flat = [B_T[0][0], B_T[0][1], B_T[1][0], B_T[1][1]]
    for base, tile in zip(B_T_BASES, B_T_flat):
        preloads += preloads_from_words_packed(base, matrix_to_fp8_words(tile))

    # Checks: the four 32×32 FP8 output tiles written back by the assembly.
    checks = []
    C_flat = [C[0][0], C[0][1], C[1][0], C[1][1]]
    for base, tile in zip(C_BASES, C_flat):
        checks += checks_from_words_packed(base, matrix_to_fp8_words(tile))

    sys.stderr.write(
        f"[gen_perf_mm_mxu0_64x64x64] seed={SEED} preloads={len(preloads)} checks={len(checks)}\n"
    )

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_perf_mm_mxu0_64x64x128.py

Preload + golden generator for perf_mm_mxu0_64x64x128.S — a single-MXU
(M=64, N=64, K=128) tiled FP8 matmul performance test that uses only
MXU0. Pass/fail is determined by comparing the DRAM output against the
golden SA model; utilization is reported (including a cycle-bracketed
`util_timed` from a scalar CSRR snapshot) but not gated.

Compared to the K=64 sibling, K is doubled (4 K-tiles instead of 2 per
output tile), so the systolic array runs 16 matmuls + 4 pops instead of
8 + 4. Utilization rises because pop overhead is amortized over twice
the MAC work.

Tiling (M=64, N=64, K=128 → 32x32 blocks):
    A is 2x4 grid:   A00 A01 A02 A03  (M=0..31)
                     A10 A11 A12 A13  (M=32..63)
    B is 4x2 grid:   B[k][j] for k in 0..3, j in 0..1
    C is 2x2 grid:   C[i][j] = sum_k A[i][k] · B[k][j]

Weights are pre-transposed in the generator (B_kj_T = B_kj.T) and
preloaded directly into the layout VMATPUSH.W expects, so the assembly
skips the XLU transpose step.

DRAM layout (must match perf_mm_mxu0_64x64x128.S):
    0x0000 A00   0x0400 A01   0x0800 A02   0x0C00 A03
    0x1000 A10   0x1400 A11   0x1800 A12   0x1C00 A13
    0x2000 B00_T 0x2400 B01_T
    0x2800 B10_T 0x2C00 B11_T
    0x3000 B20_T 0x3400 B21_T
    0x3800 B30_T 0x3C00 B31_T
    0x4000 C00   0x4400 C01   0x4800 C10   0x4C00 C11   (output, checked)

Golden model — IMPORTANT: this is an MXU0 test, so the golden uses the
SA model (SARTLLinearFunction). Using the IPT model would produce
small (±1 byte) DRAM mismatches due to differing rounding/anchor-exp
behavior.

    1. SA model in OutBF16 mode: hardware accumulates in BF16 between
       K-tiles (VMATMUL → accbuf → VMATMUL.ACC → accbuf), and the C
       model's internal K-tiling uses SA_UsePsum with a BF16 bit
       pattern.
    2. bf16_to_e4m3_scaled_bits applied element-wise to model
       VMATPOP.FP8 — the exact per-element BF16ScaleToE4M3 the RTL
       uses. SCALE_E8M0 = 127 matches SELI 0, 0x7F.

Usage:
    python3 gen_perf_mm_mxu0_64x64x128.py [seed]
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
M_TILES = 2     # M = 64 = 2 * 32
N_TILES = 2     # N = 64 = 2 * 32
K_TILES = 4     # K = 128 = 4 * 32
TIMEOUT = 600000
SCALE_E8M0 = 127  # E8M0=127 → scale_exp=0 → unit scale (matches SELI 0, 0x7F)

# DRAM beat offsets (1 beat = 32 B). Each 32×32 FP8 tile is 1024 B = 32 beats.
A_BASES = [
    0x0000 // 32, 0x0400 // 32, 0x0800 // 32, 0x0C00 // 32,  # A00 A01 A02 A03
    0x1000 // 32, 0x1400 // 32, 0x1800 // 32, 0x1C00 // 32,  # A10 A11 A12 A13
]
B_T_BASES = [
    0x2000 // 32, 0x2400 // 32,  # B00_T B01_T (k=0)
    0x2800 // 32, 0x2C00 // 32,  # B10_T B11_T (k=1)
    0x3000 // 32, 0x3400 // 32,  # B20_T B21_T (k=2)
    0x3800 // 32, 0x3C00 // 32,  # B30_T B31_T (k=3)
]
C_BASES = [
    0x4000 // 32, 0x4400 // 32,  # C00 C01
    0x4800 // 32, 0x4C00 // 32,  # C10 C11
]


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

    # A[i][k]: i in 0..M_TILES-1 (row), k in 0..K_TILES-1 (inner).
    # B[k][j]: k in 0..K_TILES-1 (inner), j in 0..N_TILES-1 (col).
    A = [[tile_fp8(10 * i + k) for k in range(K_TILES)] for i in range(M_TILES)]
    B = [[tile_fp8(50 + 10 * k + j) for j in range(N_TILES)] for k in range(K_TILES)]

    # Pre-transposed weights (what DRAM holds; what VMATPUSH.W consumes).
    B_T = [[B[k][j].T.copy() for j in range(N_TILES)] for k in range(K_TILES)]

    # Golden C[i][j] = sum_k (A[i][k] · B[k][j]). Concatenate along K so the
    # SA model's internal K-tiling drives accumulation via SA_UsePsum —
    # mirroring hardware exactly: each K=k>0 step reads the BF16 acc-buffer
    # back as psum and accumulates at PE internal precision. Then apply
    # bf16_to_e4m3_scaled_bits element-wise to mirror VMATPOP.FP8.
    C = [[None for _ in range(N_TILES)] for _ in range(M_TILES)]
    for i in range(M_TILES):
        for j in range(N_TILES):
            a_full = np.concatenate([A[i][k] for k in range(K_TILES)], axis=1)         # [32, 128]
            w_full = np.concatenate([B_T[k][j] for k in range(K_TILES)], axis=1)       # [32, 128]
            a_t = torch.from_numpy(a_full)
            w_t = torch.from_numpy(w_full.copy())
            bf16_tile = sa_bf16(a_t, w_t, scale_exp=0).numpy().astype(np.float32)
            C[i][j] = bf16_tile_to_scaled_fp8(bf16_tile, SCALE_E8M0)

    # Preloads: activations (row-major: A00..A03, A10..A13), then pre-transposed
    # weights (k-major: B[0][0], B[0][1], B[1][0], B[1][1], ...).
    preloads = []
    A_flat = [A[i][k] for i in range(M_TILES) for k in range(K_TILES)]
    for base, tile in zip(A_BASES, A_flat):
        preloads += preloads_from_words_packed(base, matrix_to_fp8_words(tile))
    B_T_flat = [B_T[k][j] for k in range(K_TILES) for j in range(N_TILES)]
    for base, tile in zip(B_T_BASES, B_T_flat):
        preloads += preloads_from_words_packed(base, matrix_to_fp8_words(tile))

    # Checks: the four 32×32 FP8 output tiles written back by the assembly.
    checks = []
    C_flat = [C[i][j] for i in range(M_TILES) for j in range(N_TILES)]
    for base, tile in zip(C_BASES, C_flat):
        checks += checks_from_words_packed(base, matrix_to_fp8_words(tile))

    sys.stderr.write(
        f"[gen_perf_mm_mxu0_64x64x128] seed={SEED} preloads={len(preloads)} checks={len(checks)}\n"
    )

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

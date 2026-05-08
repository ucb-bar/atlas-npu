#!/usr/bin/env python3
"""
gen_perf_mm_dual_128x128x128.py

Preload + golden generator for perf_mm_dual_128x128x128.S — a dual-MXU
(M=128, N=128, K=128) tiled FP8 matmul performance test that runs MXU0
(SA) and MXU1 (IPT) concurrently. Pass/fail compares DRAM output
against per-MXU goldens; utilization is reported (no threshold).

Ownership: ROW-PINNED.
    MXU0 owns the top half       (i in {0,1}, 8 output tiles)
    MXU1 owns the bottom half    (i in {2,3}, 8 output tiles)

Two sub-phases rotate the B-column-pair each MXU works on, so that A
and B reads are always on disjoint physical banks across the two MXUs:
    sub-alpha: MXU0 builds C[0..1, 0..1] (TL); MXU1 builds C[2..3, 2..3] (BR)
    sub-beta : MXU0 builds C[0..1, 2..3] (TR); MXU1 builds C[2..3, 0..1] (BL)

Both MXUs traverse K in the natural [0,1,2,3] order — no K-skew is
needed because the row-pinned split keeps each MXU on its own A rows
(banks 0..7 vs 8..15) at all times.

Tiling (M=128, N=128, K=128 -> 32x32 blocks -> 4x4 tile grid):
    A is 4x4 grid:  A[i][k] for i,k in 0..3
    B is 4x4 grid:  B[k][j] for k,j in 0..3
    C is 4x4 grid:  C[i][j] = sum_k A[i][k] * B[k][j]

Weights are pre-transposed in the generator (B_kj_T = B_kj.T) and
preloaded directly into the layout VMATPUSH.W expects.

DRAM layout (must match perf_mm_dual_128x128x128.S):
    A tiles   @ 0x0000 + (4*i + k) * 0x400, i,k in 0..3  ->  0x0000..0x3C00
    B^T tiles @ 0x4000 + (4*k + j) * 0x400, k,j in 0..3  ->  0x4000..0x7C00
    C tiles   @ 0x8000 + (4*i + j) * 0x400, i,j in 0..3  ->  0x8000..0xBC00

Golden model — BF16 K-accumulation is order-sensitive, but with vanilla
K-order both halves use the standard models:
    MXU0-owned tiles (i in {0,1}) -> SA  model, K-order [0,1,2,3]
    MXU1-owned tiles (i in {2,3}) -> IPT model, K-order [0,1,2,3]
Both models in OutBF16 mode + bf16_to_e4m3_scaled_bits to mirror
VMATPOP.FP8 quantization at SCALE_E8M0 = 127 (SELI 0, 0x7F).

Usage:
    python3 gen_perf_mm_dual_128x128x128.py [seed]
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
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TILE = 32
N_TILES = 4     # 128 / 32 = 4 along each of M, N, K
TIMEOUT = 800000
SCALE_E8M0 = 127  # E8M0=127 -> scale_exp=0 -> unit scale (matches SELI 0, 0x7F)

# DRAM beat offsets (1 beat = 32 B). Each 32x32 FP8 tile is 1024 B = 32 beats.
A_BASES   = [(0x0000 + (4 * i + k) * 0x400) // 32 for i in range(N_TILES) for k in range(N_TILES)]
B_T_BASES = [(0x4000 + (4 * k + j) * 0x400) // 32 for k in range(N_TILES) for j in range(N_TILES)]
C_BASES   = [(0x8000 + (4 * i + j) * 0x400) // 32 for i in range(N_TILES) for j in range(N_TILES)]


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
    sa_bf16  = SARTLLinearFunction(rows=TILE, cols=TILE, out_fmt_sel=OutputFmtSel.OutBF16)
    ipt_bf16 = IPTLinearRTLFunction(vec_len=TILE, num_lanes=TILE, out_fmt_sel=OutputFmtSel.OutBF16)

    # A[i][k]: i in 0..3 (row), k in 0..3 (inner).
    # B[k][j]: k in 0..3 (inner), j in 0..3 (col).
    A = [[tile_fp8(10 * i + k)        for k in range(N_TILES)] for i in range(N_TILES)]
    B = [[tile_fp8(200 + 10 * k + j)  for j in range(N_TILES)] for k in range(N_TILES)]

    # Pre-transposed weights (what DRAM holds; what VMATPUSH.W consumes).
    B_T = [[B[k][j].T.copy() for j in range(N_TILES)] for k in range(N_TILES)]

    # Golden C[i][j] = sum_k A[i][k] * B[k][j].
    # Ownership by M-row of the 4x4 C grid:
    #   MXU0 owns rows i in {0,1} -> SA  model
    #   MXU1 owns rows i in {2,3} -> IPT model
    # Both halves use vanilla K-order [0,1,2,3] (row-pinned avoids K-skew).
    C = [[None for _ in range(N_TILES)] for _ in range(N_TILES)]
    for i in range(N_TILES):
        for j in range(N_TILES):
            mxu0_owns = (i // 2) == 0
            model = sa_bf16 if mxu0_owns else ipt_bf16
            a_full = np.concatenate([A[i][k]   for k in range(N_TILES)], axis=1)  # [32, 128]
            w_full = np.concatenate([B_T[k][j] for k in range(N_TILES)], axis=1)  # [32, 128]
            a_t = torch.from_numpy(a_full)
            w_t = torch.from_numpy(w_full.copy())
            bf16_tile = model(a_t, w_t, scale_exp=0).numpy().astype(np.float32)
            C[i][j] = bf16_tile_to_scaled_fp8(bf16_tile, SCALE_E8M0)

    # Preloads: 16 A tiles (row-major: A00..A03, A10..A13, A20..A23, A30..A33)
    # then 16 B^T tiles (k-major: B[0][0..3], B[1][0..3], B[2][0..3], B[3][0..3]).
    preloads = []
    A_flat = [A[i][k] for i in range(N_TILES) for k in range(N_TILES)]
    for base, tile in zip(A_BASES, A_flat):
        preloads += preloads_from_words_packed(base, matrix_to_fp8_words(tile))
    B_T_flat = [B_T[k][j] for k in range(N_TILES) for j in range(N_TILES)]
    for base, tile in zip(B_T_BASES, B_T_flat):
        preloads += preloads_from_words_packed(base, matrix_to_fp8_words(tile))

    # Checks: 16 32x32 FP8 output tiles written back by the assembly.
    checks = []
    C_flat = [C[i][j] for i in range(N_TILES) for j in range(N_TILES)]
    for base, tile in zip(C_BASES, C_flat):
        checks += checks_from_words_packed(base, matrix_to_fp8_words(tile))

    sys.stderr.write(
        f"[gen_perf_mm_dual_128x128x128] seed={SEED} preloads={len(preloads)} checks={len(checks)}\n"
    )

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

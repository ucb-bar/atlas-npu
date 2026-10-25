#!/usr/bin/env python3
"""
gen_perf_mm_mxu1_64x64x128.py

Preload + golden generator for perf_mm_mxu1_64x64x128.S — a single-MXU
(M=64, N=64, K=128) tiled FP8 matmul performance test that uses only
MXU1. Pass/fail is determined by comparing the DRAM output against the
golden IPT model; utilization is reported (including a cycle-bracketed
`util_timed` from a scalar CSRR snapshot) but not gated.

The 64×128 × 128×64 = 64×64 matmul is tiled into 32×32 blocks. With
M=64 → 2 i-rows, N=64 → 2 j-cols, K=128 → 4 k-tiles, the per-output
sum is over 4 K-tiles:
    C[i][j] = sum_{k=0..3} A[i][k] · B[k][j]
yielding 4 output tiles (C00, C01, C10, C11), each accumulated across
4 K-tiles in the MXU1 FP32 accumulator (VMATMUL → acc, VMATMUL.ACC ×3).

Weights are pre-transposed in the generator (B_kj_T = B_kj.T) and
preloaded directly into the layout VMATPUSH.W expects, so the assembly
skips the XLU transpose step entirely.

DRAM layout (must match perf_mm_mxu1_64x64x128.S):
    A (i-major then k):                B_T (k-major then j):       C:
      0x0000  A00  (i=0,k=0)             0x2000  B00_T (k=0,j=0)    0x4000  C00
      0x0400  A01  (i=0,k=1)             0x2400  B01_T (k=0,j=1)    0x4400  C01
      0x0800  A02  (i=0,k=2)             0x2800  B10_T (k=1,j=0)    0x4800  C10
      0x0C00  A03  (i=0,k=3)             0x2C00  B11_T (k=1,j=1)    0x4C00  C11
      0x1000  A10  (i=1,k=0)             0x3000  B20_T (k=2,j=0)
      0x1400  A11  (i=1,k=1)             0x3400  B21_T (k=2,j=1)
      0x1800  A12  (i=1,k=2)             0x3800  B30_T (k=3,j=0)
      0x1C00  A13  (i=1,k=3)             0x3C00  B31_T (k=3,j=1)

Golden model (as gen_perf_mm_mxu1_64x64x64.py): IPTLinearRTLFunction in OutE4M3 mode
chains the 4 K-tiles through its FP32 accumulator and pops each tile once with
VMATPOP.FP8's E4M3_RN_SATFINITE. SCALE_E8M0 = 127 matches SELI 0, 0x7F.

Usage:
    python3 gen_perf_mm_mxu1_64x64x128.py [seed]
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
    preloads_from_words_packed,
    checks_from_words_packed,
    emit_test_data,
)
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction, e4m3_words
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TILE = 32
M_TILES = 2          # M = 64 = 2 × 32
N_TILES = 2          # N = 64 = 2 × 32
K_TILES = 4          # K = 128 = 4 × 32
TIMEOUT = 400000
SCALE_E8M0 = 127     # E8M0=127 → scale_exp=0 → unit scale (matches SELI 0, 0x7F)

# DRAM beat offsets (1 beat = 32 B). Each 32×32 FP8 tile is 1024 B = 32 beats.
# A: i-major then k.    B_T: k-major then j.    C: i-major then j.
A_BASES   = [(0x0000 + (i * K_TILES + k) * 0x400) // 32
             for i in range(M_TILES) for k in range(K_TILES)]
B_T_BASES = [(0x2000 + (k * N_TILES + j) * 0x400) // 32
             for k in range(K_TILES) for j in range(N_TILES)]
C_BASES   = [(0x4000 + (i * N_TILES + j) * 0x400) // 32
             for i in range(M_TILES) for j in range(N_TILES)]


def tile_fp8(seed_offset):
    return quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + seed_offset)).astype(np.float32)


def main():
    ipt_fp8 = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutE4M3)

    # 8 activation tiles A[i][k] (i∈{0,1}, k∈{0,1,2,3}) and 8 weight tiles B[k][j].
    # Seed offset for B bumped to 100 to keep distinct from K=64 test's seed space.
    A = [[tile_fp8(10 * i + k) for k in range(K_TILES)] for i in range(M_TILES)]
    B = [[tile_fp8(100 + 10 * k + j) for j in range(N_TILES)] for k in range(K_TILES)]

    # Pre-transposed weights (what DRAM holds; what VMATPUSH.W consumes).
    B_T = [[B[k][j].T.copy() for j in range(N_TILES)] for k in range(K_TILES)]

    # Golden C[i][j] = FP8 pop of sum_k (A[i][k] · B[k][j]). Concatenate along K so the
    # model chains the 4 K-tiles through its FP32 accumulator, as VMATMUL.ACC does.
    C = [[None] * N_TILES for _ in range(M_TILES)]
    for i in range(M_TILES):
        for j in range(N_TILES):
            a_full = np.concatenate([A[i][k]   for k in range(K_TILES)], axis=1)  # [32, 128]
            w_full = np.concatenate([B_T[k][j] for k in range(K_TILES)], axis=1)  # [32, 128]
            a_t = torch.from_numpy(a_full)
            w_t = torch.from_numpy(w_full.copy())
            C[i][j] = ipt_fp8(a_t, w_t, scale_exp=SCALE_E8M0 - 127)

    # Preloads: activations (i-major, k-minor), then pre-transposed weights (k-major, j-minor).
    preloads = []
    for i in range(M_TILES):
        for k in range(K_TILES):
            base = A_BASES[i * K_TILES + k]
            preloads += preloads_from_words_packed(base, matrix_to_fp8_words(A[i][k]))
    for k in range(K_TILES):
        for j in range(N_TILES):
            base = B_T_BASES[k * N_TILES + j]
            preloads += preloads_from_words_packed(base, matrix_to_fp8_words(B_T[k][j]))

    # Checks: the four 32×32 FP8 output tiles written back by the assembly.
    checks = []
    for i in range(M_TILES):
        for j in range(N_TILES):
            base = C_BASES[i * N_TILES + j]
            checks += checks_from_words_packed(base, e4m3_words(C[i][j]))

    sys.stderr.write(
        f"[gen_perf_mm_mxu1_64x64x128] seed={SEED} preloads={len(preloads)} checks={len(checks)}\n"
    )

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

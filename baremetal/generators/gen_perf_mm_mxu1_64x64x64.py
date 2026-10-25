#!/usr/bin/env python3
"""
gen_perf_mm_mxu1_64x64x64.py

Preload + golden generator for perf_mm_mxu1_64x64x64.S — a single-MXU
(M=64, N=64, K=64) tiled FP8 matmul performance test that uses only
MXU1. Pass/fail is determined by comparing the DRAM output against the
golden IPT model; utilization is reported (including a cycle-bracketed
`util_timed` from a scalar CSRR snapshot) but not gated.

The 64x64 × 64x64 matmul is tiled into 32x32 blocks:
    C00 = A00·B00 + A01·B10
    C01 = A00·B01 + A01·B11
    C10 = A10·B00 + A11·B10
    C11 = A10·B01 + A11·B11

Weights are pre-transposed in the generator (B_ij_T = B_ij.T) and
preloaded directly into the layout VMATPUSH.W expects, so the assembly
skips the XLU transpose step entirely.

DRAM layout (must match perf_mm_mxu1_64x64x64.S):
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

Golden model: IPTLinearRTLFunction in OutE4M3 mode. It accumulates both K-tiles in
FP32 (VMATMUL → accbuf → VMATMUL.ACC → accbuf) and pops each tile once with
VMATPOP.FP8's E4M3_RN_SATFINITE. SCALE_E8M0 = 127 matches SELI 0, 0x7F.

Usage:
    python3 gen_perf_mm_mxu1_64x64x64.py [seed]
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
TIMEOUT = 400000
SCALE_E8M0 = 127  # E8M0=127 → scale_exp=0 → unit scale (matches SELI 0, 0x7F)

# DRAM beat offsets (1 beat = 32 B). Each 32×32 FP8 tile is 1024 B = 32 beats.
A_BASES = [0x0000 // 32, 0x0400 // 32, 0x0800 // 32, 0x0C00 // 32]   # A00 A01 A10 A11
B_T_BASES = [0x1000 // 32, 0x1400 // 32, 0x1800 // 32, 0x1C00 // 32] # B00_T B01_T B10_T B11_T
C_BASES = [0x2000 // 32, 0x2400 // 32, 0x2800 // 32, 0x2C00 // 32]   # C00 C01 C10 C11


def tile_fp8(seed_offset):
    return quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + seed_offset)).astype(np.float32)


def main():
    ipt_fp8 = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutE4M3)

    # Four activation tiles (A_ik, i=row, k=inner) and four weight tiles (B_kj).
    A = [[tile_fp8(10 * i + k) for k in range(2)] for i in range(2)]
    B = [[tile_fp8(20 + 10 * k + j) for j in range(2)] for k in range(2)]

    # Pre-transposed weights (what DRAM holds; what VMATPUSH.W consumes).
    B_T = [[B[k][j].T.copy() for j in range(2)] for k in range(2)]

    # Golden C_ij = FP8 pop of sum_k (A_ik · B_kj). Concatenate along K so the model
    # chains the K-tiles through its FP32 accumulator, as VMATMUL.ACC does.
    C = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            a_full = np.concatenate([A[i][0], A[i][1]], axis=1)      # [32, 64]
            w_full = np.concatenate([B_T[0][j], B_T[1][j]], axis=1)  # [32, 64]
            a_t = torch.from_numpy(a_full)
            w_t = torch.from_numpy(w_full.copy())
            C[i][j] = ipt_fp8(a_t, w_t, scale_exp=SCALE_E8M0 - 127)

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
        checks += checks_from_words_packed(base, e4m3_words(tile))

    sys.stderr.write(
        f"[gen_perf_mm_mxu1_64x64x64] seed={SEED} preloads={len(preloads)} checks={len(checks)}\n"
    )

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

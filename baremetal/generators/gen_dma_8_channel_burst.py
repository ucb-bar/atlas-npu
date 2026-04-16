#!/usr/bin/env python3
"""
gen_dma_8chan_burst.py

Generate test vectors for dma_8chan_burst.S — 8-channel DMA burst test.

8 tiles loaded via 8 simultaneous DMA channels, then:
    MXU0 (SA):  C0 = X @ W + bias  (32×32 BF16)
    VPU:        D  = vadd(A, B)    (32×32 BF16)

4 tiles echo-tested through LOAD→STORE, 4 tiles are compute results.

Usage:
    python3 gen_dma_8chan_burst.py [seed]
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
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
TILE = 32

# ── DRAM byte offsets ──
X_BASE         = 0x0000
W_BASE         = 0x0400
BIAS_BASE      = 0x0800
A_VPU_B0_BASE  = 0x0C00
A_VPU_B1_BASE  = 0x1000
B_VPU_B0_BASE  = 0x1400
B_VPU_B1_BASE  = 0x1800
ECHO_TILE_BASE = 0x1C00

C0_LO_BASE     = 0x2000
C0_HI_BASE     = 0x2400
ADD_B0_BASE    = 0x2800
ADD_B1_BASE    = 0x2C00
ECHO0_BASE     = 0x3000   # echo of A_vpu b1
ECHO1_BASE     = 0x3400   # echo of B_vpu b0
ECHO2_BASE     = 0x3800   # echo of B_vpu b1
ECHO3_BASE     = 0x3C00   # echo of echo_tile


def beat_offset(byte_off):
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


def float_matrix_to_bf16_rows(mat):
    return [
        [float_to_bf16(float(mat[r, c])) for c in range(mat.shape[1])]
        for r in range(mat.shape[0])
    ]


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


def emit_echo_checks(checks, src_preloads, src_byte_off, dst_byte_off, num_beats=32):
    """Emit checks that verify DMA echo: dst data matches src preload data."""
    src_base = beat_offset(src_byte_off)
    dst_base = beat_offset(dst_byte_off)
    # Find preloads for the source range and replicate as checks at dest
    src_map = {}
    for p in src_preloads:
        if src_base <= p["word_offset"] < src_base + num_beats:
            src_map[p["word_offset"] - src_base] = p["data"]
    for offset in range(num_beats):
        if offset in src_map:
            checks.append({
                "word_offset": dst_base + offset,
                "expected": src_map[offset],
            })


# ── Functional model ──

sa_bf16 = SARTLLinearFunction(
    rows=32, cols=32,
    out_fmt_sel=OutputFmtSel.OutBF16,
)

# ── Generate inputs ──

X = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 0)).astype(np.float32)
W = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 1)).astype(np.float32)
zero_bias = np.zeros((TILE, TILE), dtype=np.float32)
echo_tile = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 2)).astype(np.float32)

A_vpu = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 3).astype(np.float32)
B_vpu = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 4).astype(np.float32)

A_vpu_b0_rows = float_matrix_to_bf16_rows(A_vpu[:, :16])
A_vpu_b1_rows = float_matrix_to_bf16_rows(A_vpu[:, 16:])
B_vpu_b0_rows = float_matrix_to_bf16_rows(B_vpu[:, :16])
B_vpu_b1_rows = float_matrix_to_bf16_rows(B_vpu[:, 16:])

# ── Reference: C0 = X @ W (MXU0 SA, zero bias) ──

C0 = sa_bf16(
    torch.from_numpy(X),
    torch.from_numpy(W.T.copy()),
    b_q=torch.zeros(TILE), scale_exp=0,
).numpy().astype(np.float32)

# ── Reference: D = vadd(A, B) (VPU) ──

add_b0_rows = run_binary_rows("add", A_vpu_b0_rows, B_vpu_b0_rows)
add_b1_rows = run_binary_rows("add", A_vpu_b1_rows, B_vpu_b1_rows)

# ── Emit preloads ──

preloads = []

emit_fp8_preload(preloads, X_BASE, X)
emit_fp8_preload(preloads, W_BASE, W)
emit_fp8_preload(preloads, BIAS_BASE, zero_bias)
emit_bf16_rows_preload(preloads, A_VPU_B0_BASE, A_vpu_b0_rows)
emit_bf16_rows_preload(preloads, A_VPU_B1_BASE, A_vpu_b1_rows)
emit_bf16_rows_preload(preloads, B_VPU_B0_BASE, B_vpu_b0_rows)
emit_bf16_rows_preload(preloads, B_VPU_B1_BASE, B_vpu_b1_rows)
emit_fp8_preload(preloads, ECHO_TILE_BASE, echo_tile)

# ── Emit checks ──

checks = []

# Compute results
emit_bf16_split_check(checks, C0_LO_BASE, C0_HI_BASE, C0)
emit_bf16_rows_check(checks, ADD_B0_BASE, ADD_B1_BASE, add_b0_rows, add_b1_rows)

# Echo checks: DMA LOAD→STORE round-trip verification
# echo0 = A_vpu b1  (DRAM 0x1000 → VMEM[1024] → DRAM 0x3000)
emit_echo_checks(checks, preloads, A_VPU_B1_BASE, ECHO0_BASE)
# echo1 = B_vpu b0  (DRAM 0x1400 → VMEM[1280] → DRAM 0x3400)
emit_echo_checks(checks, preloads, B_VPU_B0_BASE, ECHO1_BASE)
# echo2 = B_vpu b1  (DRAM 0x1800 → VMEM[1536] → DRAM 0x3800)
emit_echo_checks(checks, preloads, B_VPU_B1_BASE, ECHO2_BASE)
# echo3 = echo_tile (DRAM 0x1C00 → VMEM[1792] → DRAM 0x3C00)
emit_echo_checks(checks, preloads, ECHO_TILE_BASE, ECHO3_BASE)

sys.stderr.write(
    f"[gen_dma_8chan_burst] seed={SEED}\n"
    f"  C0 range: [{C0.min():.4f}, {C0.max():.4f}]\n"
    f"  preloads={len(preloads)} checks={len(checks)}\n"
)

emit_test_data(preloads, checks, timeout=1500000)

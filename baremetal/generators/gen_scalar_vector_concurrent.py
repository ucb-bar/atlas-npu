#!/usr/bin/env python3
"""
gen_scalar_vector_concurrent.py

Generate test vectors for scalar_vector_concurrent.S.

Models the interleaved-bank-safe schedule used by the assembly:
    Phase 1: VLOAD + scalar stores only
    Phase 2: VSTORE + scalar loads only
    Phase 3: VLOAD + scalar stores from Phase 2 readback
    Phase 3b: XLU window with scalar LW/NOP/SW chains
    Phase 4: VLOAD bias + scalar stores, then MXU compute + scalar chains
    Phase 5: Final VSTOREs + scalar readback overlap

Plus VSTORE echo of tile_A and MXU0 C0 = tile_A @ tile_W.

Usage:
    python3 gen_scalar_vector_concurrent.py [seed]
"""

import os
import sys
import numpy as np
try:
    import torch
except ModuleNotFoundError:
    torch = None

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (
    rand_matrix_fp8_safe,
    quantize_fp8,
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    preloads_from_words_packed,
    checks_from_words_packed,
    emit_test_data,
    fp8_matmul_reference,
)
if torch is not None:
    from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
    from software_models.mxu1_ipt.fp_formats import OutputFmtSel

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
TILE = 32
NUM_BEATS = 32
WORDS_PER_BEAT = 8

# ── DRAM byte offsets ──
TILE_A_BASE     = 0x0000
TILE_B_BASE     = 0x0400
TILE_W_BASE     = 0x0800
BIAS_BASE       = 0x0C00
ZEROS_BASE      = 0x1000
VSTORE_OUT_BASE = 0x1400
STAGING1_BASE   = 0x1800  # stage1 region
STAGING2_BASE   = 0x1C00  # stage2 region
STAGING3_BASE   = 0x2000  # stage3 region
C0_LO_BASE      = 0x2400
C0_HI_BASE      = 0x2800


def beat_offset(byte_off):
    assert byte_off % 32 == 0
    return byte_off // 32


def pack_256bit(words_8):
    """Pack 8 × 32-bit words (little-endian) into a 256-bit hex string."""
    val = 0
    for i, w in enumerate(words_8):
        val |= (int(w) & 0xFFFFFFFF) << (32 * i)
    return f"0x{val:064x}"


def emit_fp8_preload(preloads, byte_off, tile):
    preloads += preloads_from_words_packed(
        beat_offset(byte_off), matrix_to_fp8_words(tile)
    )


def emit_fp8_check(checks, byte_off, tile):
    checks += checks_from_words_packed(
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


def emit_staging_check(checks, base_byte_off, line0_words):
    """Emit checks for a staging tile: line 0 has specific words, rest zeros."""
    base = beat_offset(base_byte_off)
    zero_beat = pack_256bit([0] * WORDS_PER_BEAT)
    for b in range(NUM_BEATS):
        if b == 0:
            checks.append({
                "word_offset": base,
                "expected": pack_256bit(line0_words),
            })
        else:
            checks.append({
                "word_offset": base + b,
                "expected": zero_beat,
            })


def word_val(w):
    """Ensure word is a plain int."""
    if isinstance(w, str):
        return int(w, 16) & 0xFFFFFFFF
    return int(w) & 0xFFFFFFFF


# ── Functional model ──

if torch is not None:
    sa_bf16 = SARTLLinearFunction(
        rows=32, cols=32,
        out_fmt_sel=OutputFmtSel.OutBF16,
    )

# ── Generate inputs ──

tile_A = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 0)).astype(np.float32)
tile_B = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 1)).astype(np.float32)
tile_W = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 2)).astype(np.float32)
zero_bias = np.zeros((TILE, TILE), dtype=np.float32)
zeros_tile = np.zeros((TILE, TILE), dtype=np.float32)

# Extract specific 32-bit words from tile data
B_words = [word_val(w) for w in matrix_to_fp8_words(tile_B)]
W_words = [word_val(w) for w in matrix_to_fp8_words(tile_W)]

B_w0 = B_words[0]  # tile_B row 0, bytes [0:3]
B_w1 = B_words[1]  # tile_B row 0, bytes [4:7]
W_w0 = W_words[0]  # tile_W row 0, bytes [0:3]

CAFE = 0xCAFEBABE
DEAD = 0xDEADBEEF

# ── Trace scalar data flow ──
#
# Final state of each staging region's line 0 (8 words):

bank5_line0 = [
    CAFE,    # word 0: Phase 1 constant
    DEAD,    # word 1: Phase 1 constant
    B_w0,    # word 2: Phase 3 store from tile_B word 0
    B_w1,    # word 3: Phase 3 store from tile_B word 1
    DEAD,    # word 4: Phase 3b chain from stage3 word 2
    W_w0,    # word 5: Phase 4 VLOAD-bias window store
    B_w0,    # word 6: Phase 4 MXU-window chain from stage3 word 3
    0,       # word 7: untouched
]

bank6_line0 = [
    DEAD,    # word 0: Phase 1 constant
    CAFE,    # word 1: Phase 1 constant
    W_w0,    # word 2: Phase 3 store from tile_W word 0
    CAFE,    # word 3: Phase 3 store from stage1 word 0
    B_w0,    # word 4: Phase 3b chain from stage1 word 2
    B_w1,    # word 5: Phase 4 VLOAD-bias window store
    DEAD,    # word 6: Phase 4 MXU-window chain from stage1 word 4
    0,       # word 7: untouched
]

bank7_line0 = [
    CAFE,    # word 0: Phase 1 constant
    DEAD,    # word 1: Phase 1 constant
    DEAD,    # word 2: Phase 3 store from stage2 word 0
    B_w0,    # word 3: Phase 3 store from tile_B word 0
    CAFE,    # word 4: Phase 3b chain from stage2 word 3
    CAFE,    # word 5: Phase 4 VLOAD-bias window store
    W_w0,    # word 6: Phase 4 MXU-window chain from stage2 word 2
    0,       # word 7: untouched
]

# ── Reference: MXU0 C0 = tile_A @ tile_W ──

if torch is not None:
    C0 = sa_bf16(
        torch.from_numpy(tile_A),
        torch.from_numpy(tile_W.T.copy()),
        b_q=torch.zeros(TILE), scale_exp=0,
    ).numpy().astype(np.float32)
else:
    # Fallback for environments without torch: MXU0 BF16 output with FP8 inputs.
    C0 = fp8_matmul_reference(tile_A, tile_W).astype(np.float32)

# ── Emit preloads ──

preloads = []
emit_fp8_preload(preloads, TILE_A_BASE, tile_A)
emit_fp8_preload(preloads, TILE_B_BASE, tile_B)
emit_fp8_preload(preloads, TILE_W_BASE, tile_W)
emit_fp8_preload(preloads, BIAS_BASE, zero_bias)
emit_fp8_preload(preloads, ZEROS_BASE, zeros_tile)

# ── Emit checks ──

checks = []

# VSTORE output = tile_A
emit_fp8_check(checks, VSTORE_OUT_BASE, tile_A)

# Staging regions with scalar-written data
emit_staging_check(checks, STAGING1_BASE, bank5_line0)
emit_staging_check(checks, STAGING2_BASE, bank6_line0)
emit_staging_check(checks, STAGING3_BASE, bank7_line0)

# MXU0 result
emit_bf16_split_check(checks, C0_LO_BASE, C0_HI_BASE, C0)

# ── Summary ──

sys.stderr.write(
    f"[gen_scalar_vector_concurrent] seed={SEED}\n"
    f"  tile_B words: w0=0x{B_w0:08X} w1=0x{B_w1:08X}\n"
    f"  tile_W word:  w0=0x{W_w0:08X}\n"
    f"  C0 range: [{C0.min():.4f}, {C0.max():.4f}]\n"
    f"  preloads={len(preloads)} checks={len(checks)}\n"
    f"  scalar ops: 6 SW (Phase 1) + 5 LW (Phase 2) + 6 SW (Phase 3 VLOAD)\n"
    f"            + 3 LW/3 SW (Phase 3 XLU) + 3 SW (Phase 4 VLOAD)\n"
    f"            + 3 LW/3 SW (Phase 4 MXU) + 2 LW (Phase 5 lo)\n"
    f"            + 2 LW (Phase 5 hi) = 36 total\n"
)

emit_test_data(preloads, checks, timeout=800000)

#!/usr/bin/env python3
"""
gen_perf_mm_single.py

Preload generator for perf_mm_single.S — single-matmul MXU1
utilization characterization. Issues exactly one VMATMUL (overwrite
mode) to MXU1, then banks the mcycles delta across the timed window
into CSR_DBG1 as the utilization denominator.

No DRAM output checks: pass/fail is gated purely by the host harness's
@PERF_UTIL_THRESHOLD evaluated against MAC-based util computed from
the @PERF_MACS_MXU1 directive.

DRAM layout must match perf_mm_single.S:
    0x0000  X    (1024B FP8, 32x32) — activation tile
    0x0400  W    (1024B FP8, 32x32) — MXU1 weight tile (pre-transpose)

Usage:
    python3 gen_perf_mm_single.py [seed]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (
    rand_matrix_fp8_safe,
    quantize_fp8,
    matrix_to_fp8_words,
    preloads_from_words_packed,
    emit_test_data,
)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TILE = 32

X_BASE = 0x0000
W_BASE = 0x0400


def beat_offset(byte_off: int) -> int:
    assert byte_off % 32 == 0
    return byte_off // 32


def emit_fp8_preload(preloads, byte_off, tile):
    preloads += preloads_from_words_packed(
        beat_offset(byte_off), matrix_to_fp8_words(tile)
    )


# Values don't affect perf or correctness (no checks). Any valid FP8
# tile works; use deterministic random for reproducibility.
X = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 0)).astype(np.float32)
W = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 1)).astype(np.float32)

preloads = []
emit_fp8_preload(preloads, X_BASE, X)
emit_fp8_preload(preloads, W_BASE, W)

sys.stderr.write(
    f"[gen_perf_mm_single] seed={SEED} preloads={len(preloads)} checks=0\n"
)

emit_test_data(preloads, [], timeout=200000)

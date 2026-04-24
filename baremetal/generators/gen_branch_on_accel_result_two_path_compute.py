#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import (
    emit_test_data,
    preloads_from_words_packed,
    checks_from_words_packed,
    matrix_to_bf16_words,
    quantize_bf16,
    float_to_bf16_bits,
    bf16_bits_to_float,
)

TILE = 32

# Deterministic matrix with positive row sums so the BLT-to-neg_path is not taken.
mat = np.zeros((TILE, TILE), dtype=np.float32)
for i in range(TILE):
    for j in range(TILE):
        mat[i, j] = 0.25 + 0.03125 * ((i + j) % 8)
mat = quantize_bf16(mat).astype(np.float32)

rows = np.sum(mat.astype(np.float32), axis=1)
rows_q = np.array([bf16_bits_to_float(float_to_bf16_bits(v)) for v in rows], dtype=np.float32)
rowsum_bcast = np.repeat(rows_q[:, None], TILE, axis=1).astype(np.float32)
final_out = np.maximum(rowsum_bcast, 0.0).astype(np.float32)
final_out = quantize_bf16(final_out).astype(np.float32)

preloads = []
checks = []
preloads += preloads_from_words_packed(0x0000 // 32, matrix_to_bf16_words(mat[:, :16]))
preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_bf16_words(mat[:, 16:]))

checks += checks_from_words_packed(0x0800 // 32, matrix_to_bf16_words(rowsum_bcast[:, :16]))
checks += checks_from_words_packed(0x0C00 // 32, matrix_to_bf16_words(rowsum_bcast[:, 16:]))
checks += checks_from_words_packed(0x1000 // 32, matrix_to_bf16_words(final_out[:, :16]))
checks += checks_from_words_packed(0x1400 // 32, matrix_to_bf16_words(final_out[:, 16:]))

emit_test_data(preloads, checks, timeout=500000)

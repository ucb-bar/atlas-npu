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

def make_tile(seed_bias: float) -> np.ndarray:
    m = np.zeros((TILE, TILE), dtype=np.float32)
    for i in range(TILE):
        for j in range(TILE):
            m[i, j] = seed_bias + 0.125 + ((i * 7 + j * 3) % 5) * 0.0625
    return quantize_bf16(m).astype(np.float32)

inp0 = make_tile(0.0)
inp1 = make_tile(0.25)

preloads = []
checks = []
for idx, mat in enumerate([inp0, inp1]):
    base_in = (0x0000 + idx * 0x0800) // 32
    preloads += preloads_from_words_packed(base_in + 0x0000 // 32, matrix_to_bf16_words(mat[:, :16]))
    preloads += preloads_from_words_packed(base_in + 0x0400 // 32, matrix_to_bf16_words(mat[:, 16:]))

    rows = np.sum(mat.astype(np.float32), axis=1)
    rows_q = np.array([bf16_bits_to_float(float_to_bf16_bits(v)) for v in rows], dtype=np.float32)
    rowsum = np.repeat(rows_q[:, None], TILE, axis=1).astype(np.float32)
    final = quantize_bf16(np.maximum(rowsum, 0.0)).astype(np.float32)

    base_out = (0x2000 + idx * 0x0800) // 32
    checks += checks_from_words_packed(base_out + 0x0000 // 32, matrix_to_bf16_words(final[:, :16]))
    checks += checks_from_words_packed(base_out + 0x0400 // 32, matrix_to_bf16_words(final[:, 16:]))

emit_test_data(preloads, checks, timeout=800000)

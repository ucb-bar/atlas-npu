#!/usr/bin/env python3
import os
import sys
import math
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

def qbf16_scalar(v: float) -> float:
    return bf16_bits_to_float(float_to_bf16_bits(v))

# Build class-directed positive-domain-heavy input so all four ops are defined.
vals = [
    0.0, -0.0,
    2.0**-20, 2.0**-14, 2.0**-10, 2.0**-6,
    0.125, 0.15625, 0.1875, 0.21875,
    0.25, 0.5, 0.75,
    1.0, 1.125, 1.5, 1.875,
    2.0, 3.0, 4.0, 8.0, 16.0,
    31.0/16.0, 33.0/16.0,
    0.9921875, 1.0078125,
    63.0/64.0, 65.0/64.0,
    32.0, 64.0, 128.0, 192.0,
]
mat = np.zeros((TILE, TILE), dtype=np.float32)
for i in range(TILE):
    for j in range(TILE):
        mat[i, j] = vals[(i * TILE + j) % len(vals)]
mat = quantize_bf16(mat).astype(np.float32)

sqrt_out = np.vectorize(lambda x: qbf16_scalar(math.sqrt(float(x))))(mat).astype(np.float32)
recip_out = np.vectorize(lambda x: qbf16_scalar(1.0 / float(x)) if float(x) != 0.0 else qbf16_scalar(float('inf')))(mat).astype(np.float32)
log2_out = np.vectorize(lambda x: qbf16_scalar(math.log2(float(x))) if float(x) > 0.0 else qbf16_scalar(float('-inf')))(mat).astype(np.float32)
exp2_out = np.vectorize(lambda x: qbf16_scalar(2.0 ** float(x)) if math.isfinite(float(x)) else (qbf16_scalar(0.0) if float(x) < 0 else qbf16_scalar(float('inf'))))(log2_out).astype(np.float32)

preloads = []
checks = []

preloads += preloads_from_words_packed(0x0000 // 32, matrix_to_bf16_words(mat[:, :16]))
preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_bf16_words(mat[:, 16:]))

checks += checks_from_words_packed(0x0800 // 32, matrix_to_bf16_words(sqrt_out[:, :16]))
checks += checks_from_words_packed(0x0C00 // 32, matrix_to_bf16_words(sqrt_out[:, 16:]))

checks += checks_from_words_packed(0x1000 // 32, matrix_to_bf16_words(recip_out[:, :16]))
checks += checks_from_words_packed(0x1400 // 32, matrix_to_bf16_words(recip_out[:, 16:]))

checks += checks_from_words_packed(0x1800 // 32, matrix_to_bf16_words(log2_out[:, :16]))
checks += checks_from_words_packed(0x1C00 // 32, matrix_to_bf16_words(log2_out[:, 16:]))

checks += checks_from_words_packed(0x2000 // 32, matrix_to_bf16_words(exp2_out[:, :16]))
checks += checks_from_words_packed(0x2400 // 32, matrix_to_bf16_words(exp2_out[:, 16:]))

emit_test_data(preloads, checks, timeout=700000)

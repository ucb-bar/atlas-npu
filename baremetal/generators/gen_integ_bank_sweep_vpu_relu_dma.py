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
)

TILE = 32
TIMEOUT = 800000


def make_tile(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    # Wide signed range with forced negatives/positives/zeros to toggle ReLU paths.
    mat = rng.uniform(-6.0, 6.0, size=(TILE, TILE)).astype(np.float32)
    mat[0, 0] = -7.5
    mat[0, 1] = -0.25
    mat[0, 2] = 0.0
    mat[0, 3] = 0.25
    mat[1, 0] = 3.5
    mat[1, 1] = -3.5
    return quantize_bf16(mat).astype(np.float32)


def relu_bf16(mat: np.ndarray) -> np.ndarray:
    out = np.maximum(mat, 0.0).astype(np.float32)
    return quantize_bf16(out).astype(np.float32)


inputs = [make_tile(101), make_tile(202), make_tile(303), make_tile(404)]
outputs = [relu_bf16(x) for x in inputs]

preloads = []
checks = []

input_byte_offsets = [0x0000, 0x0800, 0x1000, 0x1800]
output_byte_offsets = [0x2000, 0x2800, 0x3000, 0x3800]

for byte_off, mat in zip(input_byte_offsets, inputs):
    words = matrix_to_bf16_words(mat[:, :16]) + matrix_to_bf16_words(mat[:, 16:])
    preloads += preloads_from_words_packed(byte_off // 32, words)

for byte_off, mat in zip(output_byte_offsets, outputs):
    words = matrix_to_bf16_words(mat[:, :16]) + matrix_to_bf16_words(mat[:, 16:])
    checks += checks_from_words_packed(byte_off // 32, words)

emit_test_data(preloads, checks, timeout=TIMEOUT)

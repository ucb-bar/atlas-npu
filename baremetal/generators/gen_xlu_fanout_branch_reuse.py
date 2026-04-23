#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import (
    emit_test_data,
    preloads_from_words_packed,
    checks_from_words_packed,
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    quantize_fp8,
    fp8_matmul_reference,
)

TILE = 32

# Deterministic nonzero patterns so transpose sentinel is nonzero.
A = np.zeros((TILE, TILE), dtype=np.float32)
B = np.zeros((TILE, TILE), dtype=np.float32)
for i in range(TILE):
    for j in range(TILE):
        A[i, j] = 0.5 + ((i * 3 + j) % 7) * 0.125
        B[i, j] = 0.25 + ((i + j * 5) % 9) * 0.125

A_q = quantize_fp8(A).astype(np.float32)
B_q = quantize_fp8(B).astype(np.float32)
BT_q = quantize_fp8(B_q.T).astype(np.float32)

# Assembly pushes BT_q as weight; MXU computes A @ (BT_q)^T = A @ B_q.
C = fp8_matmul_reference(A_q, BT_q).astype(np.float32)

preloads = []
checks = []
preloads += preloads_from_words_packed(0x0000 // 32, matrix_to_fp8_words(A_q))
preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_fp8_words(B_q))

checks += checks_from_words_packed(0x1000 // 32, matrix_to_fp8_words(BT_q))
checks += checks_from_words_packed(0x1800 // 32, matrix_to_bf16_words(C[:, :16]))
checks += checks_from_words_packed(0x1C00 // 32, matrix_to_bf16_words(C[:, 16:]))

emit_test_data(preloads, checks, timeout=600000)

#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import (
    emit_test_data,
    preloads_from_words_packed,
    checks_from_words_packed,
    bf16_bits_to_words,
    matrix_to_fp8_words,
    mxu0_sa_bf16_bits,
    quantize_fp8,
    rand_matrix_fp8_safe,
)

TILE = 32


A0 = quantize_fp8(rand_matrix_fp8_safe(32, 32, seed=101)).astype(np.float32)
B0 = quantize_fp8(rand_matrix_fp8_safe(32, 32, seed=102)).astype(np.float32)
A1 = quantize_fp8(rand_matrix_fp8_safe(32, 32, seed=103)).astype(np.float32)
B1 = quantize_fp8(rand_matrix_fp8_safe(32, 32, seed=104)).astype(np.float32)

# Assembly transposes B before pushing as MXU weight, so result is A0 @ B0.
# Use the SA model because MXU0 rounds through the PE chain, not as one fp32 dot.
C0_bits = mxu0_sa_bf16_bits(A0, B0)
B1_t = quantize_fp8(B1.T).astype(np.float32)

preloads = []
checks = []

preloads += preloads_from_words_packed(0x0000 // 32, matrix_to_fp8_words(A0))
preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_fp8_words(B0))
preloads += preloads_from_words_packed(0x0800 // 32, matrix_to_fp8_words(A1))
preloads += preloads_from_words_packed(0x0C00 // 32, matrix_to_fp8_words(B1))

checks += checks_from_words_packed(
    0x2000 // 32, bf16_bits_to_words(C0_bits[:, :16])
)
checks += checks_from_words_packed(
    0x2400 // 32, bf16_bits_to_words(C0_bits[:, 16:])
)
checks += checks_from_words_packed(0x2800 // 32, matrix_to_fp8_words(B1_t))

emit_test_data(preloads, checks, timeout=500000)

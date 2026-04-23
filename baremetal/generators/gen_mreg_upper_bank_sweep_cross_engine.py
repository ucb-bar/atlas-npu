#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from gen_utils import (
    emit_test_data,
    preloads_from_words_packed,
    checks_from_words_packed,
    matrix_to_bf16_words,
    matrix_to_fp8_words,
    quantize_bf16,
    quantize_fp8,
    rand_matrix_fp8_safe,
)
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

TILE = 32
TIMEOUT = 500000

# DRAM byte offsets from assembly.
A_FP8_OFF = 0x0000
B_FP8_RAW_OFF = 0x0400
SEED_BF16_OFF = 0x0800
BANK58_OUT_OFF = 0x1000
BANK59_OUT_OFF = 0x1400
MXU_BF16_OUT_OFF = 0x1800
RELU_OUT_OFF = 0x2000

# Deterministic, auditable inputs.
A = rand_matrix_fp8_safe(TILE, TILE, seed=101)
B = rand_matrix_fp8_safe(TILE, TILE, seed=202)
seed_lo = quantize_bf16(np.linspace(-4.0, 3.75, TILE * 16, dtype=np.float32).reshape(TILE, 16))
seed_hi = quantize_bf16(np.linspace(4.0, -3.75, TILE * 16, dtype=np.float32).reshape(TILE, 16))

A_q = quantize_fp8(A).astype(np.float32)
B_q = quantize_fp8(B).astype(np.float32)

# Assembly transposes B in XLU before pushing to MXU1, and MXU computes x @ w^T.
# Therefore pass w = B (not B.T) to the RTL model so y = A @ B^T? No: model contract is
# y = x @ w^T, so to realize A @ B after hardware transpose(B)->B^T push, give w = B.T.
# Then y = A @ (B.T)^T = A @ B.
ipt = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutBF16)
C = ipt(torch.from_numpy(A_q), torch.from_numpy(B_q.T.copy())).numpy().astype(np.float32)
C_relu = np.maximum(C, 0.0).astype(np.float32)

preloads = []
preloads += preloads_from_words_packed(A_FP8_OFF // 32, matrix_to_fp8_words(A_q))
preloads += preloads_from_words_packed(B_FP8_RAW_OFF // 32, matrix_to_fp8_words(B_q))
preloads += preloads_from_words_packed(SEED_BF16_OFF // 32, matrix_to_bf16_words(seed_lo))
preloads += preloads_from_words_packed((SEED_BF16_OFF + 0x0400) // 32, matrix_to_bf16_words(seed_hi))

checks = []
checks += checks_from_words_packed(BANK58_OUT_OFF // 32, matrix_to_bf16_words(seed_lo))
checks += checks_from_words_packed(BANK59_OUT_OFF // 32, matrix_to_bf16_words(seed_hi))
checks += checks_from_words_packed(MXU_BF16_OUT_OFF // 32, matrix_to_bf16_words(C[:, :16]))
checks += checks_from_words_packed((MXU_BF16_OUT_OFF + 0x0400) // 32, matrix_to_bf16_words(C[:, 16:]))
checks += checks_from_words_packed(RELU_OUT_OFF // 32, matrix_to_bf16_words(C_relu[:, :16]))
checks += checks_from_words_packed((RELU_OUT_OFF + 0x0400) // 32, matrix_to_bf16_words(C_relu[:, 16:]))

emit_test_data(preloads, checks, timeout=TIMEOUT)

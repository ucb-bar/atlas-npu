#!/usr/bin/env python3
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import (
    emit_test_data,
    preloads_from_words_packed,
    checks_from_words_packed,
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    quantize_fp8,
    quantize_bf16,
)
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

TILE = 32
A = np.zeros((TILE, TILE), dtype=np.float32)
B = np.zeros((TILE, TILE), dtype=np.float32)
for i in range(TILE):
    for j in range(TILE):
        A[i, j] = (-1.0 if (i % 3) == 0 else 1.0) * (0.25 + ((i + j) % 6) * 0.125)
        B[i, j] = 0.375 + ((i * 2 + j * 3) % 5) * 0.125

A_q = quantize_fp8(A).astype(np.float32)
B_q = quantize_fp8(B).astype(np.float32)

# Assembly stores B in logical orientation, XLU-transposes it in-place, then
# VMATPUSH.W.MXU1 consumes that transposed tile. MXU1's RTL model computes
# F.linear semantics y = x @ w^T, so pass w = B_q.T to realize A @ B.
ipt = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutBF16)
raw = ipt(
    torch.from_numpy(A_q),
    torch.from_numpy(B_q.T.copy()),
    scale_exp=0,
).numpy().astype(np.float32)

relu = quantize_bf16(np.maximum(raw, 0.0)).astype(np.float32)

preloads = []
checks = []
preloads += preloads_from_words_packed(0x0000 // 32, matrix_to_fp8_words(A_q))
preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_fp8_words(B_q))
checks += checks_from_words_packed(0x1000 // 32, matrix_to_bf16_words(raw[:, :16]))
checks += checks_from_words_packed(0x1400 // 32, matrix_to_bf16_words(raw[:, 16:]))
checks += checks_from_words_packed(0x1800 // 32, matrix_to_bf16_words(relu[:, :16]))
checks += checks_from_words_packed(0x1C00 // 32, matrix_to_bf16_words(relu[:, 16:]))

emit_test_data(preloads, checks, timeout=700000)

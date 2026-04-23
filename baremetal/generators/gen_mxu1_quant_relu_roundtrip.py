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
    float_to_bf16_bits,
    bf16_bits_to_float,
)
from vpu_gen_utils import run_unary_rows, run_fp8_pack_rows, run_fp8_unpack_rows
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

TILE = 32
TIMEOUT = 500000

A_OFF = 0x0000
B_OFF = 0x0400
SCALE_OFF = 0x0800
RAW_OUT_OFF = 0x1000
PACK_OUT_OFF = 0x1800
UNPACK_OUT_OFF = 0x2000

SCALE_E8M0 = 0x7E

rng_a = np.random.RandomState(202)
rng_b = np.random.RandomState(303)
A = rng_a.uniform(-1.25, 1.25, size=(TILE, TILE)).astype(np.float32)
B = rng_b.uniform(-1.25, 1.25, size=(TILE, TILE)).astype(np.float32)

A_q = quantize_fp8(A).astype(np.float32)
B_q = quantize_fp8(B).astype(np.float32)

# Assembly pushes B directly as weights; MXU computes act @ weight^T.
# Therefore preload B^T so the architectural result is A @ B.
B_preload = B_q.T.copy()

ipt = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutBF16)
C = ipt(torch.from_numpy(A_q), torch.from_numpy(B_preload)).numpy().astype(np.float32)

# MXU1 BF16 result in VPU bank-pair row layout: first 32 rows are bank 8 (cols 0:16),
# next 32 rows are bank 9 (cols 16:32).
C_bits = np.vectorize(float_to_bf16_bits)(C).astype(np.uint16)
raw_lo_rows = [list(C_bits[r, :16]) for r in range(TILE)]
raw_hi_rows = [list(C_bits[r, 16:32]) for r in range(TILE)]
raw_tensor_rows = raw_lo_rows + raw_hi_rows

relu_rows = run_unary_rows("relu", raw_tensor_rows)
pack_rows = run_fp8_pack_rows(relu_rows[:32], relu_rows[32:64], SCALE_E8M0)
unpack_rows = run_fp8_unpack_rows(pack_rows, SCALE_E8M0)

relu_lo = np.array(relu_rows[:32], dtype=np.uint16)
relu_hi = np.array(relu_rows[32:64], dtype=np.uint16)
relu_mat_bits = np.concatenate([relu_lo, relu_hi], axis=1)

unpack_lo = np.array(unpack_rows[:32], dtype=np.uint16)
unpack_hi = np.array(unpack_rows[32:64], dtype=np.uint16)
unpack_mat_bits = np.concatenate([unpack_lo, unpack_hi], axis=1)

relu_mat = np.vectorize(bf16_bits_to_float)(relu_mat_bits).astype(np.float32)
unpack_mat = np.vectorize(bf16_bits_to_float)(unpack_mat_bits).astype(np.float32)

# VSTORE of one FP8 MREG drains architectural storage exactly as one 32x32 FP8 tile:
# 32 rows, 32 FP8 elements per row, packed by matrix_to_fp8_words into 8 words/row.
# run_fp8_pack_rows returns 32 rows x 16 uint16 packed slots; reinterpret each slot as
# two FP8 bytes in little-endian lane order, then repack with the canonical helper.
def packed_fp8_rows_to_matrix(rows):
    mat = np.zeros((32, 32), dtype=np.float32)
    for r, row in enumerate(rows):
        vals = []
        for slot in row:
            slot = int(slot) & 0xFFFF
            lo = slot & 0xFF
            hi = (slot >> 8) & 0xFF
            vals.append(lo)
            vals.append(hi)
        for c, bits in enumerate(vals):
            from gen_utils import fp8_e4m3_bits_to_float
            mat[r, c] = fp8_e4m3_bits_to_float(bits)
    return mat

pack_mat = packed_fp8_rows_to_matrix(pack_rows)

preloads = []
preloads += preloads_from_words_packed(A_OFF // 32, matrix_to_fp8_words(A_q))
preloads += preloads_from_words_packed(B_OFF // 32, matrix_to_fp8_words(B_preload))
preloads += preloads_from_words_packed(SCALE_OFF // 32, [SCALE_E8M0])

checks = []
checks += checks_from_words_packed(RAW_OUT_OFF // 32, matrix_to_bf16_words(C[:, :16]))
checks += checks_from_words_packed((RAW_OUT_OFF + 1024) // 32, matrix_to_bf16_words(C[:, 16:]))
checks += checks_from_words_packed(PACK_OUT_OFF // 32, matrix_to_fp8_words(pack_mat))
checks += checks_from_words_packed(UNPACK_OUT_OFF // 32, matrix_to_bf16_words(unpack_mat[:, :16]))
checks += checks_from_words_packed((UNPACK_OUT_OFF + 1024) // 32, matrix_to_bf16_words(unpack_mat[:, 16:]))

emit_test_data(preloads, checks, timeout=TIMEOUT)

#!/usr/bin/env python3
"""Generate test vectors for `smolvla_matmul_mxu1.S`.

Port of npu-model/npu_model/configs/programs/smolvla_matmul.py.
Single-tile FP8 32x32 matmul C = A @ B. Inputs are
`torch.randint(-8, 8, int8).to(float8_e4m3fn)` pulled sequentially
from seed 42 (first A, then B). Output is BF16, popped from the MXU1
accumulator and written as two 32x16 halves (cols 0-15 in m2, cols
16-31 in m3), matching npu-model's stacked output layout.

Atlas assembly uses VTRPOSE.XLU on B before VMATPUSH.W so the MXU
computes A @ B rather than A @ B^T. Golden goes through atlas's
IPTLinearRTLFunction (RTL-accurate MXU1 software model).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (
    matrix_to_bf16_words,
    matrix_to_fp8_words,
    pack_words_into_beats,
    emit_test_data,
)
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


TILE = 32
WORDS_PER_BEAT = 8
TIMEOUT = 30000

A_BASE = 0
B_BASE = A_BASE + TILE
C_BANK2_BASE = B_BASE + TILE
C_BANK3_BASE = C_BANK2_BASE + TILE


def main():
    ipt_bf16 = IPTLinearRTLFunction(
        vec_len=TILE, num_lanes=TILE, pipeline_depth=1, out_fmt_sel=OutputFmtSel.OutBF16
    )

    torch.manual_seed(42)
    input_a_fp8 = torch.randint(-8, 8, (TILE, TILE), dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    input_b_fp8 = torch.randint(-8, 8, (TILE, TILE), dtype=torch.int8).to(
        torch.float8_e4m3fn
    )

    a_q = input_a_fp8.to(torch.float32).numpy()
    b_q = input_b_fp8.to(torch.float32).numpy()

    # IPTLinearRTLFunction computes F.linear: y = x @ w^T. For C = A @ B,
    # pass w = B^T so y = A @ B^T^T = A @ B.
    a_t = torch.from_numpy(a_q)
    w_t = torch.from_numpy(b_q.T.copy())
    c_ref = ipt_bf16(a_t, w_t, scale_exp=0).numpy().astype(np.float32)

    a_beats = pack_words_into_beats(matrix_to_fp8_words(a_q), WORDS_PER_BEAT)
    b_beats = pack_words_into_beats(matrix_to_fp8_words(b_q), WORDS_PER_BEAT)

    c_bank2_words, c_bank3_words = [], []
    for r in range(TILE):
        c_bank2_words.extend(matrix_to_bf16_words(c_ref[r, 0:16].reshape(1, -1)))
        c_bank3_words.extend(matrix_to_bf16_words(c_ref[r, 16:32].reshape(1, -1)))
    c_bank2_beats = pack_words_into_beats(c_bank2_words, WORDS_PER_BEAT)
    c_bank3_beats = pack_words_into_beats(c_bank3_words, WORDS_PER_BEAT)

    preloads = []
    for i, beat in enumerate(a_beats):
        preloads.append({"word_offset": A_BASE + i, "data": f"0x{beat:X}"})
    for i, beat in enumerate(b_beats):
        preloads.append({"word_offset": B_BASE + i, "data": f"0x{beat:X}"})

    checks = []
    for i, beat in enumerate(c_bank2_beats):
        checks.append({"word_offset": C_BANK2_BASE + i, "expected": f"0x{beat:X}"})
    for i, beat in enumerate(c_bank3_beats):
        checks.append({"word_offset": C_BANK3_BASE + i, "expected": f"0x{beat:X}"})

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

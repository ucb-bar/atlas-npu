#!/usr/bin/env python3
"""Generate test vectors for `smolvla_matmul_k_chain_mxu1.S`.

Port of npu-model/npu_model/configs/programs/smolvla_matmul_k_chain.py.
Two-K-tile FP8 matmul that exercises the MXU1 accumulator chain:

    C[32,32] = A[:, 0:32]  @ B[0:32,  :]      (VMATMUL.MXU1     → reset)
             + A[:, 32:64] @ B[32:64, :]      (VMATMUL.ACC.MXU1 → accumulate)

Seed 44 matches npu-model. Output is BF16, split across m8/m9
(cols 0-15 and 16-31). Golden through atlas's IPTLinearRTLFunction,
which handles per-K-tile BF16 rounding internally and mirrors
hardware exactly.
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
K_TILES = 2  # K = 64 = 2 × 32
WORDS_PER_BEAT = 8
TIMEOUT = 40000

A_K0_BASE = 0                       # beats 0..31
A_K1_BASE = A_K0_BASE + TILE        # beats 32..63
B_K0_BASE = A_K1_BASE + TILE        # beats 64..95
B_K1_BASE = B_K0_BASE + TILE        # beats 96..127
C_BANK2_BASE = B_K1_BASE + TILE     # beats 128..159
C_BANK3_BASE = C_BANK2_BASE + TILE  # beats 160..191


def main():
    ipt_bf16 = IPTLinearRTLFunction(
        vec_len=TILE, num_lanes=TILE, pipeline_depth=1, out_fmt_sel=OutputFmtSel.OutBF16
    )

    torch.manual_seed(44)
    input_a_fp8 = torch.randint(
        -8, 8, (TILE, TILE * K_TILES), dtype=torch.int8
    ).to(torch.float8_e4m3fn)
    input_b_fp8 = torch.randint(
        -8, 8, (TILE * K_TILES, TILE), dtype=torch.int8
    ).to(torch.float8_e4m3fn)

    a_q = input_a_fp8.to(torch.float32).numpy()
    b_q = input_b_fp8.to(torch.float32).numpy()

    a_k0 = a_q[:, :TILE]
    a_k1 = a_q[:, TILE:]
    b_k0 = b_q[:TILE, :]
    b_k1 = b_q[TILE:, :]

    a_t = torch.from_numpy(a_q)
    w_t = torch.from_numpy(b_q.T.copy())  # (TILE, K) for F.linear convention
    c_ref = ipt_bf16(a_t, w_t, scale_exp=0).numpy().astype(np.float32)

    a_k0_beats = pack_words_into_beats(matrix_to_fp8_words(a_k0), WORDS_PER_BEAT)
    a_k1_beats = pack_words_into_beats(matrix_to_fp8_words(a_k1), WORDS_PER_BEAT)
    b_k0_beats = pack_words_into_beats(matrix_to_fp8_words(b_k0), WORDS_PER_BEAT)
    b_k1_beats = pack_words_into_beats(matrix_to_fp8_words(b_k1), WORDS_PER_BEAT)

    c_bank2_words, c_bank3_words = [], []
    for r in range(TILE):
        c_bank2_words.extend(matrix_to_bf16_words(c_ref[r, 0:16].reshape(1, -1)))
        c_bank3_words.extend(matrix_to_bf16_words(c_ref[r, 16:32].reshape(1, -1)))
    c_bank2_beats = pack_words_into_beats(c_bank2_words, WORDS_PER_BEAT)
    c_bank3_beats = pack_words_into_beats(c_bank3_words, WORDS_PER_BEAT)

    preloads = []
    for i, beat in enumerate(a_k0_beats):
        preloads.append({"word_offset": A_K0_BASE + i, "data": f"0x{beat:X}"})
    for i, beat in enumerate(a_k1_beats):
        preloads.append({"word_offset": A_K1_BASE + i, "data": f"0x{beat:X}"})
    for i, beat in enumerate(b_k0_beats):
        preloads.append({"word_offset": B_K0_BASE + i, "data": f"0x{beat:X}"})
    for i, beat in enumerate(b_k1_beats):
        preloads.append({"word_offset": B_K1_BASE + i, "data": f"0x{beat:X}"})

    checks = []
    for i, beat in enumerate(c_bank2_beats):
        checks.append({"word_offset": C_BANK2_BASE + i, "expected": f"0x{beat:X}"})
    for i, beat in enumerate(c_bank3_beats):
        checks.append({"word_offset": C_BANK3_BASE + i, "expected": f"0x{beat:X}"})

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

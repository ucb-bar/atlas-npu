#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_matmul_mxu1.S` (64x32x64).

MXU1 (IPT / inner product tree) sibling of gen_smolvla_parameterized_matmul.py.
Kernel math is identical — C = A @ B with FP8 inputs and BF16 output — but
MXU1 rounds differently from MXU0 (aligned accumulation + single rounding
at the end, vs MXU0's per-PE rounding). So we route the golden through
software_models.mxu1_ipt.IPTLinearRTLFunction to get bit-exact agreement
with MXU1 RTL.

DRAM layout matches the MXU0 sibling so we can reuse the .S layout
constants and VCS test harness.
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
TIMEOUT = 400000

M = 64
K = 32
N = 64
M_TILES = M // TILE
K_TILES = K // TILE
N_TILES = N // TILE

BEATS_PER_FP8_TILE = TILE
BEATS_PER_BF16_HALF = TILE

A_BASE = 0
B_BASE = 128
OUT_BASE = 256


def main():
    ipt_bf16 = IPTLinearRTLFunction(
        vec_len=TILE, num_lanes=TILE, pipeline_depth=1,
        out_fmt_sel=OutputFmtSel.OutBF16,
    )

    torch.manual_seed(7)
    input_a_fp8 = torch.randint(-8, 8, (M, K), dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    input_b_fp8 = torch.randint(-8, 8, (K, N), dtype=torch.int8).to(
        torch.float8_e4m3fn
    )

    a_q = input_a_fp8.to(torch.float32).numpy()
    b_q = input_b_fp8.to(torch.float32).numpy()

    preloads: list = []

    a_offset = A_BASE
    for m in range(M_TILES):
        a_tile = a_q[m * TILE:(m + 1) * TILE, 0:TILE]
        tile_beats = pack_words_into_beats(
            matrix_to_fp8_words(a_tile), WORDS_PER_BEAT
        )
        for i, beat in enumerate(tile_beats):
            preloads.append({
                "word_offset": a_offset + i,
                "data": f"0x{beat:X}",
            })
        a_offset += BEATS_PER_FP8_TILE

    b_offset = B_BASE
    for n in range(N_TILES):
        b_tile = b_q[0:TILE, n * TILE:(n + 1) * TILE]
        tile_beats = pack_words_into_beats(
            matrix_to_fp8_words(b_tile), WORDS_PER_BEAT
        )
        for i, beat in enumerate(tile_beats):
            preloads.append({
                "word_offset": b_offset + i,
                "data": f"0x{beat:X}",
            })
        b_offset += BEATS_PER_FP8_TILE

    checks: list = []
    c_offset = OUT_BASE
    for m in range(M_TILES):
        for n in range(N_TILES):
            a_tile = a_q[m * TILE:(m + 1) * TILE, 0:TILE]
            b_tile = b_q[0:TILE, n * TILE:(n + 1) * TILE]
            # IPT (like SA) computes y = x @ w^T; pass w = B_tile^T for A @ B.
            a_t = torch.from_numpy(a_tile)
            w_t = torch.from_numpy(b_tile.T.copy())
            c_tile = ipt_bf16(a_t, w_t, scale_exp=0).numpy().astype(np.float32)

            low_words, high_words = [], []
            for r in range(TILE):
                low_words.extend(
                    matrix_to_bf16_words(c_tile[r, 0:16].reshape(1, -1))
                )
                high_words.extend(
                    matrix_to_bf16_words(c_tile[r, 16:32].reshape(1, -1))
                )
            low_beats = pack_words_into_beats(low_words, WORDS_PER_BEAT)
            high_beats = pack_words_into_beats(high_words, WORDS_PER_BEAT)

            for i, beat in enumerate(low_beats):
                checks.append({
                    "word_offset": c_offset + i,
                    "expected": f"0x{beat:X}",
                })
            c_offset += BEATS_PER_BF16_HALF

            for i, beat in enumerate(high_beats):
                checks.append({
                    "word_offset": c_offset + i,
                    "expected": f"0x{beat:X}",
                })
            c_offset += BEATS_PER_BF16_HALF

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

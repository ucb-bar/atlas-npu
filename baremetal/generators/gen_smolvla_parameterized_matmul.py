#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_matmul.S` (dual-MXU, 64x32x64).

Port of jrmills20 parameterized_matmul.py, column-split across engines:
  n_tile == 0 → MXU0 (SA, SARTLLinearFunction)
  n_tile == 1 → MXU1 (IPT, IPTLinearRTLFunction)

Shape M=64, K=32, N=64 yields 4 output tiles (M_tiles=2, N_tiles=2) with a
single K-tile each. Golden is routed per-tile through the engine that the
.S dispatches to, so the BF16 outputs match each engine's RTL bit-exactly
(SA rounds per-PE; IPT rounds once at the aligned-tree output).

DRAM layout (tile-major, matches .S header):
  [0x0000..0x07FF]  A FP8 (2 tiles x 1024B)
  [0x1000..0x17FF]  B FP8 (2 tiles x 1024B)
  [0x2000..0x3FFF]  C BF16 (4 tiles x 2048B; per tile: low half + high half)
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
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction


TILE = 32
WORDS_PER_BEAT = 8
TIMEOUT = 400000

M = 64
K = 32
N = 64
M_TILES = M // TILE      # 2
K_TILES = K // TILE      # 1
N_TILES = N // TILE      # 2

# word_offset units (1 beat = 32 bytes).
BEATS_PER_FP8_TILE = TILE              # 32x32 FP8 = 1024 B = 32 beats
BEATS_PER_BF16_HALF = TILE             # 32x16 BF16 = 1024 B = 32 beats
BEATS_PER_BF16_TILE = 2 * BEATS_PER_BF16_HALF

A_BASE = 0                              # DRAM 0x0000
B_BASE = 128                            # DRAM 0x1000 (128 * 32 = 0x1000)
OUT_BASE = 256                          # DRAM 0x2000


def main():
    sa_bf16 = SARTLLinearFunction(
        rows=TILE, cols=TILE, out_fmt_sel=OutputFmtSel.OutBF16
    )
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

    a_q = input_a_fp8.to(torch.float32).numpy()    # (M, K)
    b_q = input_b_fp8.to(torch.float32).numpy()    # (K, N)

    preloads: list = []

    # ── A preloads: tile-major (m, k=0) ──
    a_offset = A_BASE
    for m in range(M_TILES):
        a_tile = a_q[m * TILE:(m + 1) * TILE, 0:TILE]   # (32, 32) FP32
        tile_beats = pack_words_into_beats(
            matrix_to_fp8_words(a_tile), WORDS_PER_BEAT
        )
        for i, beat in enumerate(tile_beats):
            preloads.append({
                "word_offset": a_offset + i,
                "data": f"0x{beat:X}",
            })
        a_offset += BEATS_PER_FP8_TILE

    # ── B preloads: tile-major (k=0, n) ──
    b_offset = B_BASE
    for n in range(N_TILES):
        b_tile = b_q[0:TILE, n * TILE:(n + 1) * TILE]   # (32, 32) FP32
        tile_beats = pack_words_into_beats(
            matrix_to_fp8_words(b_tile), WORDS_PER_BEAT
        )
        for i, beat in enumerate(tile_beats):
            preloads.append({
                "word_offset": b_offset + i,
                "data": f"0x{beat:X}",
            })
        b_offset += BEATS_PER_FP8_TILE

    # ── Golden: per-output-tile matmul routed through the engine the .S uses ──
    # .S dispatches n_tile==0 to MXU0 (SA) and n_tile==1 to MXU1 (IPT).
    checks: list = []
    c_offset = OUT_BASE
    for m in range(M_TILES):
        for n in range(N_TILES):
            a_tile = a_q[m * TILE:(m + 1) * TILE, 0:TILE]
            b_tile = b_q[0:TILE, n * TILE:(n + 1) * TILE]
            # Both engines compute y = x @ w^T, so pass w = B_tile^T.
            a_t = torch.from_numpy(a_tile)
            w_t = torch.from_numpy(b_tile.T.copy())
            if n == 0:
                c_tile = sa_bf16(a_t, w_t, scale_exp=0).numpy().astype(np.float32)
            else:
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

#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_fused_matmul_bias.S` (64x32x64).

MXU0 (systolic array) path port of jrmills20 parameterized_fused_matmul_bias.py,
using the stock quantize-roundtrip pattern:

  out[m,n] = (A[m,0] @ B[0,n]) + bias[m,n]

  - A BF16 quantized on-chip to FP8 via VMATPUSH.ACC.BF16 + VMATPOP.FP8
    (golden side: bf16_tile_to_e4m3_bytes)
  - B pre-quantized FP8 in DRAM (same converter)
  - FP8 matmul via SARTLLinearFunction (MXU0 software model)
  - Bias added via run_binary_rows("add", ...) to match VADD.BF16 exactly
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data
from mxu_fp8_utils import bf16_tile_to_e4m3_bytes
from vpu_gen_utils import (
    BF16_PER_BEAT,
    FP8_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    pack_u8_le,
    run_binary_rows,
    tensor_checks,
    tensor_preloads,
)
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


TILE = ROWS_PER_REGISTER                 # 32
BEATS_PER_BF16_TILE = ROWS_PER_TENSOR    # 64 (2 banks)
BEATS_PER_FP8_SLOT = ROWS_PER_TENSOR     # 64-beat slot (32 data + 32 pad) per stock layout
TIMEOUT = 500000

M = 64
K = 32
N = 64
M_TILES = M // TILE    # 2
K_TILES = K // TILE    # 1
N_TILES = N // TILE    # 2
TOTAL_OUT_TILES = M_TILES * N_TILES  # 4

# Word offsets (1 beat = 32 bytes).
A_BASE = 0                                          # 0x0000
B_BASE = A_BASE + M_TILES * BEATS_PER_BF16_TILE     # 128 → 0x1000
BIAS_BASE = B_BASE + N_TILES * BEATS_PER_FP8_SLOT   # 256 → 0x2000
OUT_BASE = BIAS_BASE + TOTAL_OUT_TILES * BEATS_PER_BF16_TILE  # 512 → 0x4000


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> list[list[int]]:
    if tile.shape != (TILE, 2 * BF16_PER_BEAT):
        raise ValueError(
            f"expected ({TILE}, {2 * BF16_PER_BEAT}) bf16 tile, got {tuple(tile.shape)}"
        )
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bank0 = tile[:, :BF16_PER_BEAT].contiguous().view(torch.int16)
    bank1 = tile[:, BF16_PER_BEAT:].contiguous().view(torch.int16)
    rows: list[list[int]] = []
    for half in (bank0, bank1):
        for i in range(TILE):
            rows.append([int(v) & 0xFFFF for v in half[i].tolist()])
    return rows


def fp8_tile_to_beats(tile_uint8: torch.Tensor) -> list[list[int]]:
    """(32, 32) uint8 FP8 → 32 row-beats (one row per beat)."""
    if tile_uint8.shape != (TILE, FP8_PER_BEAT):
        raise ValueError(
            f"expected ({TILE}, {FP8_PER_BEAT}) uint8 tile, got {tuple(tile_uint8.shape)}"
        )
    arr = tile_uint8.numpy()
    return [[int(v) & 0xFF for v in arr[i].tolist()] for i in range(TILE)]


def main():
    torch.manual_seed(42)
    A = (torch.randn(M, K) * 0.5).to(torch.bfloat16)
    B = (torch.randn(K, N) * 0.5).to(torch.bfloat16)
    BIAS = torch.randn(M, N, dtype=torch.bfloat16)

    sa_bf16 = SARTLLinearFunction(
        rows=TILE, cols=TILE, out_fmt_sel=OutputFmtSel.OutBF16
    )

    preloads: list = []
    checks: list = []

    # ── A BF16 preloads (m-major, single k=0 tile) ─────────────────────
    a_offset = A_BASE
    for m in range(M_TILES):
        a_tile_bf16 = A[m * TILE:(m + 1) * TILE, 0:TILE]
        preloads.extend(tensor_preloads(a_offset, bf16_tile_to_bank_rows(a_tile_bf16)))
        a_offset += BEATS_PER_BF16_TILE

    # ── B FP8 preloads (n-major, single k=0 tile; 32 data beats per tile
    #    with 32 slot-pad beats skipped) ──────────────────────────────
    b_offset = B_BASE
    b_tiles_fp8: list[torch.Tensor] = []   # keep for golden
    for n in range(N_TILES):
        b_tile_bf16 = B[0:TILE, n * TILE:(n + 1) * TILE]
        b_tile_fp8 = bf16_tile_to_e4m3_bytes(b_tile_bf16, scale_exp=0)
        b_tiles_fp8.append(b_tile_fp8)
        b_beats = fp8_tile_to_beats(b_tile_fp8)
        for i, beat_lanes in enumerate(b_beats):
            preloads.append({
                "word_offset": b_offset + i,
                "data": pack_u8_le(beat_lanes),
            })
        b_offset += BEATS_PER_FP8_SLOT

    # ── bias BF16 preloads (m,n tile-major) ────────────────────────────
    bias_offset = BIAS_BASE
    for m in range(M_TILES):
        for n in range(N_TILES):
            bias_tile = BIAS[m * TILE:(m + 1) * TILE, n * TILE:(n + 1) * TILE]
            preloads.extend(tensor_preloads(
                bias_offset, bf16_tile_to_bank_rows(bias_tile)
            ))
            bias_offset += BEATS_PER_BF16_TILE

    # ── Per-output-tile golden through SA + bias add ───────────────────
    out_offset = OUT_BASE
    for m in range(M_TILES):
        a_tile_bf16 = A[m * TILE:(m + 1) * TILE, 0:TILE]
        a_tile_fp8 = bf16_tile_to_e4m3_bytes(a_tile_bf16, scale_exp=0)
        for n in range(N_TILES):
            b_tile_fp8 = b_tiles_fp8[n]
            bias_tile = BIAS[m * TILE:(m + 1) * TILE, n * TILE:(n + 1) * TILE]

            # SA: y = x @ w^T; pass w = B^T for y = A @ B.
            w_in = torch.from_numpy(b_tile_fp8.numpy().T.copy())
            mat_bf16 = sa_bf16(a_tile_fp8, w_in, scale_exp=0).to(torch.bfloat16)

            mat_rows = bf16_tile_to_bank_rows(mat_bf16)
            bias_rows = bf16_tile_to_bank_rows(bias_tile)
            out_rows = run_binary_rows("add", mat_rows, bias_rows)

            checks.extend(tensor_checks(out_offset, out_rows))
            out_offset += BEATS_PER_BF16_TILE

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

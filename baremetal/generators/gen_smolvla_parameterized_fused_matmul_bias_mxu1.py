#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_fused_matmul_bias_mxu1.S`.

MXU1 (IPT) sibling of gen_smolvla_parameterized_fused_matmul_bias.py. The
only substantive change from the MXU0 version is routing the matmul golden
through IPTLinearRTLFunction instead of SARTLLinearFunction — the quantize
roundtrip (bf16_tile_to_e4m3_bytes) and bias add (run_binary_rows("add",…))
are MXU-independent per Codex's confirmation that dequant-on-push and
quant-on-pop sequencer logic is identical across engines.
"""

import os
import sys

import numpy as np
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
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel, decode_e4m3


TILE = ROWS_PER_REGISTER
BEATS_PER_BF16_TILE = ROWS_PER_TENSOR
BEATS_PER_FP8_SLOT = ROWS_PER_TENSOR
TIMEOUT = 500000

M = 64
K = 32
N = 64
M_TILES = M // TILE
K_TILES = K // TILE
N_TILES = N // TILE
TOTAL_OUT_TILES = M_TILES * N_TILES

A_BASE = 0
B_BASE = A_BASE + M_TILES * BEATS_PER_BF16_TILE
BIAS_BASE = B_BASE + N_TILES * BEATS_PER_FP8_SLOT
OUT_BASE = BIAS_BASE + TOTAL_OUT_TILES * BEATS_PER_BF16_TILE


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
    if tile_uint8.shape != (TILE, FP8_PER_BEAT):
        raise ValueError(
            f"expected ({TILE}, {FP8_PER_BEAT}) uint8 tile, got {tuple(tile_uint8.shape)}"
        )
    arr = tile_uint8.numpy()
    return [[int(v) & 0xFF for v in arr[i].tolist()] for i in range(TILE)]


def fp8_uint8_to_fp32(tile_uint8: torch.Tensor) -> torch.Tensor:
    """Decode E4M3 uint8 bytes → FP32 via software_models.mxu1_ipt.fp_formats.

    IPTLinearRTLFunction expects float-typed inputs (unlike SARTLLinearFunction
    which accepts uint8). Using software_models' decode_e4m3 keeps the entire
    MXU1 golden path inside software_models rather than routing through
    torch.float8_e4m3fn's native conversion.
    """
    arr = tile_uint8.numpy().astype(np.uint8)
    fp32 = np.empty(arr.shape, dtype=np.float32)
    it = np.nditer(arr, flags=["multi_index"])
    for x in it:
        fp32[it.multi_index] = decode_e4m3(int(x)).value
    return torch.from_numpy(fp32)


def main():
    torch.manual_seed(42)
    A = (torch.randn(M, K) * 0.5).to(torch.bfloat16)
    B = (torch.randn(K, N) * 0.5).to(torch.bfloat16)
    BIAS = torch.randn(M, N, dtype=torch.bfloat16)

    ipt_bf16 = IPTLinearRTLFunction(
        vec_len=TILE, num_lanes=TILE, pipeline_depth=1,
        out_fmt_sel=OutputFmtSel.OutBF16,
    )

    preloads: list = []
    checks: list = []

    a_offset = A_BASE
    for m in range(M_TILES):
        a_tile = A[m * TILE:(m + 1) * TILE, 0:TILE]
        preloads.extend(tensor_preloads(a_offset, bf16_tile_to_bank_rows(a_tile)))
        a_offset += BEATS_PER_BF16_TILE

    b_offset = B_BASE
    b_tiles_fp8: list[torch.Tensor] = []
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

    bias_offset = BIAS_BASE
    for m in range(M_TILES):
        for n in range(N_TILES):
            bias_tile = BIAS[m * TILE:(m + 1) * TILE, n * TILE:(n + 1) * TILE]
            preloads.extend(tensor_preloads(
                bias_offset, bf16_tile_to_bank_rows(bias_tile)
            ))
            bias_offset += BEATS_PER_BF16_TILE

    out_offset = OUT_BASE
    for m in range(M_TILES):
        a_tile_bf16 = A[m * TILE:(m + 1) * TILE, 0:TILE]
        a_tile_fp8 = bf16_tile_to_e4m3_bytes(a_tile_bf16, scale_exp=0)
        for n in range(N_TILES):
            b_tile_fp8 = b_tiles_fp8[n]
            bias_tile = BIAS[m * TILE:(m + 1) * TILE, n * TILE:(n + 1) * TILE]

            a_fp32 = fp8_uint8_to_fp32(a_tile_fp8)
            b_fp32 = fp8_uint8_to_fp32(b_tile_fp8)
            w_in = torch.from_numpy(b_fp32.numpy().T.copy())
            mat_bf16 = ipt_bf16(a_fp32, w_in, scale_exp=0).to(torch.bfloat16)

            mat_rows = bf16_tile_to_bank_rows(mat_bf16)
            bias_rows = bf16_tile_to_bank_rows(bias_tile)
            out_rows = run_binary_rows("add", mat_rows, bias_rows)

            checks.extend(tensor_checks(out_offset, out_rows))
            out_offset += BEATS_PER_BF16_TILE

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate test vectors for `smolvla_fused_matmul_bias_mxu1.S`.

Implements the tracker contract `out = (A_bf16 @ B_fp8) + bias_bf16`
with on-chip BF16->FP8 quantization of the activation A. Matches the
atlas `smolvla_fused_attention.S` Q-tile path for the quant
roundtrip (`VMATPUSH.ACC.BF16.MXU1` + `SELI 0,0x7F` +
`VMATPOP.FP8.MXU1`). Block scale is unit (`scale_exp=0`); per-block
scaling is left as a separate port.

Atlas MXU computes `A @ W^T` natively, so B is `VTRPOSE.XLU`-d to B^T
before being pushed as the weight; the matmul then yields `A @ B`.

Golden mirrors the hardware:
  - `bf16_tile_to_e4m3_bytes` (from `mxu_fp8_utils`) reproduces the
    `VMATPUSH.ACC.BF16 + VMATPOP.FP8` quantization bit-for-bit.
  - `IPTLinearRTLFunction` (with FP32-decoded E4M3 inputs) handles the
    FP8 matmul + accumulator BF16 narrowing. IPT differs from SA here:
    SA accepts uint8 bytes directly, IPT does not, so we decode the
    same E4M3 bytes back to FP32 via `decode_e4m3`.
  - `run_binary_rows("add", ...)` adds the bias through atlas's
    `VectorEngineModel` so the VADD.BF16 result matches RTL.
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
    pack_u16_le,
    pack_u8_le,
    run_binary_rows,
    tensor_checks,
    tensor_preloads,
)
from software_models.mxu1_ipt.ipt_rtl_linear import (
    IPTLinearRTLFunction,
    e4m3_bytes_to_float,
)
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


TILE = ROWS_PER_REGISTER          # 32
BEATS_PER_TENSOR = ROWS_PER_TENSOR  # 64 BF16 beats per [32,32] tile
TIMEOUT = 30000

A_BASE = 0
B_BASE = A_BASE + BEATS_PER_TENSOR        # FP8 tile uses 32 beats but we
                                          # leave 64-beat slot for layout
                                          # symmetry with BF16 tiles
BIAS_BASE = B_BASE + BEATS_PER_TENSOR
OUT_BASE = BIAS_BASE + BEATS_PER_TENSOR


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> list[list[int]]:
    if tile.shape != (TILE, 2 * BF16_PER_BEAT):
        raise ValueError(
            f"expected ({TILE}, {2 * BF16_PER_BEAT}) bf16 tile, "
            f"got {tuple(tile.shape)}"
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
    """Lay a (32, 32) uint8 FP8 tile out as 32 row-beats (one row per beat)."""
    if tile_uint8.shape != (TILE, FP8_PER_BEAT):
        raise ValueError(
            f"expected ({TILE}, {FP8_PER_BEAT}) uint8 tile, "
            f"got {tuple(tile_uint8.shape)}"
        )
    arr = tile_uint8.numpy()
    return [[int(v) & 0xFF for v in arr[i].tolist()] for i in range(TILE)]


def main():
    torch.manual_seed(42)
    A_RAW = (torch.randn(TILE, TILE) * 0.5).to(torch.bfloat16)
    B_RAW = (torch.randn(TILE, TILE) * 0.5).to(torch.bfloat16)
    BIAS_RAW = torch.randn(TILE, TILE, dtype=torch.bfloat16)

    # B is pre-quantized to FP8 in DRAM (host-side quant matches the
    # hardware path used for A on-chip).
    B_FP8 = bf16_tile_to_e4m3_bytes(B_RAW, scale_exp=0)

    # Golden: also quantize A through the same converter (mirrors what
    # VMATPUSH.ACC.BF16 + VMATPOP.FP8 does on-chip).
    A_FP8 = bf16_tile_to_e4m3_bytes(A_RAW, scale_exp=0)

    ipt_bf16 = IPTLinearRTLFunction(
        vec_len=TILE, num_lanes=TILE, pipeline_depth=1, out_fmt_sel=OutputFmtSel.OutBF16
    )
    # IPTLinearRTLFunction computes y = x @ w^T. For A @ B, pass w = B^T.
    # IPT requires FP32 inputs (SA accepts uint8 directly, IPT does not),
    # so decode the same E4M3 bytes that feed DRAM back to FP32.
    a_fp32 = e4m3_bytes_to_float(A_FP8)
    w_fp32 = e4m3_bytes_to_float(B_FP8.T.contiguous())
    mat_bf16 = ipt_bf16(a_fp32, w_fp32, scale_exp=0).to(torch.bfloat16)

    mat_rows = bf16_tile_to_bank_rows(mat_bf16)
    bias_rows = bf16_tile_to_bank_rows(BIAS_RAW)
    out_rows = run_binary_rows("add", mat_rows, bias_rows)

    a_rows = bf16_tile_to_bank_rows(A_RAW)
    b_beats = fp8_tile_to_beats(B_FP8)

    preloads = []
    preloads.extend(tensor_preloads(A_BASE, a_rows))
    for i, beat_lanes in enumerate(b_beats):
        preloads.append({"word_offset": B_BASE + i, "data": pack_u8_le(beat_lanes)})
    preloads.extend(tensor_preloads(BIAS_BASE, bias_rows))

    checks = tensor_checks(OUT_BASE, out_rows)

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

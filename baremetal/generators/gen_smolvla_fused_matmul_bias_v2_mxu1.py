#!/usr/bin/env python3
"""Generate test vectors for `smolvla_fused_matmul_bias_v2_mxu1.S`.

Alternative interpretation matching the npu-model reference file: A and
B are BOTH pre-quantized FP8 tensors stored in DRAM, no on-chip
quantization. Matches the npu-model smolvla_fused_matmul_bias sketch and the existing
`smolvla_matmul_mxu0.S` activation path. Simpler than V1 but does NOT
exercise the BF16->FP8 hardware quant roundtrip.

Use this if V1 fails or if the tracker contract turns out to mean
"both inputs already FP8" instead of "BF16 act + on-chip quant".

Byte-exactness: the same `torch.float8_e4m3fn` tensor is `view`ed as
uint8 for the DRAM preloads; the IPT golden decodes those same bytes
back to FP32 via `mxu1_ipt.fp_formats.decode_e4m3` (IPT requires
FP32 inputs, unlike SA which accepts uint8 directly). The golden
therefore sees exactly the bytes the assembly will load.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data
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
from software_models.mxu1_ipt.ipt_rtl_linear import (
    IPTLinearRTLFunction,
    e4m3_bytes_to_float,
)
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


TILE = ROWS_PER_REGISTER          # 32
BEATS_PER_TENSOR = ROWS_PER_TENSOR  # 64 BF16 beats per [32,32] tile
TIMEOUT = 30000

A_BASE = 0
B_BASE = A_BASE + TILE                         # FP8 tile = 32 beats
BIAS_BASE = B_BASE + TILE                      # BF16 tile follows FP8
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
    a_fp8 = torch.randint(-8, 8, (TILE, TILE), dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    b_fp8 = torch.randint(-8, 8, (TILE, TILE), dtype=torch.int8).to(
        torch.float8_e4m3fn
    )
    BIAS_RAW = torch.randn(TILE, TILE, dtype=torch.bfloat16)

    # Raw E4M3 bytes — the same bytes feed DRAM preloads and the IPT
    # golden (after decode_e4m3 below), so the model sees exactly the
    # bytes the DUT will load from DRAM.
    a_bytes = a_fp8.view(torch.uint8).contiguous()
    b_bytes = b_fp8.view(torch.uint8).contiguous()

    ipt_bf16 = IPTLinearRTLFunction(
        vec_len=TILE, num_lanes=TILE, pipeline_depth=1, out_fmt_sel=OutputFmtSel.OutBF16
    )
    # IPT computes y = x @ w^T. For A @ B, pass w = B^T.
    # IPT requires FP32 inputs, so decode the same E4M3 bytes that feed
    # DRAM back to FP32 via e4m3_bytes_to_float.
    a_fp32 = e4m3_bytes_to_float(a_bytes)
    w_fp32 = e4m3_bytes_to_float(b_bytes.T.contiguous())
    mat_bf16 = ipt_bf16(a_fp32, w_fp32, scale_exp=0).to(torch.bfloat16)

    mat_rows = bf16_tile_to_bank_rows(mat_bf16)
    bias_rows = bf16_tile_to_bank_rows(BIAS_RAW)
    out_rows = run_binary_rows("add", mat_rows, bias_rows)

    a_beats = fp8_tile_to_beats(a_bytes)
    b_beats = fp8_tile_to_beats(b_bytes)

    preloads = []
    for i, beat_lanes in enumerate(a_beats):
        preloads.append({"word_offset": A_BASE + i, "data": pack_u8_le(beat_lanes)})
    for i, beat_lanes in enumerate(b_beats):
        preloads.append({"word_offset": B_BASE + i, "data": pack_u8_le(beat_lanes)})
    preloads.extend(tensor_preloads(BIAS_BASE, bias_rows))

    checks = tensor_checks(OUT_BASE, out_rows)

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

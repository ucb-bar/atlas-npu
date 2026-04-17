#!/usr/bin/env python3
"""Generate test vectors for `smolvla_attention.S`.

Port of npu-model/npu_model/configs/programs/smolvla_attention.py.
Single-tile scaled-dot-product attention:

    scores = Q @ K
    probs  = softmax(scores * scale)
    packed = VFP8PACK(probs)      (atlas 2→1 phased layout, not row-preserving)
    out    = packed @ V

Q, K, V are FP8 32x32 tiles; scale is a constant 1/sqrt(32) BF16 32x32.
Seed 49 matches npu-model.

Atlas's VFP8PACK does NOT produce npu-model's row-preserving
`cat([low, high], dim=1)` layout — it streams rows 0..31 of the low
bank then 0..31 of the high bank through a 2-row pack lane box, so
packed row i = concat(fp8(src[2i]), fp8(src[2i+1])). The second matmul
therefore does NOT compute probs@V in the PyTorch attention sense. We
still go through atlas's SARTLLinearFunction + VectorEngineModel for
the golden, so RTL matches model bit-exactly.
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (
    matrix_to_fp8_words,
    pack_words_into_beats,
    emit_test_data,
)
from vpu_gen_utils import (
    BF16_PER_BEAT,
    FP8_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    run_binary_rows,
    run_fp8_pack_rows,
    run_row_reduce_tensor,
    run_unary_rows,
    tensor_checks,
    tensor_preloads,
)
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


TILE = 32
WORDS_PER_BEAT = 8
TIMEOUT = 80000
SCALE_E8M0_UNIT = 0x7F  # biased exponent 127 → 2^0 = 1.0

BEATS_FP8 = TILE                         # one FP8 tile = 32 beats
BEATS_BF16_BANK = ROWS_PER_REGISTER      # one BF16 bank = 32 beats

Q_BASE           = 0
K_BASE           = Q_BASE + BEATS_FP8
V_BASE           = K_BASE + BEATS_FP8
SCALE_BANK0_BASE = V_BASE + BEATS_FP8
SCALE_BANK1_BASE = SCALE_BANK0_BASE + BEATS_BF16_BANK
OUT_BANK0_BASE   = SCALE_BANK1_BASE + BEATS_BF16_BANK
OUT_BANK1_BASE   = OUT_BANK0_BASE + BEATS_BF16_BANK


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> tuple[list[list[int]], list[list[int]]]:
    if tile.shape != (ROWS_PER_REGISTER, 2 * BF16_PER_BEAT):
        raise ValueError(
            f"expected ({ROWS_PER_REGISTER}, {2 * BF16_PER_BEAT}) bf16 tile, "
            f"got {tuple(tile.shape)}"
        )
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bank0 = tile[:, :BF16_PER_BEAT].contiguous().view(torch.int16)
    bank1 = tile[:, BF16_PER_BEAT:].contiguous().view(torch.int16)
    low = [[int(v) & 0xFFFF for v in bank0[i].tolist()] for i in range(ROWS_PER_REGISTER)]
    high = [[int(v) & 0xFFFF for v in bank1[i].tolist()] for i in range(ROWS_PER_REGISTER)]
    return low, high


def packed_rows_to_fp8_bytes(packed_rows: list[list[int]]) -> torch.Tensor:
    """Reassemble VFP8PACK output (32 rows × 16 UInt16 slots) into raw
    (32, 32) FP8 E4M3 bytes.

    Feed these bytes directly to SARTLLinearFunction so the golden
    uses the same already-quantized activations that hardware feeds
    into the second matmul, instead of going through the float32
    quantization shim again.
    """
    if len(packed_rows) != TILE:
        raise ValueError(f"expected {TILE} packed rows, got {len(packed_rows)}")
    bytes_u8 = np.zeros((TILE, TILE), dtype=np.uint8)
    for i, row in enumerate(packed_rows):
        if len(row) != BF16_PER_BEAT:
            raise ValueError(
                f"packed row {i} must have {BF16_PER_BEAT} slots, got {len(row)}"
            )
        for j in range(BF16_PER_BEAT):
            slot = int(row[j]) & 0xFFFF
            bytes_u8[i, 2 * j]     = slot & 0xFF
            bytes_u8[i, 2 * j + 1] = (slot >> 8) & 0xFF
    return torch.from_numpy(bytes_u8.copy())


def main():
    sa_bf16 = SARTLLinearFunction(
        rows=TILE, cols=TILE, out_fmt_sel=OutputFmtSel.OutBF16
    )

    torch.manual_seed(49)
    Q = torch.randint(-4, 4, (TILE, TILE), dtype=torch.int8).to(torch.float8_e4m3fn)
    K = torch.randint(-4, 4, (TILE, TILE), dtype=torch.int8).to(torch.float8_e4m3fn)
    V = torch.randint(-4, 4, (TILE, TILE), dtype=torch.int8).to(torch.float8_e4m3fn)
    scale_tile = torch.full(
        (TILE, TILE), 1.0 / math.sqrt(float(TILE)), dtype=torch.bfloat16
    )

    q_f32 = Q.to(torch.float32).numpy()
    k_f32 = K.to(torch.float32).numpy()
    v_f32 = V.to(torch.float32).numpy()
    q_u8 = Q.view(torch.uint8)
    k_u8_t = K.view(torch.uint8).T.contiguous()
    v_u8_t = V.view(torch.uint8).T.contiguous()

    # scores = Q @ K.  Atlas MXU does A @ W^T, so push K^T as weight:
    # sa_bf16(Q, K.T) computes Q @ K.
    scores_bf16 = sa_bf16(q_u8, k_u8_t, scale_exp=0).to(torch.bfloat16)

    scores_low, scores_high = bf16_tile_to_bank_rows(scores_bf16)
    scale_low, scale_high = bf16_tile_to_bank_rows(scale_tile)

    scores_rows = scores_low + scores_high
    scale_rows = scale_low + scale_high
    scaled_rows = run_binary_rows("mul", scores_rows, scale_rows)
    rowmax_rows = run_row_reduce_tensor("rmax", scaled_rows)
    shifted_rows = run_binary_rows("sub", scaled_rows, rowmax_rows)
    expd_rows = run_unary_rows("exp", shifted_rows)
    rowsum_rows = run_row_reduce_tensor("rsum", expd_rows)
    inv_sum_rows = run_unary_rows("rcp", rowsum_rows)
    probs_rows = run_binary_rows("mul", expd_rows, inv_sum_rows)

    # VFP8PACK streams low bank then high bank through the 2→1 phased
    # lane box, so packed[i] pairs (2i, 2i+1) of the stream.
    packed_rows = run_fp8_pack_rows(
        probs_rows[:ROWS_PER_REGISTER],
        probs_rows[ROWS_PER_REGISTER:],
        SCALE_E8M0_UNIT,
    )
    packed_u8 = packed_rows_to_fp8_bytes(packed_rows)

    # out = packed @ V.  Again push V^T so MXU computes packed @ V.
    out_bf16 = sa_bf16(packed_u8, v_u8_t, scale_exp=0).to(torch.bfloat16)

    out_low, out_high = bf16_tile_to_bank_rows(out_bf16)

    q_beats = pack_words_into_beats(matrix_to_fp8_words(q_f32), WORDS_PER_BEAT)
    k_beats = pack_words_into_beats(matrix_to_fp8_words(k_f32), WORDS_PER_BEAT)
    v_beats = pack_words_into_beats(matrix_to_fp8_words(v_f32), WORDS_PER_BEAT)

    preloads: list[dict] = []
    for i, beat in enumerate(q_beats):
        preloads.append({"word_offset": Q_BASE + i, "data": f"0x{beat:X}"})
    for i, beat in enumerate(k_beats):
        preloads.append({"word_offset": K_BASE + i, "data": f"0x{beat:X}"})
    for i, beat in enumerate(v_beats):
        preloads.append({"word_offset": V_BASE + i, "data": f"0x{beat:X}"})
    preloads.extend(tensor_preloads(SCALE_BANK0_BASE, scale_low))
    preloads.extend(tensor_preloads(SCALE_BANK1_BASE, scale_high))

    checks = list(tensor_checks(OUT_BANK0_BASE, out_low))
    checks.extend(tensor_checks(OUT_BANK1_BASE, out_high))

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

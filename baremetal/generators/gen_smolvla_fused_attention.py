#!/usr/bin/env python3
"""Generate test vectors for `smolvla_fused_attention.S`.

Port of npu-model/npu_model/configs/programs/smolvla_fused_attention.py.
Single-tile flash attention over 2 K-tiles (k_seq=64), head_dim=64:

    for each K tile:
        scores  = Q @ K^T              (MXU0, BF16 inputs quantized to FP8
                                         on-chip via acc roundtrip; K-chain
                                         accumulator over head_dim=64)
        scaled  = scores * scale       (VPU pair op)
        m_new   = max(m_prev, rowmax(scaled))
        exp_diff= exp(m_prev - m_new)
        O      *= exp_diff
        exp_s   = exp(scaled - m_new)
        exp_s_q = quantize(exp_s) BF16→FP8 (acc roundtrip)
        O      += exp_s_q @ V_left | V_right (two MXU passes)
        l       = exp_diff * l + rowsum(exp_s)
        m_prev  = m_new
    O /= l

Inputs use seed 42 + (randn * 0.5).to(bfloat16), matching npu-model.

Atlas's MXU computes A @ W^T natively. Q @ K^T pushes K as W (no XLU).
exp_s @ V_left pushes V_mlir[:32, :] as W (since V_left = V_mlir[:32, :]^T).

Golden mirrors atlas hardware:
  - SARTLLinearFunction handles the K-chain FP32 accumulator + final BF16
    truncation, and quantizes BF16-valued FP32 inputs to FP8 internally
    via `float_to_e4m3_bytes_c`. For BF16 inputs (whose float32 reps have
    zero trailing mantissa bits past bit 23-7=16), this RNE rounding to
    FP8-E4M3 produces the same bytes as VMATPUSH.ACC.BF16 + VMATPOP.FP8
    (`bf16_scale_to_e4m3` with scale_exp=0).
  - VPU pair ops, row-reduces, and unaries route through
    vpu_gen_utils.MODEL (atlas's VectorEngineModel) for bit-exact pipeline
    match. VMAX.BF16 → "pairmax".

DRAM beat layout (column-blocked [32, 16] BF16 blocks; 32 beats/block):
    Q_BASE   = 0   (Q   [32,64] = 4 blocks × 32 beats = 128 beats)
    K0_BASE  = 128 (K0  [32,64])
    K1_BASE  = 256 (K1  [32,64])
    V0_BASE  = 384 (V0_mlir [64,32] = 4 blocks: rows {0:32, 32:64} × cols {0:16, 16:32})
    V1_BASE  = 512 (V1_mlir [64,32])
    OUT_BASE = 768 (DRAM byte 0x6000 / 32-byte beat = beat 768)
"""

import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    constant_bf16_rows,
    float_to_bf16,
    pack_u16_le,
    run_binary_rows,
    run_row_reduce_tensor,
    run_unary_rows,
)
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.converters import bf16_scale_to_e4m3
from software_models.mxu1_ipt.fp_formats import OutputFmtSel


def bf16_tile_to_e4m3_bytes(tile: torch.Tensor) -> torch.Tensor:
    """Quantize BF16 tile to E4M3 uint8 via atlas's hardware-equivalent
    `bf16_scale_to_e4m3` (RNE to nearest normal, including subnormal-edge
    rounding). Mirrors VMATPUSH.ACC.BF16 + VMATPOP.FP8 with E8M0 scale=0x7F.
    `shim_float_to_e4m3` (used inside SARTLLinearFunction's float path) flushes
    sub-2^-6 magnitudes to zero, which diverges from the hardware on values
    like 0xBC79 (-0.0152) → hw=0x88, shim=0x00."""
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bits = tile.contiguous().view(torch.int16).numpy().astype(np.uint16)
    flat = bits.ravel()
    out = np.empty(flat.size, dtype=np.uint8)
    for i in range(flat.size):
        out[i] = bf16_scale_to_e4m3(int(flat[i]), 0) & 0xFF
    return torch.from_numpy(out.reshape(bits.shape))


Q_ROWS    = 32
HEAD_DIM  = 64
SCALE_VAL = 1.0 / math.sqrt(float(HEAD_DIM))   # 0.125 → 0x3E00 BF16
INIT_NEG  = -100.0                              # 0xC2C8 BF16
TIMEOUT   = 200000

BEATS_PER_BLOCK = ROWS_PER_REGISTER             # 32 beats per [32, 16] block
BLOCKS_PER_TILE = 4                             # [32,64] or [64,32] → 4 [32,16] blocks
BEATS_PER_TILE  = BLOCKS_PER_TILE * BEATS_PER_BLOCK  # 128 beats

Q_BASE   = 0
K0_BASE  = Q_BASE  + BEATS_PER_TILE
K1_BASE  = K0_BASE + BEATS_PER_TILE
V0_BASE  = K1_BASE + BEATS_PER_TILE
V1_BASE  = V0_BASE + BEATS_PER_TILE
OUT_BASE = 0x6000 // 32                         # DRAM 0x6000 → beat 768


def col_block_beats(tile: torch.Tensor) -> list[list[int]]:
    """Pack a 2D BF16 tile into atlas DRAM column-blocked layout. Each
    [32, 16] sub-block is one mreg's worth (32 beats × 16 BF16 lanes).
    Iteration: row_block (outer), col_block (inner)."""
    rows, cols = tile.shape
    if rows % ROWS_PER_REGISTER != 0 or cols % BF16_PER_BEAT != 0:
        raise ValueError(
            f"col_block_beats: shape ({rows},{cols}) not divisible by "
            f"({ROWS_PER_REGISTER},{BF16_PER_BEAT})"
        )
    beats: list[list[int]] = []
    for row_start in range(0, rows, ROWS_PER_REGISTER):
        for col_start in range(0, cols, BF16_PER_BEAT):
            block = tile[
                row_start:row_start + ROWS_PER_REGISTER,
                col_start:col_start + BF16_PER_BEAT,
            ].contiguous().view(torch.int16).numpy().astype(np.uint16)
            for i in range(ROWS_PER_REGISTER):
                beats.append([int(v) & 0xFFFF for v in block[i].tolist()])
    return beats


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> tuple[list[list[int]], list[list[int]]]:
    """[32, 32] BF16 → (bank_low_rows, bank_high_rows)."""
    if tile.shape != (ROWS_PER_REGISTER, 2 * BF16_PER_BEAT):
        raise ValueError(f"expected (32, 32), got {tuple(tile.shape)}")
    low_view  = tile[:, :BF16_PER_BEAT].contiguous().view(torch.int16)
    high_view = tile[:, BF16_PER_BEAT:].contiguous().view(torch.int16)
    low  = [[int(v) & 0xFFFF for v in low_view[i].tolist()]  for i in range(ROWS_PER_REGISTER)]
    high = [[int(v) & 0xFFFF for v in high_view[i].tolist()] for i in range(ROWS_PER_REGISTER)]
    return low, high


def bank_rows_to_bf16_tile(rows: list[list[int]]) -> torch.Tensor:
    """64 rows (32 low + 32 high) → [32, 32] BF16 tile."""
    if len(rows) != 2 * ROWS_PER_REGISTER:
        raise ValueError(f"expected 64 rows, got {len(rows)}")
    out_u16 = np.zeros((ROWS_PER_REGISTER, 2 * BF16_PER_BEAT), dtype=np.uint16)
    for i in range(ROWS_PER_REGISTER):
        out_u16[i, :BF16_PER_BEAT] = rows[i]
        out_u16[i, BF16_PER_BEAT:] = rows[i + ROWS_PER_REGISTER]
    int16_view = out_u16.view(np.int16).copy()
    return torch.from_numpy(int16_view).view(torch.bfloat16)


def main():
    # ── Seeded inputs (match npu-model exactly) ──
    torch.manual_seed(42)
    Q_RAW   = (torch.randn(Q_ROWS, HEAD_DIM) * 0.5).to(torch.bfloat16)
    K0_RAW  = (torch.randn(Q_ROWS, HEAD_DIM) * 0.5).to(torch.bfloat16)
    K1_RAW  = (torch.randn(Q_ROWS, HEAD_DIM) * 0.5).to(torch.bfloat16)
    V0_MLIR = (torch.randn(HEAD_DIM, Q_ROWS) * 0.5).to(torch.bfloat16)
    V1_MLIR = (torch.randn(HEAD_DIM, Q_ROWS) * 0.5).to(torch.bfloat16)

    sa_bf16 = SARTLLinearFunction(
        rows=Q_ROWS, cols=Q_ROWS, out_fmt_sel=OutputFmtSel.OutBF16
    )

    # ── Online-softmax state (broadcast tiles, stored as bank rows) ──
    NEG_BITS   = float_to_bf16(INIT_NEG)
    SCALE_BITS = float_to_bf16(SCALE_VAL)

    m_low  = constant_bf16_rows(ROWS_PER_REGISTER, NEG_BITS)
    m_high = constant_bf16_rows(ROWS_PER_REGISTER, NEG_BITS)
    l_low  = constant_bf16_rows(ROWS_PER_REGISTER, 0)
    l_high = constant_bf16_rows(ROWS_PER_REGISTER, 0)
    O_left_low   = constant_bf16_rows(ROWS_PER_REGISTER, 0)
    O_left_high  = constant_bf16_rows(ROWS_PER_REGISTER, 0)
    O_right_low  = constant_bf16_rows(ROWS_PER_REGISTER, 0)
    O_right_high = constant_bf16_rows(ROWS_PER_REGISTER, 0)
    scale_rows = (
        constant_bf16_rows(ROWS_PER_REGISTER, SCALE_BITS)
        + constant_bf16_rows(ROWS_PER_REGISTER, SCALE_BITS)
    )

    Q_FP8 = bf16_tile_to_e4m3_bytes(Q_RAW)

    # ── Per K-tile flash-attention body ──
    for K_RAW, V_MLIR in [(K0_RAW, V0_MLIR), (K1_RAW, V1_MLIR)]:
        K_FP8 = bf16_tile_to_e4m3_bytes(K_RAW)
        scores_bf16 = sa_bf16(
            Q_FP8, K_FP8, scale_exp=0
        ).to(torch.bfloat16)
        scores_low, scores_high = bf16_tile_to_bank_rows(scores_bf16)
        scores_rows = scores_low + scores_high

        scaled_rows = run_binary_rows("mul", scores_rows, scale_rows)
        tile_max_rows = run_row_reduce_tensor("rmax", scaled_rows)

        m_prev_rows = m_low + m_high
        m_new_rows  = run_binary_rows("pairmax", m_prev_rows, tile_max_rows)

        m_diff_rows  = run_binary_rows("sub", m_prev_rows, m_new_rows)
        exp_diff_rows = run_unary_rows("exp", m_diff_rows)

        # O *= exp_diff (split low/high banks)
        O_left_rows = run_binary_rows(
            "mul", O_left_low + O_left_high, exp_diff_rows
        )
        O_right_rows = run_binary_rows(
            "mul", O_right_low + O_right_high, exp_diff_rows
        )

        exp_s_pre_rows = run_binary_rows("sub", scaled_rows, m_new_rows)
        exp_s_rows = run_unary_rows("exp", exp_s_pre_rows)

        exp_s_bf16 = bank_rows_to_bf16_tile(exp_s_rows)
        exp_s_fp8  = bf16_tile_to_e4m3_bytes(exp_s_bf16)

        # V_top = V_mlir[:32, :] [32, 32]; pushed as W → A @ V_top^T = exp_s @ V_left
        V_top_fp8 = bf16_tile_to_e4m3_bytes(V_MLIR[:Q_ROWS, :].contiguous())
        V_bot_fp8 = bf16_tile_to_e4m3_bytes(V_MLIR[Q_ROWS:, :].contiguous())

        vc_left_bf16  = sa_bf16(exp_s_fp8, V_top_fp8, scale_exp=0).to(torch.bfloat16)
        vc_right_bf16 = sa_bf16(exp_s_fp8, V_bot_fp8, scale_exp=0).to(torch.bfloat16)

        vc_left_low, vc_left_high   = bf16_tile_to_bank_rows(vc_left_bf16)
        vc_right_low, vc_right_high = bf16_tile_to_bank_rows(vc_right_bf16)

        O_left_rows  = run_binary_rows("add", O_left_rows,  vc_left_low + vc_left_high)
        O_right_rows = run_binary_rows("add", O_right_rows, vc_right_low + vc_right_high)

        weighted_l_rows = run_binary_rows("mul", exp_diff_rows, l_low + l_high)
        rowsum_rows = run_row_reduce_tensor("rsum", exp_s_rows)
        l_rows = run_binary_rows("add", weighted_l_rows, rowsum_rows)

        # roll state forward
        m_low,  m_high  = m_new_rows[:ROWS_PER_REGISTER],  m_new_rows[ROWS_PER_REGISTER:]
        l_low,  l_high  = l_rows[:ROWS_PER_REGISTER],     l_rows[ROWS_PER_REGISTER:]
        O_left_low,  O_left_high  = O_left_rows[:ROWS_PER_REGISTER],  O_left_rows[ROWS_PER_REGISTER:]
        O_right_low, O_right_high = O_right_rows[:ROWS_PER_REGISTER], O_right_rows[ROWS_PER_REGISTER:]

    # ── Normalize: O /= l ──
    inv_l_rows = run_unary_rows("rcp", l_low + l_high)
    O_left_rows  = run_binary_rows("mul", O_left_low + O_left_high,   inv_l_rows)
    O_right_rows = run_binary_rows("mul", O_right_low + O_right_high, inv_l_rows)

    O_left_bf16  = bank_rows_to_bf16_tile(O_left_rows)
    O_right_bf16 = bank_rows_to_bf16_tile(O_right_rows)
    O_full_bf16 = torch.cat([O_left_bf16, O_right_bf16], dim=1)   # [32, 64]
    out_beats = col_block_beats(O_full_bf16)

    # ── DRAM preloads ──
    preloads: list[dict] = []
    for tile, base in [
        (Q_RAW,   Q_BASE),
        (K0_RAW,  K0_BASE),
        (K1_RAW,  K1_BASE),
        (V0_MLIR, V0_BASE),
        (V1_MLIR, V1_BASE),
    ]:
        beats = col_block_beats(tile)
        for i, row in enumerate(beats):
            preloads.append({
                "word_offset": base + i,
                "data": pack_u16_le(row),
            })

    checks: list[dict] = [
        {"word_offset": OUT_BASE + i, "expected": pack_u16_le(row)}
        for i, row in enumerate(out_beats)
    ]

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

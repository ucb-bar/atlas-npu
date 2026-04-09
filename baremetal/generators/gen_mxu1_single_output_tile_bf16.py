#!/usr/bin/env python3
"""
gen_mxu1_single_output_tile_bf16.py

Generate one 32x32 output-tile test for a pi0-like large matmul:
    GemmaMLP.up_proj: (816, 2048, 16384)

This test computes:
    C_tile = A_tile @ B_tile + bias

where:
    A_tile is 32 x 2048
    B_tile is 2048 x 32
    bias   is length 32

Conventions:
  - A tiles are stored normally in DRAM.
  - B tiles are stored normally in DRAM as logical 32x32 submatrices.
  - The assembly DMA-loads each B tile, then XLU-transposes it before VMATPUSH.W.
  - Bias is stored as a 32x32 FP8 matrix with repeated rows.
  - VMATPUSH.ACC.FP8.MXU1 initializes the MXU1 accumulator from that FP8 tile.
  - Final output is BF16 popped from the accumulator.

Reference model:
  - Uses IPTLinearRTLFunction (RTL-accurate MXU1 software model) for the
    matmul accumulation with optional FP8 bias.
  - IPTLinearRTLFunction computes F.linear: y = x @ w^T + b
  - For C = A @ B + bias, pass w = B^T so y = A @ B^T^T = A @ B.

Usage:
    python3 gen_mxu1_single_output_tile_bf16.py 42 1
"""

import math
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (
    rand_matrix_fp8_safe,
    quantize_fp8,
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    pack_words_into_beats,
    emit_test_data,
)
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

# Representative pi0-like large matmul
M_FULL = 816
K_FULL = 2048
N_FULL = 16384

TILE = 32
WORDS_PER_BEAT = 8
TILE_BYTES = 1024
OUT_BANK_BYTES = 1024

# ------------------------------------------------------------------
# IPT functional model
# ------------------------------------------------------------------

ipt_bf16 = IPTLinearRTLFunction(
    vec_len=32, num_lanes=32, pipeline_depth=1,
    out_fmt_sel=OutputFmtSel.OutBF16,
)


def pad_cols_to_tile(mat: np.ndarray) -> np.ndarray:
    rows, cols = mat.shape
    if rows != TILE:
        raise ValueError(f"Expected {TILE} rows, got {rows}")
    if cols > TILE:
        raise ValueError(f"Expected <= {TILE} cols, got {cols}")
    if cols == TILE:
        return mat
    pad = np.zeros((rows, TILE - cols), dtype=mat.dtype)
    return np.hstack([mat, pad])


def pad_rows_to_tile(mat: np.ndarray) -> np.ndarray:
    rows, cols = mat.shape
    if cols != TILE:
        raise ValueError(f"Expected {TILE} cols, got {cols}")
    if rows > TILE:
        raise ValueError(f"Expected <= {TILE} rows, got {rows}")
    if rows == TILE:
        return mat
    pad = np.zeros((TILE - rows, cols), dtype=mat.dtype)
    return np.vstack([mat, pad])


def pack_fp8_tile(tile32x32: np.ndarray):
    """
    Pack a logical 32x32 FP8 tile directly as stored in DRAM.
    Each row is 32 FP8 elements = 32 bytes = 8 x 32-bit words = 1 beat.
    Total = 32 beats per tile.
    """
    return pack_words_into_beats(matrix_to_fp8_words(tile32x32), WORDS_PER_BEAT)


def pack_bf16_output_tile(c32x32: np.ndarray):
    """
    Pack BF16 output tile into:
      bank2 = lanes 0..15
      bank3 = lanes 16..31
    Each bank stores one 32-byte row per output row => 32 beats per bank.
    """
    bank2_words = []
    bank3_words = []
    for r in range(TILE):
        bank2_words.extend(matrix_to_bf16_words(c32x32[r, 0:16].reshape(1, -1)))
        bank3_words.extend(matrix_to_bf16_words(c32x32[r, 16:32].reshape(1, -1)))
    bank2_beats = pack_words_into_beats(bank2_words, WORDS_PER_BEAT)
    bank3_beats = pack_words_into_beats(bank3_words, WORDS_PER_BEAT)
    return bank2_beats, bank3_beats


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    use_bias = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True

    # One 32x32 output tile:
    #   A_tile_full: 32 x K_FULL
    #   B_tile_full: K_FULL x 32
    A = rand_matrix_fp8_safe(TILE, K_FULL, seed=seed)
    B = rand_matrix_fp8_safe(K_FULL, TILE, seed=seed + 1)

    A_q = quantize_fp8(A).astype(np.float32)
    B_q = quantize_fp8(B).astype(np.float32)

    if use_bias:
        bias = rand_matrix_fp8_safe(1, TILE, seed=seed + 2).reshape(TILE)
        bias_q = quantize_fp8(bias.reshape(1, -1)).reshape(TILE).astype(np.float32)
    else:
        bias_q = np.zeros((TILE,), dtype=np.float32)

    num_k_tiles = math.ceil(K_FULL / TILE)

    # ------------------------------------------------------------------
    # Reference computation via IPT functional model
    #
    # IPTLinearRTLFunction computes F.linear: y = x @ w^T + b
    # We want C = A @ B + bias, so pass w = B^T → y = A @ B^T^T = A @ B.
    #
    # Bias handling: IPTLinearRTLFunction adds the E4M3 bias at k_tile 0
    # via AddendSel.UseBias, which is numerically equivalent to the
    # hardware's VMATPUSH.ACC.FP8 + VMATMUL.ACC flow (the E4M3→aligned
    # and BF16(E4M3)→aligned conversions produce identical results).
    # ------------------------------------------------------------------

    A_t = torch.from_numpy(A_q)
    W_t = torch.from_numpy(B_q.T.copy())   # (32, K_FULL) — F.linear convention
    b_t = torch.from_numpy(bias_q) if use_bias else None

    C_ref = ipt_bf16(A_t, W_t, b_q=b_t, scale_exp=0).numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Build DRAM payloads for A tiles and logical B tiles
    #
    # IMPORTANT:
    #   We store B in logical orientation.
    #   Assembly XLU-transposes each B tile before VMATPUSH.W.MXU1.
    # ------------------------------------------------------------------

    act_beats = []
    wgt_beats = []

    for t in range(num_k_tiles):
        k0 = t * TILE
        k1 = min((t + 1) * TILE, K_FULL)

        # A tile: 32 x Ksub -> pad to 32 x 32
        a_tile = pad_cols_to_tile(A_q[:, k0:k1]).astype(np.float32)

        # Logical B tile: Ksub x 32 -> pad to 32 x 32
        b_tile = pad_rows_to_tile(B_q[k0:k1, :]).astype(np.float32)

        # DRAM payloads
        act_beats.extend(pack_fp8_tile(a_tile))
        wgt_beats.extend(pack_fp8_tile(b_tile))

    # Bias tile for accumulator preload
    # bias_bank[r, j] = bias[j] for all rows r
    # Stored as FP8 in DRAM; hardware upcasts on VMATPUSH.ACC.FP8.MXU1.
    if use_bias:
        bias_bank = np.tile(bias_q.reshape(1, TILE), (TILE, 1))
    else:
        bias_bank = np.zeros((TILE, TILE), dtype=np.float32)

    bias_bank_q = quantize_fp8(bias_bank)
    bias_beats = pack_fp8_tile(bias_bank_q)

    # Expected BF16 output banks
    c_bank2_beats, c_bank3_beats = pack_bf16_output_tile(C_ref)

    assert len(act_beats) == num_k_tiles * 32
    assert len(wgt_beats) == num_k_tiles * 32
    assert len(bias_beats) == 32
    assert len(c_bank2_beats) == 32
    assert len(c_bank3_beats) == 32

    # DRAM layout
    A_BASE = 0
    W_BASE = A_BASE + len(act_beats)
    BIAS_BASE = W_BASE + len(wgt_beats)
    C2_BASE = BIAS_BASE + len(bias_beats)
    C3_BASE = C2_BASE + len(c_bank2_beats)

    preloads = []
    for i, beat in enumerate(act_beats):
        preloads.append({"word_offset": A_BASE + i, "data": f"0x{beat:X}"})

    for i, beat in enumerate(wgt_beats):
        preloads.append({"word_offset": W_BASE + i, "data": f"0x{beat:X}"})

    for i, beat in enumerate(bias_beats):
        preloads.append({"word_offset": BIAS_BASE + i, "data": f"0x{beat:X}"})

    checks = []
    for i, beat in enumerate(c_bank2_beats):
        checks.append({"word_offset": C2_BASE + i, "expected": f"0x{beat:X}"})

    for i, beat in enumerate(c_bank3_beats):
        checks.append({"word_offset": C3_BASE + i, "expected": f"0x{beat:X}"})

    sys.stderr.write(
        f"[gen_mxu_single_output_tile_bf16] "
        f"shape=({M_FULL},{K_FULL},{N_FULL}) "
        f"use_bias={int(use_bias)} "
        f"num_k_tiles={num_k_tiles}\n"
    )
    sys.stderr.write(
        f"[gen_mxu_single_output_tile_bf16] "
        f"A_BASE={A_BASE} W_BASE={W_BASE} BIAS_BASE={BIAS_BASE} "
        f"C2_BASE={C2_BASE} C3_BASE={C3_BASE}\n"
    )

    emit_test_data(preloads, checks, timeout=500000)


if __name__ == "__main__":
    main()
    
#!/usr/bin/env python3
"""Generate test vectors for `perf_vec_layernorm_32x32.S`.

Row-wise layer normalization on a seeded random BF16 32×32 tile:

    Y = (X - rowmean(X)) / sqrt(rowvar(X) + eps)

All computation routes through atlas's VectorEngineModel (vpu_gen_utils)
for bit-exact match with VREDSUM.ROW, VSUB, VSQUARE, VSQRT, VRECIP, VMUL.

Hardware pipeline (mirrors the .S critical path):
    1. VREDSUM.ROW  X        → row_sum        [39 cy]
    2. VMUL         × (1/32) → row_mean       [65 cy]
    3. VSUB         X - mean → x_centered     [65 cy]
    4. VSQUARE      x_c²     → x_sq           [65 cy]
    5. VREDSUM.ROW  x_sq     → row_sum_sq     [39 cy]
    6. VMUL         × (1/32) → variance       [65 cy]
    7. VSQRT        √var     → std            [65 cy]
    8. VRECIP       1/std    → inv_std        [65 cy]
    9. VMUL         x_c × inv_std → Y         [65 cy]

eps is absorbed into the seed: all values are drawn from randn×0.5, so
variance is never near zero and sqrt/recip are numerically safe. No
explicit eps tile is needed in hardware or golden.

DRAM layout (must match perf_vec_layernorm_32x32.S):
    0x0000  X  (2048 B, BF16 32×32, two [32,16] column blocks)
    0x0800  Y  (2048 B, BF16 32×32, output, checked vs golden)

Column-blocked layout: each [32, 16] BF16 sub-block = one mreg (32 beats
× 16 BF16 lanes). A [32, 32] tile = 2 column blocks = 64 beats total.

Scalar registers in .S use 0x3000 BF16 (= 1/32) for mean/variance
scaling. The golden uses the same constant routed through the VPU model
so any BF16 rounding in the multiply is reproduced exactly.
"""

import os
import sys

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

ROWS      = 32
COLS      = 32
TIMEOUT   = 20000

# 1/32 in BF16 — must match VLI.ALL 0x3000 in the .S setup block
INV32_BITS = 0x3D00   # BF16 encoding of 0.03125

X_BASE  = 0x0000 // 32   # beat 0
OUT_BASE = 0x0800 // 32  # beat 64


def col_block_beats(tile: torch.Tensor) -> list[list[int]]:
    """Pack a [32, 32] BF16 tile into atlas column-blocked DRAM layout.

    Produces 2 column blocks of 32 beats each (64 beats total).
    Iteration order: col_block (outer), row (inner) — matching DMA.LOAD
    stride in the .S file.
    """
    rows, cols = tile.shape
    if rows % ROWS_PER_REGISTER != 0 or cols % BF16_PER_BEAT != 0:
        raise ValueError(
            f"col_block_beats: shape ({rows},{cols}) not divisible by "
            f"({ROWS_PER_REGISTER},{BF16_PER_BEAT})"
        )
    beats: list[list[int]] = []
    for col_start in range(0, cols, BF16_PER_BEAT):
        block = (
            tile[:, col_start : col_start + BF16_PER_BEAT]
            .contiguous()
            .view(torch.int16)
            .numpy()
        )
        for i in range(rows):
            beats.append([int(v) & 0xFFFF for v in block[i].tolist()])
    return beats


def tile_to_bank_rows(tile: torch.Tensor) -> tuple[list[list[int]], list[list[int]]]:
    """Split a [32, 32] BF16 tile into (low_bank_rows, high_bank_rows).

    low  ← tile[:, :16]   (first  16 BF16 lanes, mreg low  bank)
    high ← tile[:, 16:]   (second 16 BF16 lanes, mreg high bank)
    Each bank is a list of 32 rows, each row a list of 16 uint16 values.
    """
    assert tile.shape == (ROWS, COLS), f"expected (32,32), got {tuple(tile.shape)}"
    low  = tile[:, :BF16_PER_BEAT].contiguous().view(torch.int16)
    high = tile[:, BF16_PER_BEAT:].contiguous().view(torch.int16)
    low_rows  = [[int(v) & 0xFFFF for v in low[i].tolist()]  for i in range(ROWS)]
    high_rows = [[int(v) & 0xFFFF for v in high[i].tolist()] for i in range(ROWS)]
    return low_rows, high_rows


def bank_rows_to_tile(
    low_rows: list[list[int]], high_rows: list[list[int]]
) -> torch.Tensor:
    """Reconstruct a [32, 32] BF16 tile from (low_bank_rows, high_bank_rows)."""
    import numpy as np
    low_np  = np.array(low_rows,  dtype=np.uint16).view(np.int16)
    high_np = np.array(high_rows, dtype=np.uint16).view(np.int16)
    low_t  = torch.from_numpy(low_np.copy()).view(torch.bfloat16)
    high_t = torch.from_numpy(high_np.copy()).view(torch.bfloat16)
    return torch.cat([low_t, high_t], dim=1)


def main() -> None:
    # ── Seeded input (must match .S seed and gen script contract) ──
    torch.manual_seed(42)
    X_raw = (torch.randn(ROWS, COLS) * 0.5).to(torch.bfloat16)

    # ── Decompose X into (low, high) bank row lists ──
    x_low, x_high = tile_to_bank_rows(X_raw)
    x_rows = x_low + x_high   # 64 rows: first 32 = low bank, last 32 = high

    # 1/32 broadcast tile (same constant used by VLI.ALL 0x3000 in the .S)
    inv32_rows = constant_bf16_rows(ROWS, INV32_BITS) * 2   # 64 rows (low+high)

    # ── Step 1: row sum ──
    rowsum_rows = run_row_reduce_tensor("rsum", x_rows)

    # ── Step 2: row mean = rowsum × (1/32) ──
    mean_rows = run_binary_rows("mul", rowsum_rows, inv32_rows)

    # ── Step 3: x - mean ──
    centered_rows = run_binary_rows("sub", x_rows, mean_rows)

    # ── Step 4: (x - mean)² ──
    sq_rows = run_unary_rows("square", centered_rows)

    # ── Step 5: row sum of squares ──
    rowsum_sq_rows = run_row_reduce_tensor("rsum", sq_rows)

    # ── Step 6: variance = rowsum_sq × (1/32) ──
    var_rows = run_binary_rows("mul", rowsum_sq_rows, inv32_rows)

    # ── Step 7: std = sqrt(var) ──
    std_rows = run_unary_rows("sqrt", var_rows)

    # ── Step 8: inv_std = 1 / std ──
    inv_std_rows = run_unary_rows("rcp", std_rows)

    # ── Step 9: Y = (x - mean) * inv_std ──
    y_rows = run_binary_rows("mul", centered_rows, inv_std_rows)

    # ── Reconstruct [32, 32] BF16 output tile ──
    y_low  = y_rows[:ROWS]
    y_high = y_rows[ROWS:]
    Y_tile = bank_rows_to_tile(y_low, y_high)

    # ── DRAM preload: X → 0x0000 ──
    preloads: list[dict] = []
    for i, row in enumerate(col_block_beats(X_raw)):
        preloads.append({
            "word_offset": X_BASE + i,
            "data": pack_u16_le(row),
        })

    # ── Golden checks: Y → 0x0800 ──
    checks: list[dict] = [
        {"word_offset": OUT_BASE + i, "expected": pack_u16_le(row)}
        for i, row in enumerate(col_block_beats(Y_tile))
    ]

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()
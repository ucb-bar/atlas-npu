#!/usr/bin/env python3
"""Generate test vectors for `perf_vec_rmsnorm_softmax_32x32.S`.

Computes two independent activation kernels over seeded random BF16 32×32
tiles, mirroring the hardware's schedule exactly:

    Y_x = X / sqrt(mean(X²))                           [RMSNorm]
    Y_z = exp(Z - rowmax(Z)) / rowsum(exp(Z - rowmax)) [Softmax]

Hardware schedule (timed region, 637 cycles):
    Slot  1 [  0– 34]: VREDMAX.ROW m6←m4                    34 cy
    Slot  2 [ 35–100]: VSQUARE m2←m0                        65 cy
    Slot  3 [101–139]: VREDSUM.ROW m2←m2                    39 cy
    Slot  4 [141–206]: VSUB m6←m4,m6                        65 cy
    Slot  5 [207–272]: VMUL m2←m2,m12                       65 cy
    Slot  6 [273–338]: VSQRT m2←m2  ∥  VEXP m8←m6          65 cy  ← PAIR ✓
    Slot  7 [339–378]: VREDSUM.ROW m8←m8                    39 cy
    Slot  8 [379–444]: VRECIP m2←m2  (X: 1/rms)            65 cy
    Slot  9 [445–510]: VRECIP m8←m8  (Z: 1/rowsum)         65 cy
    Slot 10 [511–576]: VMUL m10←m0,m2  (Y_x)               65 cy
    Slot 11 [577–637]: VMUL m8←m6,m8   (Y_z)               65 cy
    Total: 637 cy

    element-ops: 12 ops × 1024 = 12288
    actual throughput: 12288 / 637 = 19.3 elem/cy
    peak throughput:   2 × 1024 / 65 = 31.5 elem/cy
    utilization: 19.3 / 31.5 = 61.3%

DRAM layout (must match perf_vec_rmsnorm_softmax_32x32.S):
    0x0000  X   (2048 B, BF16 32×32)  RMSNorm input
    0x0800  Z   (2048 B, BF16 32×32)  Softmax input
    0x1000  Y_x (2048 B, BF16 32×32)  RMSNorm output, checked vs golden
    0x1800  Y_z (2048 B, BF16 32×32)  Softmax output, checked vs golden
"""

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
    pack_u16_le,
    run_binary_rows,
    run_row_reduce_tensor,
    run_unary_rows,
)

ROWS    = 32
COLS    = 32
TIMEOUT = 25000

INV32_BITS = 0x3000   # BF16 for 1/32 → m12,m13

X_BASE  = 0x0000 // 32   # beat   0
Z_BASE  = 0x0800 // 32   # beat  64
YX_BASE = 0x1000 // 32   # beat 128
YZ_BASE = 0x1800 // 32   # beat 192


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def col_block_beats(tile: torch.Tensor) -> list[list[int]]:
    """Pack a [32, 32] BF16 tile into atlas column-blocked DRAM layout."""
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


def tile_to_bank_rows(
    tile: torch.Tensor,
) -> tuple[list[list[int]], list[list[int]]]:
    """Split a [32, 32] BF16 tile into (low_bank_rows, high_bank_rows)."""
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
    low_np  = np.array(low_rows,  dtype=np.uint16).view(np.int16).copy()
    high_np = np.array(high_rows, dtype=np.uint16).view(np.int16).copy()
    low_t  = torch.from_numpy(low_np).view(torch.bfloat16)
    high_t = torch.from_numpy(high_np).view(torch.bfloat16)
    return torch.cat([low_t, high_t], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Golden streams
# ─────────────────────────────────────────────────────────────────────────────

def compute_rmsnorm_golden(
    x_rows: list[list[int]],
    inv32_rows: list[list[int]],
) -> list[list[int]]:
    """RMSNorm: Y_x = X / sqrt(mean(X²)).

    Mirrors hardware X-stream in slot order:
        Slot 2:  VSQUARE  → run_unary_rows("square", x_rows)
        Slot 3:  VREDSUM.ROW → run_row_reduce_tensor("rsum", sq_rows)
        Slot 5:  VMUL ×1/32  → run_binary_rows("mul", ..., inv32_rows)
        Slot 6:  VSQRT       → run_unary_rows("sqrt", meansq_rows)
        Slot 8:  VRECIP      → run_unary_rows("rcp",  rms_rows)
        Slot 10: VMUL        → run_binary_rows("mul", x_rows, inv_rms_rows)
    """
    sq_rows        = run_unary_rows("square", x_rows)
    rowsum_sq_rows = run_row_reduce_tensor("rsum", sq_rows)
    meansq_rows    = run_binary_rows("mul", rowsum_sq_rows, inv32_rows)
    rms_rows       = run_unary_rows("sqrt", meansq_rows)
    inv_rms_rows   = run_unary_rows("rcp",  rms_rows)
    yx_rows        = run_binary_rows("mul", x_rows, inv_rms_rows)
    return yx_rows


def compute_softmax_golden(
    z_rows: list[list[int]],
) -> list[list[int]]:
    """Stable softmax: Y_z = exp(Z - rowmax(Z)) / rowsum(exp(Z - rowmax(Z))).

    Mirrors hardware Z-stream in slot order:
        Slot 1:  VREDMAX.ROW → run_row_reduce_tensor("rmax", z_rows)
        Slot 4:  VSUB        → run_binary_rows("sub",  z_rows, rowmax_rows)
        Slot 6:  VEXP        → run_unary_rows("exp",   centered_rows)
        Slot 7:  VREDSUM.ROW → run_row_reduce_tensor("rsum", exp_rows)
        Slot 9:  VRECIP      → run_unary_rows("rcp",   rowsum_rows)
        Slot 11: VMUL        → run_binary_rows("mul",  exp_rows, inv_rowsum_rows)

    VREDMAX.ROW broadcasts the per-row maximum to all lanes (same semantics
    as VREDSUM.ROW), so the subtraction Z - rowmax is elementwise correct.
    """
    rowmax_rows     = run_row_reduce_tensor("rmax", z_rows)
    centered_rows   = run_binary_rows("sub",  z_rows, rowmax_rows)
    exp_rows        = run_unary_rows("exp",   centered_rows)
    rowsum_rows     = run_row_reduce_tensor("rsum", exp_rows)
    inv_rowsum_rows = run_unary_rows("rcp",   rowsum_rows)
    yz_rows         = run_binary_rows("mul",  exp_rows, inv_rowsum_rows)
    return yz_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(42)
    X_raw = (torch.randn(ROWS, COLS) * 0.5).to(torch.bfloat16)
    Z_raw = (torch.randn(ROWS, COLS) * 0.5).to(torch.bfloat16)

    x_low, x_high = tile_to_bank_rows(X_raw)
    z_low, z_high = tile_to_bank_rows(Z_raw)
    x_rows = x_low + x_high
    z_rows = z_low + z_high

    inv32_rows = constant_bf16_rows(ROWS, INV32_BITS) * 2

    yx_rows = compute_rmsnorm_golden(x_rows, inv32_rows)
    yz_rows = compute_softmax_golden(z_rows)

    Yx_tile = bank_rows_to_tile(yx_rows[:ROWS], yx_rows[ROWS:])
    Yz_tile = bank_rows_to_tile(yz_rows[:ROWS], yz_rows[ROWS:])

    preloads: list[dict] = []
    for tile_raw, base in [(X_raw, X_BASE), (Z_raw, Z_BASE)]:
        for i, row in enumerate(col_block_beats(tile_raw)):
            preloads.append({"word_offset": base + i, "data": pack_u16_le(row)})

    checks: list[dict] = []
    for tile_out, base in [(Yx_tile, YX_BASE), (Yz_tile, YZ_BASE)]:
        for i, row in enumerate(col_block_beats(tile_out)):
            checks.append({"word_offset": base + i, "expected": pack_u16_le(row)})

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()
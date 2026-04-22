#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_rms_norm.S` (128x32).

Port of jrmills20 parameterized_rms_norm.py. N=32 fixed, M=128 (4 groups).
y = x * rsqrt(mean(x^2) + eps). Per-group independent — the row-sum
reduction spans only a single 32-lane row thanks to atlas's pair-op
VREDSUM.ROW.BF16.
"""

import json

import torch

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    run_binary_rows,
    run_row_reduce_tensor,
    run_unary_rows,
    tensor_checks,
    tensor_preloads,
)


M = 512
N = 32
TILE = ROWS_PER_REGISTER
M_GROUPS = M // TILE
BEATS_PER_TILE = ROWS_PER_TENSOR
BEATS_PER_OPERAND = M_GROUPS * BEATS_PER_TILE

X_BASE = 0
OUT_BASE = BEATS_PER_OPERAND

BF16_INV_DIM = 0x3D00   # 1/32
BF16_EPS = 0x3586       # ~1e-6

TIMEOUT = 2500000


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> list[list[int]]:
    if tile.shape != (ROWS_PER_REGISTER, 2 * BF16_PER_BEAT):
        raise ValueError(
            f"expected ({ROWS_PER_REGISTER}, {2 * BF16_PER_BEAT}) bf16 tile, "
            f"got {tuple(tile.shape)}"
        )
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bank0 = tile[:, :BF16_PER_BEAT].contiguous().view(torch.int16)
    bank1 = tile[:, BF16_PER_BEAT:].contiguous().view(torch.int16)
    rows: list[list[int]] = []
    for half in (bank0, bank1):
        for i in range(ROWS_PER_REGISTER):
            rows.append([int(v) & 0xFFFF for v in half[i].tolist()])
    return rows


def stack_groups_to_rows(mat: torch.Tensor) -> list[list[int]]:
    if mat.shape != (M, N):
        raise ValueError(f"expected ({M},{N}) tensor, got {tuple(mat.shape)}")
    rows: list[list[int]] = []
    for g in range(M_GROUPS):
        group = mat[g * TILE:(g + 1) * TILE, :].contiguous()
        rows.extend(bf16_tile_to_bank_rows(group))
    return rows


def const_rows(value_u16: int, n_rows: int) -> list[list[int]]:
    return [[value_u16] * BF16_PER_BEAT for _ in range(n_rows)]


def main():
    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=torch.bfloat16)
    x_rows = stack_groups_to_rows(x)

    n = len(x_rows)
    inv_dim_rows = const_rows(BF16_INV_DIM, n)
    eps_rows = const_rows(BF16_EPS, n)

    x_sq = run_unary_rows("square", x_rows)
    row_sum = run_row_reduce_tensor("rsum", x_sq)
    mean_sq = run_binary_rows("mul", row_sum, inv_dim_rows)
    var_eps = run_binary_rows("add", mean_sq, eps_rows)
    sqrt_rows = run_unary_rows("sqrt", var_eps)
    inv_rms = run_unary_rows("rcp", sqrt_rows)
    y_rows = run_binary_rows("mul", x_rows, inv_rms)

    preloads = tensor_preloads(X_BASE, x_rows)
    checks = tensor_checks(OUT_BASE, y_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_silu.S` (64x64).

Port of jrmills20 parameterized_silu.py. silu(x) = x / (1 + exp(-x)),
synthesized through atlas primitives (mul, exp, add, rcp, mul). Golden
walks the same chain via VectorEngineModel so every intermediate
matches what the RTL computes.
"""

import json

import torch

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    run_binary_rows,
    run_unary_rows,
    tensor_checks,
    tensor_preloads,
)


M = 128
N = 128
TILE = ROWS_PER_REGISTER
M_TILES = M // TILE
N_TILES = N // TILE
TOTAL_TILES = M_TILES * N_TILES

BEATS_PER_TILE = ROWS_PER_TENSOR
BEATS_PER_OPERAND = TOTAL_TILES * BEATS_PER_TILE

X_BASE = 0
OUT_BASE = BEATS_PER_OPERAND

BF16_PLUS_ONE = 0x3F80
BF16_MINUS_ONE = 0xBF80

TIMEOUT = 2000000


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


def tile_tensor_to_rows(mat: torch.Tensor) -> list[list[int]]:
    if mat.shape != (M, N):
        raise ValueError(f"expected ({M},{N}) tensor, got {tuple(mat.shape)}")
    rows: list[list[int]] = []
    for mi in range(M_TILES):
        for ni in range(N_TILES):
            tile = mat[mi * TILE:(mi + 1) * TILE,
                       ni * TILE:(ni + 1) * TILE].contiguous()
            rows.extend(bf16_tile_to_bank_rows(tile))
    return rows


def const_rows(value_u16: int, n_rows: int) -> list[list[int]]:
    return [[value_u16] * BF16_PER_BEAT for _ in range(n_rows)]


def main():
    torch.manual_seed(42)
    x = torch.randn(M, N, dtype=torch.bfloat16)
    x_rows = tile_tensor_to_rows(x)

    n = len(x_rows)
    minus_one = const_rows(BF16_MINUS_ONE, n)
    plus_one = const_rows(BF16_PLUS_ONE, n)

    minus_x = run_binary_rows("mul", x_rows, minus_one)
    exp_neg_x = run_unary_rows("exp", minus_x)
    denom = run_binary_rows("add", exp_neg_x, plus_one)
    sigmoid = run_unary_rows("rcp", denom)
    silu = run_binary_rows("mul", x_rows, sigmoid)

    preloads = tensor_preloads(X_BASE, x_rows)
    checks = tensor_checks(OUT_BASE, silu)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

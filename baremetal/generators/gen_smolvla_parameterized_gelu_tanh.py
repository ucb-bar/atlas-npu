#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_gelu_tanh.S` (64x64).

Port of jrmills20 parameterized_gelu_tanh.py. GELU tanh approximation:
    y = 0.5 * X * (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X^3)))
Golden walks the same chain through atlas's VectorEngineModel.
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

BF16_ONE = 0x3F80
BF16_HALF = 0x3F00
BF16_044715 = 0x3D37
BF16_SQRT_2_OVER_PI = 0x3F4C

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
    x = (torch.randn(M, N, dtype=torch.bfloat16) * 0.5).to(torch.bfloat16)
    x_rows = tile_tensor_to_rows(x)

    n = len(x_rows)
    c_044715 = const_rows(BF16_044715, n)
    c_sqrt = const_rows(BF16_SQRT_2_OVER_PI, n)
    c_one = const_rows(BF16_ONE, n)
    c_half = const_rows(BF16_HALF, n)

    x_cubed = run_unary_rows("cube", x_rows)
    t1 = run_binary_rows("mul", c_044715, x_cubed)
    t2 = run_binary_rows("add", x_rows, t1)
    t3 = run_binary_rows("mul", c_sqrt, t2)
    t4 = run_unary_rows("tanh", t3)
    t5 = run_binary_rows("add", c_one, t4)
    t6 = run_binary_rows("mul", c_half, t5)
    gelu = run_binary_rows("mul", x_rows, t6)

    preloads = tensor_preloads(X_BASE, x_rows)
    checks = tensor_checks(OUT_BASE, gelu)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

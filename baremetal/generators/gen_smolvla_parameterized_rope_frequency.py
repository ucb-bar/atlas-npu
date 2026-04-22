#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_rope_frequency.S` (64x64).

Port of jrmills20 parameterized_rope_frequency.py (npu-model-tapeout-kernels).
Elementwise cos on a 64x64 BF16 tensor split into four 32x32 tiles.
Golden via atlas's `VectorEngineModel` (run_unary_rows("cos", ...)).
"""

import json

import torch

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
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

TIMEOUT = 1000000


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


def main():
    torch.manual_seed(47)
    x = (torch.randn(M, N, dtype=torch.bfloat16) * 0.5).to(torch.bfloat16)
    x_rows = tile_tensor_to_rows(x)
    y_rows = run_unary_rows("cos", x_rows)

    preloads = tensor_preloads(X_BASE, x_rows)
    checks = tensor_checks(OUT_BASE, y_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

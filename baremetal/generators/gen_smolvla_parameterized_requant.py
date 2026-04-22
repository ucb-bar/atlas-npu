#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_requant.S` (64x64).

Port of jrmills20 parameterized_requant.py. BF16 → FP8 E4M3 cast at
unit E8M0 scale (0x7F = 2^0). Golden via atlas run_fp8_pack_rows.
Output tile stride is half the input stride (FP8 packs 32 lanes per
beat vs BF16's 16 lanes).
"""

import json

import torch

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    fp8_checks,
    run_fp8_pack_rows,
    tensor_preloads,
)


M = 128
N = 128
TILE = ROWS_PER_REGISTER
M_TILES = M // TILE
N_TILES = N // TILE
TOTAL_TILES = M_TILES * N_TILES

BEATS_PER_TILE_BF16 = ROWS_PER_TENSOR          # 64 beats / tile (input)
BEATS_PER_TILE_FP8 = ROWS_PER_REGISTER         # 32 beats / tile (packed output)

BEATS_X_ALL = TOTAL_TILES * BEATS_PER_TILE_BF16  # 1024 beats (32 KiB)
BEATS_Y_ALL = TOTAL_TILES * BEATS_PER_TILE_FP8   # 512 beats (16 KiB)

X_BASE = 0
OUT_BASE = BEATS_X_ALL

SCALE_E8M0_UNIT = 0x7F
TIMEOUT = 1000000


def bf16_tile_to_bank_rows(
    tile: torch.Tensor,
) -> tuple[list[list[int]], list[list[int]]]:
    if tile.shape != (ROWS_PER_REGISTER, 2 * BF16_PER_BEAT):
        raise ValueError(
            f"expected ({ROWS_PER_REGISTER}, {2 * BF16_PER_BEAT}) bf16 tile, "
            f"got {tuple(tile.shape)}"
        )
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bank0 = tile[:, :BF16_PER_BEAT].contiguous().view(torch.int16)
    bank1 = tile[:, BF16_PER_BEAT:].contiguous().view(torch.int16)
    low = [[int(v) & 0xFFFF for v in bank0[i].tolist()]
           for i in range(ROWS_PER_REGISTER)]
    high = [[int(v) & 0xFFFF for v in bank1[i].tolist()]
            for i in range(ROWS_PER_REGISTER)]
    return low, high


def main():
    torch.manual_seed(48)
    x = (torch.randn(M, N, dtype=torch.bfloat16) * 0.5).to(torch.bfloat16)

    preload_rows: list[list[int]] = []
    packed_rows: list[list[int]] = []
    for mi in range(M_TILES):
        for ni in range(N_TILES):
            tile = x[mi * TILE:(mi + 1) * TILE,
                     ni * TILE:(ni + 1) * TILE].contiguous()
            low, high = bf16_tile_to_bank_rows(tile)
            preload_rows.extend(low)
            preload_rows.extend(high)
            packed_rows.extend(
                run_fp8_pack_rows(low, high, SCALE_E8M0_UNIT)
            )

    preloads = tensor_preloads(X_BASE, preload_rows)
    checks = fp8_checks(OUT_BASE, packed_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

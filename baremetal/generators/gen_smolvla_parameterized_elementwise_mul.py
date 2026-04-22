#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_elementwise_mul.S` (64x64).

Port of jrmills20 parameterized_elementwise_mul.py (npu-model-tapeout-kernels).
Tiles a 64x64 bf16 multiply into four 32x32 tiles in tile-major order.
Each tile is laid out bank-split in DRAM (Atlas convention, matches stock
smolvla_elementwise_mul). Golden comes from atlas's `VectorEngineModel`
(via `run_binary_rows("mul", ...)`), not PyTorch.
"""

import json

import torch

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    run_binary_rows,
    tensor_checks,
    tensor_preloads,
)


M = 128
N = 128
TILE = ROWS_PER_REGISTER                    # 32
M_TILES = M // TILE                         # 4
N_TILES = N // TILE                         # 4
TOTAL_TILES = M_TILES * N_TILES             # 16

# One beat = 32 bytes (16 bf16 lanes). One tile = 64 beats (bank0 + bank1).
BEATS_PER_TILE = ROWS_PER_TENSOR            # 64
BEATS_PER_OPERAND = TOTAL_TILES * BEATS_PER_TILE   # 1024 = 32 KiB

A_BASE = 0
B_BASE = BEATS_PER_OPERAND
OUT_BASE = 2 * BEATS_PER_OPERAND

TIMEOUT = 1000000


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> list[list[int]]:
    """Lay a (32, 32) bf16 tile out as 64 rows of 16 BF16 lanes.

    Bank 0 is cols 0..15 (32 rows), bank 1 is cols 16..31 (32 rows).
    Matches gen_smolvla_elementwise_mul.bf16_tile_to_bank_rows.
    """
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
    """(M, N) bf16 tensor → bank-split rows concatenated in tile-major order."""
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
    torch.manual_seed(43)
    input_a = torch.randn(M, N, dtype=torch.bfloat16)
    input_b = torch.randn(M, N, dtype=torch.bfloat16)

    a_rows = tile_tensor_to_rows(input_a)
    b_rows = tile_tensor_to_rows(input_b)

    preloads = tensor_preloads(A_BASE, a_rows)
    preloads.extend(tensor_preloads(B_BASE, b_rows))

    out_rows = run_binary_rows("mul", a_rows, b_rows)
    checks = tensor_checks(OUT_BASE, out_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

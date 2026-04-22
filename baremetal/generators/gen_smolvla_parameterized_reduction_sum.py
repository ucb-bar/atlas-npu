#!/usr/bin/env python3
"""Generate test vectors for `smolvla_parameterized_reduction_sum.S` (128x32).

Port of jrmills20 parameterized_reduction_sum.py. N is fixed at 32
(Jeremy's constraint); M scales via stacked 32-row groups. Row-reduce is
local per group (VREDSUM.ROW.BF16 acts on a single 32x32 pair-op), so the
generalized `run_row_reduce_tensor` just folds over each 64-row chunk.
"""

import json

import torch

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    run_row_reduce_tensor,
    tensor_checks,
    tensor_preloads,
)


M = 512
N = 32
TILE = ROWS_PER_REGISTER
M_GROUPS = M // TILE                         # 16
BEATS_PER_TILE = ROWS_PER_TENSOR             # 64 (bank0 + bank1)
BEATS_PER_OPERAND = M_GROUPS * BEATS_PER_TILE

X_BASE = 0
OUT_BASE = BEATS_PER_OPERAND

TIMEOUT = 1500000


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
    """(M, 32) bf16 → bank-split rows concatenated in group-major order."""
    if mat.shape != (M, N):
        raise ValueError(f"expected ({M},{N}) tensor, got {tuple(mat.shape)}")
    rows: list[list[int]] = []
    for g in range(M_GROUPS):
        group = mat[g * TILE:(g + 1) * TILE, :].contiguous()
        rows.extend(bf16_tile_to_bank_rows(group))
    return rows


def main():
    torch.manual_seed(46)
    x = torch.randn(M, N, dtype=torch.bfloat16)
    x_rows = stack_groups_to_rows(x)

    out_rows = run_row_reduce_tensor("rsum", x_rows)

    preloads = tensor_preloads(X_BASE, x_rows)
    checks = tensor_checks(OUT_BASE, out_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

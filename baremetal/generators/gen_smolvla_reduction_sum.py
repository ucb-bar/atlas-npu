#!/usr/bin/env python3
"""Generate test vectors for `smolvla_reduction_sum.S`.

Port of npu-model/npu_model/configs/programs/smolvla_reduction_sum.py.
Atlas exposes a one-shot `VREDSUM.ROW.BF16` that reduces across all
32 lanes of the (m, m+1) bank pair in a single op, broadcasting the
scalar back to all 16 lanes of both output banks. Golden uses atlas's
row-reduce model — semantics differ from npu-model's halves-add-then-
reduce in BF16 rounding edge cases, but the harness reference is
`VectorEngineModel`, not PyTorch.
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


BEATS_PER_TENSOR = ROWS_PER_TENSOR
TIMEOUT = 30000

X_BASE = 0
OUT_BASE = BEATS_PER_TENSOR


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


def main():
    torch.manual_seed(46)
    x_tile = torch.randn(32, 32, dtype=torch.bfloat16)
    x_rows = bf16_tile_to_bank_rows(x_tile)

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

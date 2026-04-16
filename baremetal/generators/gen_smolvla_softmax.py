#!/usr/bin/env python3
"""Generate test vectors for `smolvla_softmax.S`.

Port of npu-model/npu_model/configs/programs/smolvla_softmax.py.
Row-wise stable softmax on a 32x32 BF16 tile:

    y = exp(X - rowmax(X)) / rowsum(exp(...))

Atlas's VREDMAX.ROW / VREDSUM.ROW reduce the full (m, m+1) pair in
one op, so the npu-model bank-wise split (per-bank reduce then
combine) collapses into a single reduction per stage.

Golden uses atlas's VectorEngineModel — bit-exact with RTL.
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
    torch.manual_seed(50)
    x_tile = (torch.randn(32, 32, dtype=torch.bfloat16) * 2.0).to(torch.bfloat16)
    x_rows = bf16_tile_to_bank_rows(x_tile)

    row_max = run_row_reduce_tensor("rmax", x_rows)
    shifted = run_binary_rows("sub", x_rows, row_max)
    exped = run_unary_rows("exp", shifted)
    row_sum = run_row_reduce_tensor("rsum", exped)
    inv_sum = run_unary_rows("rcp", row_sum)
    y_rows = run_binary_rows("mul", exped, inv_sum)

    preloads = tensor_preloads(X_BASE, x_rows)
    checks = tensor_checks(OUT_BASE, y_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

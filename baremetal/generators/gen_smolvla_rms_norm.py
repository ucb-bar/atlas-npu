#!/usr/bin/env python3
"""Generate test vectors for `smolvla_rms_norm.S`.

Port of npu-model/npu_model/configs/programs/smolvla_rms_norm.py.
Computes y = x * rsqrt(mean(x^2) + eps) on a 32x32 BF16 tile.

Atlas's VREDSUM.ROW.BF16 is a one-shot pair op over the full 32-lane
(m, m+1) bank pair, so we don't need npu-model's two-bank split
(two row-sums then add); a single VREDSUM produces the full row sum
broadcast across both output banks.

Constants (baked in via VLI.ALL):
    inv_dim = 1/32       → BF16 0x3D00
    eps     = 1e-6       → BF16 0x3586 (rounds to ~9.98e-7)

Golden walks the exact pipeline through atlas's VectorEngineModel so
every BF16 intermediate matches the RTL at bit level.
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

BF16_INV_DIM = 0x3D00  # 1/32
BF16_EPS = 0x3586      # 1e-6


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


def const_rows(value_u16: int, n_rows: int) -> list[list[int]]:
    return [[value_u16] * BF16_PER_BEAT for _ in range(n_rows)]


def main():
    torch.manual_seed(42)
    x_tile = torch.randn(32, 32, dtype=torch.bfloat16)
    x_rows = bf16_tile_to_bank_rows(x_tile)

    n = len(x_rows)
    inv_dim_rows = const_rows(BF16_INV_DIM, n)
    eps_rows = const_rows(BF16_EPS, n)

    # X^2 via VSQUARE (unary) — VMUL x,x would double-read bank 0 and
    # trip MregFile's one-read-port-per-bank assertion.
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

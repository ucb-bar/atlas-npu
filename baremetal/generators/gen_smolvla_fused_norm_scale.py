#!/usr/bin/env python3
"""Generate test vectors for `smolvla_fused_norm_scale.S`.

Port of npu-model/npu_model/configs/programs/smolvla_fused_norm_scale.py,
widened from npu-model's [32, 16] single-bank shape to atlas's canonical
[32, 32] two-bank shape so the kernel exercises both mreg banks.

Computes `output[i, j] = matrix[i, j] * rsqrt(variance[i, j])` where the
host pre-broadcasts the 1-D variance to the matrix shape (matches the
demo simplification in the npu-model reference).

Atlas has no `VRSQRT` opcode, so rsqrt is synthesized as
`VRECIP.BF16(VSQRT(...))`. Variance is forced positive via `abs() + 0.1`
to keep `VSQRT` in its valid input range.
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


BEATS_PER_TENSOR = ROWS_PER_TENSOR
TIMEOUT = 30000

VAR_BASE = 0
MAT_BASE = VAR_BASE + BEATS_PER_TENSOR
OUT_BASE = MAT_BASE + BEATS_PER_TENSOR


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
    torch.manual_seed(42)
    variance_tile = (torch.randn(32, 32).abs() + 0.1).to(torch.bfloat16)
    matrix_tile = torch.randn(32, 32, dtype=torch.bfloat16)

    var_rows = bf16_tile_to_bank_rows(variance_tile)
    mat_rows = bf16_tile_to_bank_rows(matrix_tile)

    sqrt_rows = run_unary_rows("sqrt", var_rows)
    rsqrt_rows = run_unary_rows("rcp", sqrt_rows)
    out_rows = run_binary_rows("mul", mat_rows, rsqrt_rows)

    preloads = tensor_preloads(VAR_BASE, var_rows) + tensor_preloads(MAT_BASE, mat_rows)
    checks = tensor_checks(OUT_BASE, out_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

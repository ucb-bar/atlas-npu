#!/usr/bin/env python3
"""Generate test vectors for `smolvla_rope_frequency.S`.

Port of npu-model/npu_model/configs/programs/smolvla_rope_frequency.py.
Elementwise cos on a 32x32 BF16 tile — the trailing cos leg of RoPE
frequency precompute. Uses atlas VCOS pair op; golden through
VectorEngineModel's `cos` path.
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
    torch.manual_seed(47)
    x_tile = (torch.randn(32, 32, dtype=torch.bfloat16) * 0.5).to(torch.bfloat16)
    x_rows = bf16_tile_to_bank_rows(x_tile)

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

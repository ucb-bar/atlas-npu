#!/usr/bin/env python3
"""Generate test vectors for `smolvla_elementwise_sub.S`.

Port of npu-model/npu_model/configs/programs/smolvla_elementwise_sub.py.
Inputs match that kernel's seed (`torch.manual_seed(44); randn(32,32,bf16)`
for A then B). Golden output comes from atlas's `VectorEngineModel`
(via `run_binary_rows("sub", ...)`), not PyTorch — so the harness
reflects atlas's BF16 sub semantics.
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


BEATS_PER_TENSOR = ROWS_PER_TENSOR
TIMEOUT = 30000

A_BASE = 0
B_BASE = BEATS_PER_TENSOR
OUT_BASE = 2 * BEATS_PER_TENSOR


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> list[list[int]]:
    """Lay a (32, 32) bf16 tile out as 64 rows of 16 BF16 lanes.

    Bank 0 is cols 0..15 (32 rows), bank 1 is cols 16..31 (32 rows).
    Returns rows as plain Python lists of uint16 bit patterns so they
    plug straight into vpu_gen_utils.tensor_preloads / run_binary_rows.
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


def main():
    torch.manual_seed(44)
    input_a = torch.randn(32, 32, dtype=torch.bfloat16)
    input_b = torch.randn(32, 32, dtype=torch.bfloat16)

    a_rows = bf16_tile_to_bank_rows(input_a)
    b_rows = bf16_tile_to_bank_rows(input_b)

    preloads = tensor_preloads(A_BASE, a_rows)
    preloads.extend(tensor_preloads(B_BASE, b_rows))

    out_rows = run_binary_rows("sub", a_rows, b_rows)
    checks = tensor_checks(OUT_BASE, out_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

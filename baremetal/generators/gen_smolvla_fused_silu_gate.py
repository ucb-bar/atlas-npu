#!/usr/bin/env python3
"""Generate test vectors for `smolvla_fused_silu_gate.S`.

Tests out = silu(x) = x * sigmoid(x) = x / (1 + exp(-x)) — single
input, single output.

The "gate" in the kernel name is a misnomer inherited from the upstream
coverage tracker phrase "silu + elementwise_mul", where elementwise_mul
is the MLIR primitive opcode of op_b (inner sigmoid*x multiply), NOT a
separate gate operand. Reference at
`.claude/npu-model/smolvla_fused_silu_gate.py` is unambiguous: single-arg
function, op_a = sigmoid(x), op_b = silu(x) = sigmoid(x) * x.

Functionally equivalent to `smolvla_silu`; kept under the
`fused_silu_gate` name for parity with the npu-model program list.
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

X_BASE = 0
OUT_BASE = X_BASE + BEATS_PER_TENSOR

BF16_PLUS_ONE = 0x3F80
BF16_MINUS_ONE = 0xBF80


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
    minus_one_rows = const_rows(BF16_MINUS_ONE, n)
    plus_one_rows = const_rows(BF16_PLUS_ONE, n)

    minus_x = run_binary_rows("mul", x_rows, minus_one_rows)
    exp_neg_x = run_unary_rows("exp", minus_x)
    denom = run_binary_rows("add", exp_neg_x, plus_one_rows)
    sigmoid = run_unary_rows("rcp", denom)
    silu = run_binary_rows("mul", x_rows, sigmoid)

    preloads = tensor_preloads(X_BASE, x_rows)
    checks = tensor_checks(OUT_BASE, silu)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

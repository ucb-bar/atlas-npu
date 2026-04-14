#!/usr/bin/env python3
"""Generate test vectors for `vpu_binary.S`."""

import json

from vpu_gen_utils import (
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    constant_bf16_rows,
    float_to_bf16,
    run_binary_rows,
    tensor_checks,
    tensor_preloads,
)


BEATS_PER_BANK = ROWS_PER_REGISTER
BEATS_PER_TENSOR = ROWS_PER_TENSOR
TIMEOUT = 50000

A_BASE = 0
B_BASE = 64
ADD_BASE = 128
SUB_BASE = 192
MUL_BASE = 256
MIN_BASE = 320
MAX_BASE = 384

A_ROWS = constant_bf16_rows(BEATS_PER_TENSOR, float_to_bf16(2.0))
B_ROWS = constant_bf16_rows(BEATS_PER_TENSOR, float_to_bf16(3.0))


def main():
    preloads = tensor_preloads(A_BASE, A_ROWS)
    preloads.extend(tensor_preloads(B_BASE, B_ROWS))

    checks = []
    for op, base in [
        ("add", ADD_BASE),
        ("sub", SUB_BASE),
        ("mul", MUL_BASE),
        ("pairmin", MIN_BASE),
        ("pairmax", MAX_BASE),
    ]:
        checks.extend(tensor_checks(base, run_binary_rows(op, A_ROWS, B_ROWS)))

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

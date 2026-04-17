#!/usr/bin/env python3
"""Generate test vectors for `vpu_same_src.S`."""

import json

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_TENSOR,
    float_to_bf16,
    run_binary_rows,
    tensor_checks,
    tensor_preloads,
)


TIMEOUT = 50000

A_BASE = 0
ADD_BASE = 64
SUB_BASE = 128
MUL_BASE = 192
MIN_BASE = 256
MAX_BASE = 320

A_ROWS = [
    [
        float_to_bf16(((row_idx * BF16_PER_BEAT) + lane_idx + 1) / 8.0)
        for lane_idx in range(BF16_PER_BEAT)
    ]
    for row_idx in range(ROWS_PER_TENSOR)
]


def main():
    preloads = tensor_preloads(A_BASE, A_ROWS)
    checks = []
    for op, base in [
        ("add", ADD_BASE),
        ("sub", SUB_BASE),
        ("mul", MUL_BASE),
        ("pairmin", MIN_BASE),
        ("pairmax", MAX_BASE),
    ]:
        checks.extend(tensor_checks(base, run_binary_rows(op, A_ROWS, A_ROWS)))

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

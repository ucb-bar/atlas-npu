#!/usr/bin/env python3
"""Generate test vectors for `vpu_col_reduce.S`."""

import json

from vpu_gen_utils import (
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    constant_bf16_rows,
    float_to_bf16,
    run_col_reduce_tensor,
    tensor_checks,
    tensor_preloads,
)


BEATS_PER_BANK = ROWS_PER_REGISTER
BEATS_PER_TENSOR = ROWS_PER_TENSOR
TIMEOUT = 40000

A_ROWS = (
    constant_bf16_rows(BEATS_PER_BANK, float_to_bf16(1.0))
    + constant_bf16_rows(BEATS_PER_BANK, float_to_bf16(2.0))
)


def main():
    preloads = tensor_preloads(0, A_ROWS)

    checks = []
    for op, base in [
        ("csum", 64),
        ("cmin", 128),
        ("cmax", 192),
    ]:
        checks.extend(
            tensor_checks(base, run_col_reduce_tensor(op, A_ROWS, BEATS_PER_TENSOR))
        )

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

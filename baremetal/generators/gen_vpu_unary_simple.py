#!/usr/bin/env python3
"""Generate test vectors for `vpu_unary_simple.S`."""

import json

from vpu_gen_utils import (
    ROWS_PER_REGISTER,
    constant_bf16_rows,
    float_to_bf16,
    run_unary_rows,
    tensor_checks,
    tensor_preloads,
)


BEATS_PER_BANK = ROWS_PER_REGISTER
TIMEOUT = 30000

A_ROWS = (
    constant_bf16_rows(BEATS_PER_BANK, float_to_bf16(-2.0))
    + constant_bf16_rows(BEATS_PER_BANK, float_to_bf16(4.0))
)


def main():
    preloads = tensor_preloads(0, A_ROWS)

    checks = []
    for op, base in [
        ("mov", 64),
        ("relu", 128),
        ("rcp", 192),
        ("square", 256),
        ("cube", 320),
    ]:
        checks.extend(tensor_checks(base, run_unary_rows(op, A_ROWS)))

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

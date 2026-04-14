#!/usr/bin/env python3
"""Generate test vectors for `vpu_unary_math.S`."""

import json

from vpu_gen_utils import (
    ROWS_PER_TENSOR,
    constant_bf16_rows,
    float_to_bf16,
    run_unary_rows,
    tensor_checks,
    tensor_preloads,
)


BEATS_PER_TENSOR = ROWS_PER_TENSOR
TIMEOUT = 60000

INPUT_ROWS = constant_bf16_rows(BEATS_PER_TENSOR, float_to_bf16(1.0))

OPS = [
    ("exp", 64),
    ("exp2", 128),
    ("sin", 192),
    ("cos", 256),
    ("tanh", 320),
    ("log", 384),
    ("sqrt", 448),
]


def main():
    preloads = tensor_preloads(0, INPUT_ROWS)

    checks = []
    for op, base in OPS:
        checks.extend(tensor_checks(base, run_unary_rows(op, INPUT_ROWS)))

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

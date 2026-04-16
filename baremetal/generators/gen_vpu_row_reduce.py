#!/usr/bin/env python3
"""Generate test vectors for `vpu_row_reduce.S`."""

import json

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_TENSOR,
    float_to_bf16,
    repeat_bf16_row,
    run_row_reduce_tensor,
    tensor_checks,
    tensor_preloads,
)


BEATS_PER_TENSOR = ROWS_PER_TENSOR
TIMEOUT = 30000

INPUT_ROW = [float_to_bf16(lane + 1.0) for lane in range(BF16_PER_BEAT)]
INPUT_ROWS = repeat_bf16_row(INPUT_ROW, BEATS_PER_TENSOR)


def main():
    preloads = tensor_preloads(0, INPUT_ROWS)

    checks = []
    for op, base in [("rsum", 64), ("rmin", 128), ("rmax", 192)]:
        checks.extend(tensor_checks(base, run_row_reduce_tensor(op, INPUT_ROWS)))

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

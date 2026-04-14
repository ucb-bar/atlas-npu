#!/usr/bin/env python3
"""Generate test vectors for `vpu_pack.S`."""

import json

from vpu_gen_utils import (
    ROWS_PER_REGISTER,
    constant_bf16_rows,
    float_to_bf16,
    fp8_checks,
    run_fp8_pack_rows,
    run_fp8_unpack_rows,
    tensor_checks,
    tensor_preloads,
)


BEATS_PER_FP8_TENSOR = ROWS_PER_REGISTER
TIMEOUT = 30000

LOW_ROWS = constant_bf16_rows(BEATS_PER_FP8_TENSOR, float_to_bf16(2.0))
HIGH_ROWS = constant_bf16_rows(BEATS_PER_FP8_TENSOR, float_to_bf16(2.0))
INPUT_ROWS = LOW_ROWS + HIGH_ROWS


def main():
    preloads = tensor_preloads(0, INPUT_ROWS)
    checks = []

    packed_rows = run_fp8_pack_rows(LOW_ROWS, HIGH_ROWS, scale_e8m0=0x80)
    checks.extend(fp8_checks(64, packed_rows))

    unpacked_rows = run_fp8_unpack_rows(packed_rows, scale_e8m0=0x81)
    checks.extend(tensor_checks(128, unpacked_rows))

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

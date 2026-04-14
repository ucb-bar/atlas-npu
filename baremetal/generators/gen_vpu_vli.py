#!/usr/bin/env python3
"""Generate test vectors for `vpu_vli.S`."""

import json

from vpu_gen_utils import ROWS_PER_REGISTER, float_to_bf16, run_vli_rows, tensor_checks


BEATS_PER_REGISTER = ROWS_PER_REGISTER
TIMEOUT = 20000

BF16_1 = float_to_bf16(1.0)
BF16_2 = float_to_bf16(2.0)
BF16_3 = float_to_bf16(3.0)
BF16_4 = float_to_bf16(4.0)
BF16_5 = float_to_bf16(5.0)


def main():
    preloads = []
    checks = []
    sentinel_rows = run_vli_rows("vliAll", BF16_5)

    checks.extend(tensor_checks(0, run_vli_rows("vliAll", BF16_1)))
    checks.extend(tensor_checks(32, sentinel_rows))

    checks.extend(tensor_checks(64, run_vli_rows("vliRow", BF16_2)))
    checks.extend(tensor_checks(96, sentinel_rows))

    checks.extend(tensor_checks(128, run_vli_rows("vliCol", BF16_3)))
    checks.extend(tensor_checks(160, sentinel_rows))

    checks.extend(tensor_checks(192, run_vli_rows("vliOne", BF16_4)))
    checks.extend(tensor_checks(224, sentinel_rows))

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

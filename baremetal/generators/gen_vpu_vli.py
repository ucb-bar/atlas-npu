#!/usr/bin/env python3
"""Generate test vectors for `vpu_vli.S`."""

import json

from vpu_gen_utils import (
    float_to_bf16,
    run_vli_registers,
    tensor_checks,
)


TIMEOUT = 20000

BF16_1 = float_to_bf16(1.0)
BF16_2 = float_to_bf16(2.0)
BF16_3 = float_to_bf16(3.0)
BF16_4 = float_to_bf16(4.0)

EVEN_BANK = 4
ODD_BANK = 5


def main():
    preloads = []
    checks = []

    all_regs = run_vli_registers("vliAll", BF16_1, dst_bank=EVEN_BANK)
    checks.extend(tensor_checks(0, all_regs[EVEN_BANK]))
    checks.extend(tensor_checks(32, all_regs[ODD_BANK]))

    row_regs = run_vli_registers("vliRow", BF16_2, dst_bank=EVEN_BANK)
    checks.extend(tensor_checks(64, row_regs[EVEN_BANK]))
    checks.extend(tensor_checks(96, row_regs[ODD_BANK]))

    zero_regs = run_vli_registers("vliAll", 0, dst_bank=EVEN_BANK)
    col_regs = run_vli_registers("vliCol", BF16_3, dst_bank=ODD_BANK)
    checks.extend(tensor_checks(128, zero_regs[EVEN_BANK]))
    checks.extend(tensor_checks(160, col_regs[ODD_BANK]))

    one_regs = run_vli_registers("vliOne", BF16_4, dst_bank=ODD_BANK)
    checks.extend(tensor_checks(192, zero_regs[EVEN_BANK]))
    checks.extend(tensor_checks(224, one_regs[ODD_BANK]))

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

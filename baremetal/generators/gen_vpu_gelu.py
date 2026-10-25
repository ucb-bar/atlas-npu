#!/usr/bin/env python3
"""Generate test vectors for `vpu_gelu.S`: vgelu.bf16 and the eight .bf16 instructions of its
definition give the same results on all 65,536 BF16 codes."""

import json

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_TENSOR,
    run_binary_rows,
    run_unary_rows,
    tensor_checks,
    tensor_preloads,
)

TIMEOUT = 400000
TILES = (1 << 16) // (ROWS_PER_TENSOR * BF16_PER_BEAT)
IN_BASE, GELU_BASE, SEQ_BASE = 0, 0x20000 // 32, 0x40000 // 32   # 32-byte DRAM words
C1, C2, ONE, HALF = 0x3D37, 0x3F4C, 0x3F80, 0x3F00               # 0.044715, sqrt(2/pi), 1, 0.5


def sequence(x):
    """vcube; vmul c1; vadd x; vmul c2; vtanh; vadd 1; vmul x; vmul 0.5, row by row."""
    def const(bits):
        return [[bits] * BF16_PER_BEAT] * len(x)
    t = run_unary_rows("cube", x)
    t = run_binary_rows("mul", t, const(C1))
    t = run_binary_rows("add", t, x)
    t = run_binary_rows("mul", t, const(C2))
    t = run_unary_rows("tanh", t)
    t = run_binary_rows("add", t, const(ONE))
    y = run_binary_rows("mul", x, t)
    return run_binary_rows("mul", y, const(HALF))


def main():
    x = [[row * BF16_PER_BEAT + lane for lane in range(BF16_PER_BEAT)]
         for row in range(TILES * ROWS_PER_TENSOR)]
    y = sequence(x)
    checks = tensor_checks(GELU_BASE, y) + tensor_checks(SEQ_BASE, y)
    print(json.dumps({"dram_preloads": tensor_preloads(IN_BASE, x), "dram_checks": checks, "timeout": TIMEOUT},
                     indent=2))


if __name__ == "__main__":
    main()

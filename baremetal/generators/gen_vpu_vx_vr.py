#!/usr/bin/env python3
"""Generate test vectors for `vpu_vx_vr.S`: each .vx and .vr form equals its .bf16 instruction
with the broadcast operand as the second tile."""

import json
import random

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    pack_u16_le,
    run_binary_rows,
    tensor_checks,
    tensor_preloads,
)

TIMEOUT = 50000
A_BASE, B_BASE, SCALAR_BASE, OUT_BASE = 0, 64, 128, 256    # 32-byte DRAM words
SCALAR = 0x3FC0                                             # 1.5, in bits 31:16 of the loaded word
LUI_SCALAR = 0xC0A0                                         # -5.0, from lui x8, 0xc0a00
SPECIALS = [0x0000, 0x8000, 0x0001, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x3FC0, 0xBFC0, 0x7F7F]


def tile(rng):
    return [[rng.choice(SPECIALS) if rng.random() < 0.2 else rng.getrandbits(16) for _ in range(BF16_PER_BEAT)]
            for _ in range(ROWS_PER_TENSOR)]


def main():
    rng = random.Random(7)
    a, b = tile(rng), tile(rng)
    scalar = lambda s: [[s] * BF16_PER_BEAT] * ROWS_PER_TENSOR
    # Row 0 of T(vs2): row 0 of m[vs2] for the rows of m[vs1], row 0 of m[vs2+1] for those of m[vs1+1].
    row0 = [b[0]] * ROWS_PER_REGISTER + [b[ROWS_PER_REGISTER]] * ROWS_PER_REGISTER
    cases = [("add", scalar(SCALAR)), ("sub", scalar(LUI_SCALAR)), ("mul", scalar(0)),
             ("pairmin", scalar(SCALAR)), ("pairmax", scalar(SCALAR)),
             ("add", row0), ("sub", row0), ("mul", row0), ("pairmin", row0), ("pairmax", row0)]
    preloads = tensor_preloads(A_BASE, a) + tensor_preloads(B_BASE, b)
    preloads.append({"word_offset": SCALAR_BASE, "data": pack_u16_le([0xBEEF, SCALAR] + [0] * 14)})
    checks = []
    for k, (op, second) in enumerate(cases):
        checks.extend(tensor_checks(OUT_BASE + k * ROWS_PER_TENSOR, run_binary_rows(op, a, second)))
    print(json.dumps({"dram_preloads": preloads, "dram_checks": checks, "timeout": TIMEOUT}, indent=2))


if __name__ == "__main__":
    main()

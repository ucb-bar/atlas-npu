#!/usr/bin/env python3
"""Generate test vectors for `vpu_rows.S`: VPU instructions at row counts M (vsetm).

An instruction of M rows writes rows 0..M-1 of each register of its destination (vli.col and
vli.one: of m[vd]) with the rows of its 32-row result; the other rows keep their values. A column
reduction reduces rows 0..M-1 of both registers of its source pair. vpack and vunpack take all
rows whatever M is.
"""

import json
import random

from gen_vpu_gelu import sequence as gelu
from vpu_gen_utils import (
    BF16_PER_BEAT,
    FP8_PER_BEAT,
    ROWS_PER_REGISTER,
    fp8_checks,
    pack_u8_le,
    run_binary_rows,
    run_col_reduce_tensor,
    run_fp8_pack_rows,
    run_fp8_unpack_rows,
    run_row_reduce_tensor,
    run_unary_rows,
    run_vli_registers,
    tensor_checks,
    tensor_preloads,
)

TIMEOUT = 60000
R = ROWS_PER_REGISTER
# 32-byte DRAM words: A = T(m0), B = T(m2), the E4M3 register m4, the start values of m8..m39,
# then the results m8..m39.
A_BASE, B_BASE, FP8_BASE, START_BASE, OUT_BASE = 0, 64, 128, 160, 2048
SCALAR = 0x3FC0                                             # 1.5: lui x7, 0x3fc00


def tile(rng):
    return [[rng.getrandbits(16) & 0x8FFF | 0x3000 for _ in range(BF16_PER_BEAT)] for _ in range(2 * R)]


def pair(rows, dst):
    """A 64-row tile result (rows 0..31 of m[dst], then of m[dst+1]) by register."""
    return {dst: rows[:R], dst + 1: rows[R:]}


def main():
    rng = random.Random(23)
    a, b = tile(rng), tile(rng)
    fp8 = [[rng.choice([c for c in range(256) if c & 0x7F != 0x7F]) for _ in range(FP8_PER_BEAT)] for _ in range(R)]
    start = {reg: [[rng.getrandbits(16) for _ in range(BF16_PER_BEAT)] for _ in range(R)] for reg in range(8, 40)}

    slots = [[row[2 * i] | row[2 * i + 1] << 8 for i in range(BF16_PER_BEAT)] for row in fp8]   # 16-bit slots
    row0 = [b[0]] * R + [b[R]] * R
    scalar = [[SCALAR] * BF16_PER_BEAT] * (2 * R)

    def col(op, src, m):
        return run_col_reduce_tensor(op, src[:m] + src[R:R + m], 2 * R)

    def vli(op, imm, dst):
        return run_vli_registers(op, imm, dst_bank=dst)

    # (M, result by register); M = 32 for vpack and vunpack, which take all rows.
    results = [
        (3, pair(run_binary_rows("add", a, b), 8)),                 # vadd.bf16 m8, m0, m2
        (1, pair(run_binary_rows("mul", a, scalar), 10)),           # vmul.bf16.vx m10, m0, x7
        (5, pair(run_binary_rows("sub", a, row0), 12)),             # vsub.bf16.vr m12, m0, m2
        (7, pair(run_unary_rows("exp", a), 14)),                    # vexp.bf16 m14, m0
        (2, pair(gelu(b), 16)),                                     # vgelu.bf16 m16, m2
        (6, pair(run_row_reduce_tensor("rsum", a), 18)),            # vredsum.row.bf16 m18, m0
        (3, pair(col("cmax", a, 3), 20)),                           # vredmax.bf16 m20, m0
        (4, pair(col("csum", b, 4), 22)),                           # vredsum.bf16 m22, m2
        (4, vli("vliAll", 0x3F80, 24)),                             # vli.all m24, 0x3f80
        (2, vli("vliCol", 0x4000, 26)),                             # vli.col m26, 0x4000
        (3, vli("vliRow", 0xBF80, 28)),                             # vli.row m28, 0xbf80
        (5, vli("vliOne", 0x4040, 30)),                             # vli.one m30, 0x4040
        (32, {32: run_fp8_pack_rows(a[:R], a[R:], 127)}),           # vpack.bf16.fp8 m32, m0, e0 at M = 3
        (32, pair(run_fp8_unpack_rows(slots, 127), 34)),            # vunpack.fp8.bf16 m34, m4, e0 at M = 3
        (31, pair(b, 36)),                                          # vmov m36, m2
        (2, pair(run_binary_rows("pairmin", a, b), 38)),            # vminimum.bf16 m38, m0, m2
    ]
    regs = {reg: [list(row) for row in rows] for reg, rows in start.items()}
    for m, written in results:
        for reg, rows in written.items():
            regs[reg][:m] = rows[:m]

    preloads = tensor_preloads(A_BASE, a) + tensor_preloads(B_BASE, b)
    preloads += [{"word_offset": FP8_BASE + i, "data": pack_u8_le(row)} for i, row in enumerate(fp8)]
    for reg, rows in start.items():
        preloads += tensor_preloads(START_BASE + R * (reg - 8), rows)
    checks = []
    for reg, rows in regs.items():
        base = OUT_BASE + R * (reg - 8)
        checks += fp8_checks(base, rows) if reg == 32 else tensor_checks(base, rows)
    print(json.dumps({"dram_preloads": preloads, "dram_checks": checks, "timeout": TIMEOUT}, indent=2))


if __name__ == "__main__":
    main()

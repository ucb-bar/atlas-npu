#!/usr/bin/env python3
"""gen_mxu1_rows.py — golden data for assembly/mxu1_rows.S: mxu1 at row counts M (vsetm).

A command of M rows moves rows 0..M-1 of its accumulator and m-registers (a weight push: all rows);
the other rows keep their values. The matmul chains at M = 1 and 2 run back to back on one
accumulator and on alternating ones: each matmul must read the rows the previous one wrote.

DRAM beats (one m-register = 32): m0-m2 A (E4M3), m3 W, C = T(m4), D = T(m6), the start values of
m8..m43 at 256 + 32 (r - 8); the results m8..m43 at 2048 + 32 (r - 8).
"""
import numpy as np

from gen_utils import checks_from_words_packed, emit_test_data, preloads_from_words_packed
from mreg_layout import bf16_pair, fp32_quad, words
from npu_model.configs import cvt, fp8_mma

rng = np.random.RandomState(4093)


def e4m3_tiles(n):
    """n m-registers of E4M3 codes with exponent fields 0..10, both signs (no NaN)."""
    shape = (n, 32, 32)
    sign, field, mantissa = rng.randint(0, 2, shape), rng.randint(0, 11, shape), rng.randint(0, 8, shape)
    return (sign << 7 | field << 3 | mantissa).astype(np.uint8)


def bf16_tile():
    return cvt.bf16_rne((rng.standard_normal((32, 32)) * 4).astype(np.float32).view(np.uint32))


def reg(x):
    """An m-register as 32 rows of 32 bytes."""
    return np.ascontiguousarray(x).view(np.uint8).reshape(32, 32)


def pair_tile(lo, hi):
    """The BF16 tile of {m[d], m[d+1]}."""
    return np.concatenate([lo.view(np.uint16), hi.view(np.uint16)], axis=1)


def quad_tile(regs):
    """The FP32 tile of m[d..d+3]."""
    return np.concatenate([r.view(np.uint32) for r in regs], axis=1)


a, w = e4m3_tiles(3), e4m3_tiles(1)[0]
c, d = bf16_tile(), bf16_tile()
start = {r: rng.randint(0, 256, (32, 32)).astype(np.uint8) for r in range(8, 44)}
start[28], start[29] = map(reg, bf16_pair(bf16_tile()))           # vmatpush.acc.bf16 source
start[30] = e4m3_tiles(1)[0]                                       # vmatpush.acc.fp8 source
regs = {r: x.copy() for r, x in start.items()}


def matmul(acc, k, m, accumulate=True):
    rows = acc.copy()
    rows[:m] = fp8_mma.matmul(a[k], w, acc if accumulate else np.zeros_like(acc))[:m]
    return rows


def write(dst, tiles, m):
    for i, t in enumerate(tiles):
        regs[dst + i][:m] = reg(t)[:m]


acc0, acc1 = c.astype(np.uint32) << 16, d.astype(np.uint32) << 16
for k in (0, 1, 2):                                                # M = 1: acc0 += A_k W^T
    acc0 = matmul(acc0, k, 1)
acc1 = matmul(acc1, 0, 2)                                          # M = 2, alternating
acc0 = matmul(acc0, 1, 2)
acc1 = matmul(acc1, 2, 2)
write(8, bf16_pair(cvt.bf16_rne(acc0)), 3)                        # M = 3
write(10, [cvt.e4m3_rn_satfinite(acc1, 120)], 3)
write(12, bf16_pair(cvt.bf16_rne(acc1, 131)), 5)                  # M = 5
write(16, fp32_quad(acc0), 2)                                      # M = 2
write(20, fp32_quad(acc0), 32)
write(24, fp32_quad(acc1), 32)
acc0[:4] = (pair_tile(start[28], start[29]).astype(np.uint32) << 16)[:4]   # M = 4
acc1[:6] = cvt.e4m3_to_f32(start[30])[:6]                         # M = 6
acc1[:3] = quad_tile([start[r] for r in range(32, 36)])[:3]       # M = 3
acc0 = matmul(acc0, 1, 7, accumulate=False)                        # M = 7
write(36, fp32_quad(acc0), 32)
write(40, fp32_quad(acc1), 32)

preloads = words([*a, w, *bf16_pair(c), *bf16_pair(d), *(start[r] for r in range(8, 44))])
checks = words([regs[r] for r in range(8, 44)])
emit_test_data(preloads_from_words_packed(0, preloads), checks_from_words_packed(2048, checks))

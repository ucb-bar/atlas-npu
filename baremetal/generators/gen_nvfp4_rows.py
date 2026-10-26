#!/usr/bin/env python3
"""gen_nvfp4_rows.py — golden data for assembly/nvfp4_rows.S: the NVFP4 MXU at row counts M.

A command of M rows moves rows 0..M-1 of its accumulator and m-registers (a weight push: all rows);
the other rows keep their values. The matmul chains at M = 1, 5 and 7 are below the matmul spacing
of the NVFP4 MXU (8 cycles): each matmul must read the rows the previous one wrote. vload.scale and
vstore.scale move all 128 bytes of scales at M = 3.

DRAM beats: m_r at 32 r for r = 0..43 (m0-m2 A, m3 W, T(m4) = C, the start values of m8..m43),
the scales of m0..m3 at 1408 + 4 r; the results m8..m43 at 2048 + 32 (r - 8), then the scales of
m1 and m5 (= those of m2).
"""
import random
import struct

import numpy as np

from nvfp4_common import Image, bf16_regs, cvt, f32, fp32_regs, matmul, rand_scales, rand_tile

rng = random.Random(4094)
tiles = [b"".join(rand_tile(rng)) for _ in range(4)]                 # m0-m2 A, m3 W
scales = [rand_scales(rng) for _ in range(4)]
c = [[int(cvt.bf16_rne(f32(rng.uniform(-8.0, 8.0)))) << 16 for _ in range(32)] for _ in range(32)]
start = {r: bytes(rng.randrange(256) for _ in range(1024)) for r in range(4, 44)}
start[4], start[5] = bf16_regs(c)
start[28], start[29] = bf16_regs([[int(cvt.bf16_rne(f32(rng.uniform(-8.0, 8.0)))) << 16 for _ in range(32)]
                                  for _ in range(32)])
regs = {r: bytearray(start[r]) for r in range(8, 44)}

rows = lambda data: [data[32 * i:32 * i + 32] for i in range(32)]
a = [rows(t) for t in tiles[:3]]
w, sw = rows(tiles[3]), scales[3]


def mm(acc, k, m, accumulate=True):
    c_rows = acc[:m] if accumulate else [[0] * 32 for _ in range(m)]
    return matmul(a[k][:m], scales[k][:m], w, sw, c_rows) + acc[m:]


def write(dst, data, m):
    for i, reg in enumerate(data):
        regs[dst + i][:32 * m] = reg[:32 * m]


def bf16_scaled(tile, e):
    bits = cvt.bf16_rne(np.array(tile, dtype=np.uint32), e)
    return [bits[:, 16 * h:16 * h + 16].astype("<u2").tobytes() for h in range(2)]


def words(reg):
    return [list(struct.unpack("<8I", reg[32 * i:32 * i + 32])) for i in range(32)]


acc0 = acc1 = c
for k in (0, 1, 2):                                                  # M = 1: acc0 += A_k W^T
    acc0 = mm(acc0, k, 1)
acc1 = mm(acc1, 0, 5)                                                # M = 5, alternating
acc0 = mm(acc0, 1, 5)
acc1 = mm(acc1, 2, 5)
acc1 = mm(acc1, 1, 7, accumulate=False)                              # M = 7
acc1 = mm(acc1, 2, 7)
write(8, bf16_regs(acc0), 3)                                         # M = 3
write(12, bf16_scaled(acc1, 131), 6)                                 # M = 6
write(16, fp32_regs(acc0), 2)                                        # M = 2
write(20, fp32_regs(acc0), 32)
write(24, fp32_regs(acc1), 32)
lo, hi = (np.frombuffer(start[r], dtype="<u2").reshape(32, 16) for r in (28, 29))
acc0 = [[int(x) << 16 for x in np.concatenate([lo[i], hi[i]])] for i in range(4)] + acc0[4:]          # M = 4
quad = [words(start[r]) for r in range(32, 36)]
acc1 = [[quad[j // 8][i][j % 8] for j in range(32)] for i in range(3)] + acc1[3:]                    # M = 3
write(36, fp32_regs(acc0), 32)
write(40, fp32_regs(acc1), 32)

img = Image()
img.preload(0, b"".join(tiles) + b"".join(start[r] for r in range(4, 44)))
img.preload(1408, b"".join(b"".join(s) for s in scales))
img.check(2048, b"".join(bytes(regs[r]) for r in range(8, 44)) + b"".join(scales[1] + scales[2]))
img.dump()

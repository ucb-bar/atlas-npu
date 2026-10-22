#!/usr/bin/env python3
"""gen_nvfp4_fp32_specials.py — golden data for assembly/nvfp4_fp32_specials.S.

acc0 = C + A @ W^T with C pushed as FP32 (vmatpush.acc.fp32.nv) and holding infinities, NaN,
-0, subnormals and the largest finite value; the block scales include zero, subnormal, NaN
(0x7F) and the largest scale; two A rows are all zero (C passes through, -0 becomes +0).
Popped as FP32 (vmatpop.fp32.acc.nv) and compared bit for bit.

DRAM beats: A data 0-31, A scales 32-35, W data 40-71, W scales 72-75, C 80-207 (4 KiB),
result 224-351.
"""
import random

from nvfp4_common import Image, fp32_regs, matmul, rand_fp32_tile, rand_scales, rand_tile

rng = random.Random(4004)
a, sa = rand_tile(rng, zero_rows={2, 19}), rand_scales(rng, specials=True)
w, sw = rand_tile(rng), rand_scales(rng, specials=True)
c = rand_fp32_tile(rng, specials=True)
c[2][0] = 0x8000_0000  # -0 on a zero row: becomes +0

img = Image()
img.preload(0, b"".join(a)); img.preload(32, b"".join(sa))
img.preload(40, b"".join(w)); img.preload(72, b"".join(sw))
img.preload(80, b"".join(fp32_regs(c)))
img.check(224, b"".join(fp32_regs(matmul(a, sa, w, sw, c))))
img.dump()

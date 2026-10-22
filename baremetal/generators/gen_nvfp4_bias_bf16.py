#!/usr/bin/env python3
"""gen_nvfp4_bias_bf16.py — golden data for assembly/nvfp4_bias_bf16.S.

acc1 = bias + A @ W^T: vmatpush.acc.bf16.nv loads a 32 x 32 BF16 bias (exact in FP32),
vmatmul.acc.nv accumulates, vmatpop.bf16.acc.nv rounds to BF16 (nearest even).

DRAM beats: A data 0-31, A scales 32-35, W data 40-71, W scales 72-75, bias 80-143 (2 KiB:
columns 0-15 then 16-31), result 160-223.
"""
import random
import struct

from nvfp4_common import Image, bf16_regs, f32, matmul, nvfp4, rand_scales, rand_tile

rng = random.Random(4003)
a, sa = rand_tile(rng), rand_scales(rng)
w, sw = rand_tile(rng), rand_scales(rng)
bias = [[nvfp4.f32_to_bf16(f32(rng.uniform(-8.0, 8.0))) for _ in range(32)] for _ in range(32)]

bias_regs = [b"".join(struct.pack("<16H", *row[16 * h:16 * h + 16]) for row in bias) for h in range(2)]
c = [[x << 16 for x in row] for row in bias]

img = Image()
img.preload(0, b"".join(a)); img.preload(32, b"".join(sa))
img.preload(40, b"".join(w)); img.preload(72, b"".join(sw))
img.preload(80, b"".join(bias_regs))
img.check(160, b"".join(bf16_regs(matmul(a, sa, w, sw, c))))
img.dump()

#!/usr/bin/env python3
"""gen_nvfp4_fp32_spill.py — golden data for assembly/nvfp4_fp32_spill.S.

acc = C + A0 W0^T + A1 W1^T + A2 W2^T + A3 W3^T: a chain of four NVFP4 MMAs (K = 64 each) from a
BF16 C. The spill is the chain after two steps.

DRAM beats: A data 0-127, A scales 128-143, W data 144-271, W scales 272-287, C 288-351,
X 352-415; results 416-863: BF16_RNE(X + A2 W2^T) (m20, m21), the spill (m24..m27), acc0
(m28..m31) and acc1 (m32..m35).
"""
import random

from nvfp4_common import Image, bf16_regs, cvt, f32, fp32_regs, matmul, rand_scales, rand_tile

rng = random.Random(4008)
a = [rand_tile(rng) for _ in range(4)]; sa = [rand_scales(rng) for _ in range(4)]
w = [rand_tile(rng) for _ in range(4)]; sw = [rand_scales(rng) for _ in range(4)]
# C and X: FP32 tiles of BF16 values (BF16 bits << 16), pushed with vmatpush.acc.bf16.nv.
c, x = ([[int(cvt.bf16_rne(f32(rng.uniform(-8.0, 8.0)))) << 16 for _ in range(32)] for _ in range(32)]
        for _ in range(2))

acc = c
for k in (0, 1):
    acc = matmul(a[k], sa[k], w[k], sw[k], acc)
spill = acc
other = matmul(a[2], sa[2], w[2], sw[2], x)
for k in (2, 3):
    acc = matmul(a[k], sa[k], w[k], sw[k], acc)

img = Image()
img.preload(0, b"".join(b"".join(t) for t in a))
img.preload(128, b"".join(b"".join(s) for s in sa))
img.preload(144, b"".join(b"".join(t) for t in w))
img.preload(272, b"".join(b"".join(s) for s in sw))
img.preload(288, b"".join(bf16_regs(c) + bf16_regs(x)))
img.check(416, b"".join(bf16_regs(other) + fp32_regs(spill) + fp32_regs(acc) * 2))
img.dump()

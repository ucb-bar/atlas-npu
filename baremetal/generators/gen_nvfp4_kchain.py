#!/usr/bin/env python3
"""gen_nvfp4_kchain.py — golden data for assembly/nvfp4_kchain.S.

acc0 = sum over k = 0..3 of A_k @ W_k^T (K = 256): one vmatmul.nv and three vmatmul.acc.nv. Each
step equals NVIDIA's m16n8k64 MMA with C from the previous step, so the chain matches a chain of
MMAs bit for bit, apart from the known gap in docs/nvfp4-mxu.md.

DRAM beats: A data 0-127 (4 tiles), A scales 128-143, W data 144-271, W scales 272-287,
result 288-415.
"""
import random

from nvfp4_common import Image, fp32_regs, matmul, rand_scales, rand_tile

rng = random.Random(4002)
a = [rand_tile(rng) for _ in range(4)]; sa = [rand_scales(rng) for _ in range(4)]
w = [rand_tile(rng) for _ in range(4)]; sw = [rand_scales(rng) for _ in range(4)]

acc = None
for k in range(4):
    acc = matmul(a[k], sa[k], w[k], sw[k], acc)

img = Image()
img.preload(0, b"".join(b"".join(t) for t in a))
img.preload(128, b"".join(b"".join(s) for s in sa))
img.preload(144, b"".join(b"".join(t) for t in w))
img.preload(272, b"".join(b"".join(s) for s in sw))
img.check(288, b"".join(fp32_regs(acc)))
img.dump()

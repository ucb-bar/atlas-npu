#!/usr/bin/env python3
"""gen_nvfp4_matmul.py — golden data for assembly/nvfp4_matmul.S.

acc0 = A @ W^T, A and W 32 x 64 NVFP4 tiles with random UE4M3 block scales (zeros, subnormals and
the largest scale included), popped as FP32 and compared bit for bit.

DRAM beats: A data 0-31, A scales 32-35, W data 40-71, W scales 72-75, result 128-255 (4 KiB).
"""
import random

from nvfp4_common import Image, fp32_regs, matmul, rand_scales, rand_tile

rng = random.Random(4001)
a, sa = rand_tile(rng, zero_rows={7}), rand_scales(rng)
w, sw = rand_tile(rng, zero_rows={3}), rand_scales(rng)
sa[5] = bytes([0x00, 0x01, 0x7E, 0x07])  # zero, subnormal, largest, subnormal scales on one row

img = Image()
img.preload(0, b"".join(a)); img.preload(32, b"".join(sa))
img.preload(40, b"".join(w)); img.preload(72, b"".join(sw))
img.check(128, b"".join(fp32_regs(matmul(a, sa, w, sw))))
img.dump()

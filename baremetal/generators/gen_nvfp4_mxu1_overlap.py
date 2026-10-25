#!/usr/bin/env python3
"""gen_nvfp4_mxu1_overlap.py — golden data for assembly/nvfp4_mxu1_overlap.S.

The NV MXU and mxu1 run the same program at the same time:
  nv acc0   = C_nv + A_nv @ W_nv^T (FP32 push)    popped as FP32
  nv acc1   = A_nv @ W_nv^T                       popped as FP32
  mxu1 acc0 = C_1 + A_1 @ W_1^T (E4M3 push)       popped as BF16
  mxu1 acc1 = A_1 @ W_1^T                         popped as E4M3 at e = 124 (x 2^-3)
mxu1 data are E4M3 codes of magnitude at most 8 (subnormals included), so the E4M3 pop does not
saturate. The arithmetic is npu_model's: configs/nvfp4.py (through nvfp4_common), fp8_mma.py and
cvt.py.

DRAM beats: NV A data 0-31, A scales 32-35, W data 40-71, W scales 72-75, C_nv 128-255 (4 KiB),
A_1 256-287, W_1 288-319, C_1 320-351; results 512-863: nv acc0 (4 KiB), mxu1 acc0 (2 KiB),
nv acc1 (4 KiB), mxu1 acc1 (1 KiB).
"""
import random

import numpy as np
from npu_model.configs import cvt, fp8_mma

from nvfp4_common import Image, fp32_regs, matmul, rand_fp32_tile, rand_scales, rand_tile

rng = random.Random(4011)
a, sa = rand_tile(rng), rand_scales(rng)
w, sw = rand_tile(rng), rand_scales(rng)
c_nv = rand_fp32_tile(rng)


def e4m3_tile() -> np.ndarray:
    """32 x 32 E4M3 codes, both signs, exponent field 0-10 (|x| <= 8)."""
    return np.array([[rng.randrange(2) << 7 | rng.randrange(11) << 3 | rng.randrange(8)
                      for _ in range(32)] for _ in range(32)], np.uint8)


a1, w1, c1 = e4m3_tile(), e4m3_tile(), e4m3_tile()
acc0 = fp8_mma.matmul(a1, w1, cvt.e4m3_to_f32(c1))
acc1 = fp8_mma.matmul(a1, w1, np.zeros((32, 32), np.uint32))
bf16 = cvt.bf16_rne(acc0)

img = Image()
img.preload(0, b"".join(a)); img.preload(32, b"".join(sa))
img.preload(40, b"".join(w)); img.preload(72, b"".join(sw))
img.preload(128, b"".join(fp32_regs(c_nv)))
img.preload(256, a1.tobytes()); img.preload(288, w1.tobytes()); img.preload(320, c1.tobytes())
img.check(512, b"".join(fp32_regs(matmul(a, sa, w, sw, c_nv))))
img.check(640, bf16[:, :16].astype("<u2").tobytes() + bf16[:, 16:].astype("<u2").tobytes())
img.check(704, b"".join(fp32_regs(matmul(a, sa, w, sw))))
img.check(832, cvt.e4m3_rn_satfinite(acc1, 124).astype(np.uint8).tobytes())
img.dump()

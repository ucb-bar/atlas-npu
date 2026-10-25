#!/usr/bin/env python3
"""gen_mxu1_scaled_pop.py — golden data for assembly/mxu1_scaled_pop.S (and nvfp4_scaled_pop.S).

acc1 is an FP32 tile, pushed with the FP32 push: the values of npu-model tests/test_mxu_pops.py,
FP32 values of every class, every E4M3 value, values whose BF16 at small e is subnormal, and random
bit patterns. The program pops it with the scaled BF16 pop at each of SCALES, with the plain BF16
pop and with the FP32 pop. Goldens are npu_model's: cvt.bf16_rne(acc, e), one rounding of
acc * 2^(e - 127), and the bit copy.

DRAM beats: tile 0-127 (m4..m7, register q holds columns 8q..8q+7), results 128-767 (m8..m27).
"""
import numpy as np

from gen_utils import checks_from_words_packed, emit_test_data, preloads_from_words_packed
from mreg_layout import bf16_pair, fp32_quad, words
from npu_model.configs import cvt

SCALES = (0, 1, 126, 127, 128, 254, 255)
DIRECTED = [
    0x3F80_0000, 0xBFC0_0000, 0x7F7F_FFFF, 0x3F80_8000, 0x3F81_8000, 0x0000_0001, 0x007F_FFFF,
    0x0040_0000, 0x8000_0001, 0x0060_0000, 0x00A0_0000, 0x3CBF_FFC0, 0x0000_0000, 0x8000_0000,
    0x0080_0000, 0x8080_0000, 0xFF7F_FFFF, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0000,
    0x7FC1_2345, 0xFFFF_FFFF, 0x7F80_0001, 0xFF80_0001, 0x7FBF_FFFF,
]

rng = np.random.RandomState(4005)
e4m3 = cvt.e4m3_to_f32(np.arange(256, dtype=np.uint8))
# BF16 subnormal results near e = 0: exponent fields 118..135 times 2^-127.
near = rng.randint(0, 2, 300) << 31 | rng.randint(118, 136, 300) << 23 | rng.randint(0, 1 << 23, 300)
fill = 1024 - len(DIRECTED) - len(e4m3) - len(near)
parts = [DIRECTED, e4m3, near, rng.randint(0, 1 << 32, fill, dtype=np.uint64)]
tile = np.concatenate([np.asarray(p).astype(np.uint32) for p in parts]).reshape(32, 32)


quad = fp32_quad(tile)
results = [r for e in SCALES for r in bf16_pair(cvt.bf16_rne(tile, e))] + bf16_pair(cvt.bf16_rne(tile)) + quad
emit_test_data(preloads_from_words_packed(0, words(quad)), checks_from_words_packed(128, words(results)))

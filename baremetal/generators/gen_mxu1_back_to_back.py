#!/usr/bin/env python3
"""gen_mxu1_back_to_back.py — golden data for assembly/mxu1_back_to_back.S.

acc0 = C0 + A0 W0^T + A2 W1^T + A4 W2^T with C0 the FP32 values of E4M3 codes (FP8 push), and
acc1 = C1 + A1 W1^T + A3 W2^T + A5 W3^T with a BF16 C1: chains of npu_model fp8_mma steps. The
pops give cvt.bf16_rne and cvt.e4m3_rn_satfinite (e = 123 and 121) of each accumulator.

DRAM beats: A0..A5 0-191, W0..W3 192-319, C0 320-351, C1 352-415; results 512-703: acc0 BF16
(m16, m17), acc0 FP8 (m18), acc1 BF16 (m20, m21), acc1 FP8 (m22).
"""
import numpy as np

from gen_utils import checks_from_words_packed, emit_test_data, preloads_from_words_packed
from mreg_layout import bf16_pair, words
from npu_model.configs import cvt, fp8_mma

rng = np.random.RandomState(4007)


def e4m3_tiles(n):
    """n m-registers of E4M3 codes with exponent fields 0..9, both signs (no NaN)."""
    shape = (n, 32, 32)
    sign, field, mantissa = rng.randint(0, 2, shape), rng.randint(0, 10, shape), rng.randint(0, 8, shape)
    return (sign << 7 | field << 3 | mantissa).astype(np.uint8)


a, w, c0 = e4m3_tiles(6), e4m3_tiles(4), e4m3_tiles(1)[0]
c1 = cvt.bf16_rne((rng.standard_normal((32, 32)) * 8).astype(np.float32).view(np.uint32))
acc0 = cvt.e4m3_to_f32(c0)
for k, s in ((0, 0), (2, 1), (4, 2)):
    acc0 = fp8_mma.matmul(a[k], w[s], acc0)
acc1 = c1.astype(np.uint32) << 16
for k, s in ((1, 1), (3, 2), (5, 3)):
    acc1 = fp8_mma.matmul(a[k], w[s], acc1)

preloads = words([*a, *w, c0, *bf16_pair(c1)])
checks = words([*bf16_pair(cvt.bf16_rne(acc0)), cvt.e4m3_rn_satfinite(acc0, 123),
                *bf16_pair(cvt.bf16_rne(acc1)), cvt.e4m3_rn_satfinite(acc1, 121)])
emit_test_data(preloads_from_words_packed(0, preloads), checks_from_words_packed(512, checks))

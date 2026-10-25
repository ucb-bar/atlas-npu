#!/usr/bin/env python3
"""gen_mxu1_fp32_spill.py — golden data for assembly/mxu1_fp32_spill.S.

acc = C + A0 W0^T + A1 W1^T + A2 W2^T + A3 W3^T: a chain of four npu_model fp8_mma steps from a
BF16 C. The spill is the chain after two steps. A and W are E4M3 codes of every class but NaN, with
subnormals and signed zeros, so the FP32 sums use all 24 bits.

DRAM beats: A 0-127, W 128-255, C 256-319, X 320-383; results 384-831: BF16_RNE(X + A2 W2^T)
(m20, m21), the spill (m24..m27), acc0 (m28..m31) and acc1 (m32..m35).
"""
import numpy as np

from gen_utils import checks_from_words_packed, emit_test_data, preloads_from_words_packed
from mreg_layout import bf16_pair, fp32_quad, words
from npu_model.configs import cvt, fp8_mma

rng = np.random.RandomState(4006)


def e4m3_tiles(n):
    """n m-registers of E4M3 codes with exponent fields 0..10, both signs (no NaN)."""
    shape = (n, 32, 32)
    sign, field, mantissa = rng.randint(0, 2, shape), rng.randint(0, 11, shape), rng.randint(0, 8, shape)
    return (sign << 7 | field << 3 | mantissa).astype(np.uint8)


def bf16_tile():
    return cvt.bf16_rne((rng.standard_normal((32, 32)) * 4).astype(np.float32).view(np.uint32))


a, w = e4m3_tiles(4), e4m3_tiles(4)
c, x = bf16_tile(), bf16_tile()
acc = c.astype(np.uint32) << 16
for k in (0, 1):
    acc = fp8_mma.matmul(a[k], w[k], acc)
spill = acc
other = cvt.bf16_rne(fp8_mma.matmul(a[2], w[2], x.astype(np.uint32) << 16))
for k in (2, 3):
    acc = fp8_mma.matmul(a[k], w[k], acc)

preloads = words([*a, *w, *bf16_pair(c), *bf16_pair(x)])
checks = words(bf16_pair(other) + fp32_quad(spill) + fp32_quad(acc) * 2)
emit_test_data(preloads_from_words_packed(0, preloads), checks_from_words_packed(384, checks))

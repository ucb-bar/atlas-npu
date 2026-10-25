"""Vectorized Atlas RTL arithmetic, bit-exact with npu_model.configs.rtl_math.

npu_model's RtlNumerics evaluates the inner-product-tree MXU and its FP8 pop one element at a
time in Python, about 30 ms per 32 x 32 x 32 matmul. A DiT sampler runs some 400,000 of
them, so FastRtlNumerics computes the same integer datapaths on whole tiles with torch int64
tensors. tests/test_numerics.py checks every function against rtl_math bit for bit.
"""
from functools import lru_cache

import torch

from npu_model.configs.numerics import RtlNumerics
from npu_model.configs.rtl_math import product


@lru_cache(maxsize=None)
def _product_tables():
    """E4M3Mul for every (a, b) byte pair: sign, biased E5 exponent (0 = zero), 7 fraction bits."""
    sign = torch.zeros(256, 256, dtype=torch.int64)
    exp = torch.zeros(256, 256, dtype=torch.int64)
    frac = torch.zeros(256, 256, dtype=torch.int64)
    for a in range(256):
        for b in range(256):
            s, e, m = product(a, b)
            sign[a, b], exp[a, b], frac[a, b] = s, e, m
    return sign, exp, frac


def _bits16(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF


def _from_bits16(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int32).to(torch.int16).view(torch.bfloat16)


def ipt_matmul(a: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """AnchorAccumulationTree: c + a @ w.T, one rounding per output.

    a: [R, K] FP8 activations; w: [N, K] FP8, row j = output column j (a weight slot);
    c: [R, N] BF16 partial sums. Returns [R, N] BF16. Same arithmetic as rtl_math.ipt_row.
    """
    ps, pe, pm = _product_tables()
    ai = a.contiguous().view(torch.uint8).to(torch.int64)
    wi = w.contiguous().view(torch.uint8).to(torch.int64)
    idx = ai[:, None, :] * 256 + wi[None, :, :]              # [R, N, K]
    s, e, m = ps.view(-1)[idx], pe.view(-1)[idx], pm.view(-1)[idx]
    valid = e != 0

    cb = _bits16(c)
    cpe, cpm = (cb >> 7) & 255, cb & 127
    c_nonzero = (cb & 0x7FFF) != 0
    neg_big = torch.full_like(cpe, -1024)
    anchor = torch.maximum(torch.where(valid, e - 13, -1024).amax(-1),
                           torch.where(c_nonzero, cpe - 127, neg_big)) + 7

    shift = (anchor[..., None] - (e - 13)).clamp(0, 62)
    magnitude = ((128 | m) << 24) >> shift
    total = torch.where(valid, torch.where(s == 1, -magnitude, magnitude), 0).sum(-1)

    cshift = anchor - (cpe - 127) - 24
    sig = torch.where(cpe != 0, 128, 0) | cpm
    right = sig >> cshift.clamp(0, 62)
    left = (sig << (-cshift).clamp(0, 62)) & 0xFFFFFFFF
    cmag = torch.where((cshift >= 32) | (cshift < -31), 0, torch.where(cshift >= 0, right, left))
    total = total + torch.where((cb & 0x8000) != 0, -cmag, cmag)
    total = ((total + (1 << 31)) & 0xFFFFFFFF) - (1 << 31)

    sign = (total < 0).to(torch.int64)
    mag = total.abs()
    _, exponent = torch.frexp(mag.to(torch.float64))
    lead = exponent.to(torch.int64) - 1                       # bit_length - 1
    sh = (lead - 7).clamp(min=0)
    sigr = mag >> sh
    rem = mag & ((1 << sh) - 1)
    half = torch.where(sh > 0, 1 << (sh - 1).clamp(min=0), 0)
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((sigr & 1) == 1)))
    sigr = sigr + up.to(torch.int64)
    over = sigr >= 256
    sigr = torch.where(over, sigr >> 1, sigr)
    lead = lead + over.to(torch.int64)
    bexp = lead + anchor - 31 + 127
    bits = torch.where(bexp <= 0, 0,
                       torch.where(bexp >= 255, (sign << 15) | 0x7F7F,
                                   (sign << 15) | (bexp.clamp(0, 255) << 7) | (sigr & 127)))
    bits = torch.where(total == 0, 0, bits)
    return _from_bits16(bits)


def pack_fp8(value: torch.Tensor, scale: int) -> torch.Tensor:
    """BF16ScaleToE4M3 (MXU pop), rtl_math.pack_row."""
    raw = _bits16(value.flatten())
    sign, exp, mant = raw >> 15, (raw >> 7) & 255, raw & 127
    shift = min(127, max(-128, int(scale) - 127))
    adjusted = exp - 127 + shift
    sig = 128 | mant
    rounded = (sig >> 4) + (((sig & 8) != 0) & (((sig & 7) != 0) | (((sig >> 4) & 1) != 0))).to(torch.int64)
    carry = rounded == 16
    adjusted = adjusted + carry.to(torch.int64)
    rounded = torch.where(carry, 8, rounded)
    frac = (rounded - 8) & 7
    out = torch.where(adjusted > 8, (sign << 7) | 0x7E,
                      torch.where(adjusted < -6, 0, (sign << 7) | ((adjusted + 7).clamp(0, 15) << 3) | frac))
    out = torch.where((exp == 255) & (mant == 0), (sign << 7) | 0x7E, out)
    out = torch.where((exp == 0) | ((exp == 255) & (mant != 0)), 0, out)
    return out.to(torch.uint8)


class FastRtlNumerics(RtlNumerics):
    """RtlNumerics with the per-element Python loops replaced by tensor code (same bits)."""

    @staticmethod
    def acc_to_fp8(x: torch.Tensor, scale: int) -> torch.Tensor:
        return pack_fp8(x.flatten(), scale).view(torch.float8_e4m3fn).reshape(x.shape)

    @staticmethod
    def inner_product_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # b is the right operand (K x N); the trees read it as N weight rows.
        return ipt_matmul(a, b.contiguous().view(torch.uint8).T.contiguous(), c)

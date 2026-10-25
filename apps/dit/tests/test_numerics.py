"""FastRtlNumerics' mxu1 matmul against npu_model's fp8_mma, bit for bit."""
import numpy as np
import torch

from ditflow.numerics import ipt_matmul
from npu_model.configs import fp8_mma

SPECIAL_FP8 = [0x00, 0x80, 0x01, 0x07, 0x08, 0x7E, 0x7F, 0xFE, 0xFF, 0x38, 0xB8, 0x40, 0x81, 0x87]
SPECIAL_F32 = [0x0000_0000, 0x8000_0000, 0x0000_0001, 0x807F_FFFF, 0x0080_0000, 0x7F80_0000, 0xFF80_0000,
               0x7FC0_0000, 0xFFC0_0001, 0x7F80_0001, 0x7F7F_FFFF, 0xFF7F_FFFF, 0x3F80_0000, 0xBF80_0000]


def _fp8(n, gen, special=0.1):
    x = torch.randint(0, 256, (n,), generator=gen, dtype=torch.int64)
    pick = torch.rand(n, generator=gen) < special
    x[pick] = torch.tensor(SPECIAL_FP8)[torch.randint(0, len(SPECIAL_FP8), (int(pick.sum()),), generator=gen)]
    return x.to(torch.uint8)


def _f32(n, gen, special=0.1, spread=1.0):
    x = torch.randn(n, generator=gen) * spread * torch.exp2(torch.randint(-20, 20, (n,), generator=gen).float())
    bits = x.view(torch.int32).long() & 0xFFFF_FFFF
    pick = torch.rand(n, generator=gen) < special
    bits[pick] = torch.tensor(SPECIAL_F32)[torch.randint(0, len(SPECIAL_F32), (int(pick.sum()),), generator=gen)]
    return torch.where(bits >= 1 << 31, bits - (1 << 32), bits).to(torch.int32).view(torch.float32)


def _check(a, w, c):
    fast = ipt_matmul(a.view(torch.float8_e4m3fn), w.view(torch.float8_e4m3fn), c).view(torch.int32).numpy()
    ref = fp8_mma.matmul(a.numpy(), w.numpy(), c.view(torch.int32).numpy().view(np.uint32))
    assert np.array_equal(fast.view(np.uint32), ref)


def test_ipt_matmul_random_tiles():
    gen = torch.Generator().manual_seed(1)
    for trial in range(6):
        a = _fp8(32 * 32, gen, special=0.05 * trial).view(32, 32)
        w = _fp8(32 * 32, gen, special=0.05 * trial).view(32, 32)
        c = _f32(32 * 32, gen, special=0.05 * trial, spread=2.0 ** (trial * 3)).view(32, 32)
        _check(a, w, c)


def test_ipt_matmul_cancellation_zero_rows_and_chains():
    gen = torch.Generator().manual_seed(2)
    a = _fp8(8 * 32, gen, special=0.0).view(8, 32)
    w = _fp8(8 * 32, gen, special=0.0).view(8, 32)
    a[:, 16:], w[:, 16:] = a[:, :16] ^ 0x80, w[:, :16]  # elements 16..31 cancel 0..15
    w[0] = 0
    a[1] = 0x80
    c = _f32(8 * 8, gen, special=0.3).view(8, 8)
    c[2] = -0.0
    _check(a, w, c)
    # A K = 128 chain through FP32 accumulators.
    acc = torch.zeros(32, 32)
    ref = np.zeros((32, 32), np.uint32)
    for _ in range(4):
        a, w = _fp8(32 * 32, gen).view(32, 32), _fp8(32 * 32, gen).view(32, 32)
        acc = ipt_matmul(a.view(torch.float8_e4m3fn), w.view(torch.float8_e4m3fn), acc)
        ref = fp8_mma.matmul(a.numpy(), w.numpy(), ref)
        assert np.array_equal(acc.view(torch.int32).numpy().view(np.uint32), ref)

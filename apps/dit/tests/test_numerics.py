"""FastRtlNumerics against npu_model's element-wise RTL models, bit for bit."""
import torch

from ditflow.numerics import ipt_matmul, pack_fp8
from npu_model.configs import rtl_math

SPECIAL_FP8 = [0x00, 0x80, 0x01, 0x07, 0x08, 0x7E, 0x7F, 0xFE, 0xFF, 0x38, 0xB8, 0x40]
SPECIAL_BF16 = [0x0000, 0x8000, 0x0001, 0x807F, 0x7F80, 0xFF80, 0x7FC0, 0x7F7F, 0xFF7F,
                0x3F80, 0xBF80, 0x43E0, 0x43F0, 0x4400, 0x3C00, 0x0080]


def _fp8(n, gen, special=0.1):
    x = torch.randint(0, 256, (n,), generator=gen, dtype=torch.int64)
    pick = torch.rand(n, generator=gen) < special
    x[pick] = torch.tensor(SPECIAL_FP8)[torch.randint(0, len(SPECIAL_FP8), (int(pick.sum()),), generator=gen)]
    return x.to(torch.uint8)


def _bf16(n, gen, special=0.1, spread=1.0):
    x = (torch.randn(n, generator=gen) * spread * torch.exp2(torch.randint(-20, 20, (n,), generator=gen).float()))
    raw = x.to(torch.bfloat16).view(torch.int16).to(torch.int64) & 0xFFFF
    pick = torch.rand(n, generator=gen) < special
    raw[pick] = torch.tensor(SPECIAL_BF16)[torch.randint(0, len(SPECIAL_BF16), (int(pick.sum()),), generator=gen)]
    return raw.to(torch.int32).to(torch.int16).view(torch.bfloat16)


def _bits(t):
    return t.contiguous().view(torch.int16).tolist() if t.dtype == torch.bfloat16 else t.view(torch.uint8).tolist()


def test_ipt_matmul_random_tiles():
    gen = torch.Generator().manual_seed(1)
    for trial in range(6):
        a = _fp8(32 * 32, gen, special=0.05 * trial).view(32, 32)
        w = _fp8(32 * 32, gen, special=0.05 * trial).view(32, 32)
        c = _bf16(32 * 32, gen, special=0.05 * trial, spread=2.0 ** (trial * 3)).view(32, 32)
        fast = ipt_matmul(a.view(torch.float8_e4m3fn), w.view(torch.float8_e4m3fn), c)
        ref = torch.stack([rtl_math.ipt_row(a[r].view(torch.float8_e4m3fn), w.view(torch.float8_e4m3fn), c[r])
                           for r in range(32)])
        assert _bits(fast) == _bits(ref), f"trial {trial}"


def test_ipt_matmul_cancellation_and_zero_rows():
    gen = torch.Generator().manual_seed(2)
    a = _fp8(8 * 32, gen, special=0.0).view(8, 32)
    w = torch.cat([a, a ^ 0x80])[:8].contiguous()  # rows that cancel the products of a
    w[0] = 0
    c = _bf16(8 * 8, gen, special=0.3).view(8, 8)
    c[1] = 0
    fast = ipt_matmul(a.view(torch.float8_e4m3fn), w.view(torch.float8_e4m3fn), c)
    ref = torch.stack([rtl_math.ipt_row(a[r].view(torch.float8_e4m3fn), w.view(torch.float8_e4m3fn), c[r])
                       for r in range(8)])
    assert _bits(fast) == _bits(ref)


def test_pop_fp8_exhaustive():
    every_bf16 = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    for scale in (0, 1, 100, 119, 127, 130, 135, 150, 254, 255):
        assert pack_fp8(every_bf16, scale).tolist() == rtl_math.pack_row(every_bf16, scale).tolist(), scale

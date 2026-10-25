"""Vectorized Atlas mxu1 arithmetic, bit-exact with npu_model.configs.fp8_mma.

A DiT sampler runs some 400,000 32 x 32 x 32 matmuls on mxu1. FastRtlNumerics computes each one
as a torch port of fp8_mma's integer datapath (one NVIDIA Blackwell FP8 MMA step per output, FP32
accumulators). tests/test_numerics.py checks it against fp8_mma bit for bit.
"""
import torch

from npu_model.configs.numerics import RtlNumerics

_CODES = torch.arange(256)
_FIELD = (_CODES >> 3) & 15
_SIG = torch.where(_FIELD > 0, 8 | (_CODES & 7), _CODES & 7)  # |E4M3| = _SIG / 8 x 2^_EXP
_EXP = _FIELD.clamp(min=1) - 7
_NAN = _CODES & 0x7F == 0x7F
_NEG = _CODES >> 7 == 1


def ipt_matmul(a: torch.Tensor, w: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """AnchorAccumulationTree: c + a @ w.T, fp8_mma.matmul in torch.

    a: [R, K] FP8 activations; w: [N, K] FP8, row j = output column j (a weight slot);
    c: [R, N] FP32 accumulators. Returns [R, N] FP32 with fp8_mma's bits (K = 32 on mxu1).
    """
    ai = a.contiguous().view(torch.uint8).long()[:, None, :]
    wi = w.contiguous().view(torch.uint8).long()[None, :, :]
    product = _SIG[ai] * _SIG[wi]  # x 2^(e - 6)
    e = _EXP[ai] + _EXP[wi]
    negative = _NEG[ai] ^ _NEG[wi]
    cb = c.contiguous().view(torch.int32).long() & 0xFFFF_FFFF
    c_field = (cb >> 23) & 255
    c_exp = c_field.clamp(min=1) - 127
    c_sig = (cb & 0x7F_FFFF) | torch.where(c_field > 0, 0x80_0000, 0)  # x 2^(c_exp - 23)
    emax = torch.maximum(c_exp, torch.where(product != 0, e, -1000).amax(-1))
    # Terms in units of 2^(emax - 25), cut toward zero: P sits at bits 26..19 and C's significand
    # at bits 25..2 when its exponent is emax.
    term = (product << 19) >> (emax[..., None] - e).clamp(0, 63)
    c_term = (c_sig << 2) >> (emax - c_exp).clamp(0, 63)
    total = torch.where(negative, -term, term).sum(-1) + torch.where(cb >> 31 == 1, -c_term, c_term)
    magnitude = total.abs()
    lead = torch.frexp(magnitude.double())[1].long() - 1
    mantissa = torch.where(lead > 23, magnitude >> (lead - 23).clamp(min=0), magnitude << (23 - lead).clamp(min=0))
    d = torch.where(total < 0, 1 << 31, 0) | (lead + emax - 25 + 127) << 23 | (mantissa & 0x7F_FFFF)
    d = torch.where(total == 0, 0, d)
    # Without nonzero products the sum is C exactly; this also covers a subnormal C.
    d = torch.where((product == 0).all(-1), torch.where(cb == 0x8000_0000, 0, cb), d)
    d = torch.where(c_field == 255, cb, d)
    nan = _NAN[ai].any(-1) | _NAN[wi].any(-1) | ((c_field == 255) & (cb & 0x7F_FFFF != 0))
    d = torch.where(nan, 0x7FFF_FFFF, d)
    return torch.where(d >= 1 << 31, d - (1 << 32), d).to(torch.int32).view(torch.float32)


class FastRtlNumerics(RtlNumerics):
    """RtlNumerics with mxu1's matmul as torch tensor code (same bits)."""

    @staticmethod
    def inner_product_matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # b is the right operand (K x N); the trees read it as N weight rows.
        return ipt_matmul(a, b.contiguous().view(torch.uint8).T.contiguous(), c)

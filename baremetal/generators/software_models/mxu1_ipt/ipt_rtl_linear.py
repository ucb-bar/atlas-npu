"""mxu1, the inner-product-tree MXU, as F.linear, bit for bit with the RTL.

Each 32-wide K tile is one vmatmul(.acc).mxu1 step into FP32 accumulators: one NVIDIA Blackwell
FP8 MMA step per output (npu_model configs/fp8_mma.py). The result leaves through vmatpop.bf16
or vmatpop.fp8 (configs/cvt.py). E4M3 values convert exactly, with subnormals and NaN.
"""
from typing import Optional

import numpy as np
import torch

from npu_model.configs import cvt, fp8_mma

from .fp_formats import OutputFmtSel

_E4M3_FLOAT_LUT = torch.from_numpy(cvt.e4m3_to_f32(np.arange(256, dtype=np.uint8)).view(np.float32))


def torch_float_to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    x_f32 = x.float().contiguous()
    u32 = x_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = (u32 >> 16) & 0xFFFF
    lsb = upper & 1
    rounded = u32 + (0x7FFF + lsb)
    return ((rounded >> 16) & 0xFFFF).to(torch.int32)


def torch_bf16_bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    u32 = (bits.to(torch.int32) & 0xFFFF) << 16
    return u32.view(torch.float32)


def quant_bf16_tensor(x: torch.Tensor) -> torch.Tensor:
    bits = torch_float_to_bf16_bits(x.float())
    return torch_bf16_bits_to_float(bits)


def float_to_e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    """E4M3 codes of x, rounded to nearest even and saturated to 448 (exact for E4M3 values)."""
    bits = x.detach().float().contiguous().cpu().numpy().view(np.uint32)
    return torch.from_numpy(cvt.e4m3_rn_satfinite(bits)).to(x.device)


def e4m3_words(x) -> list[int]:
    """32-bit words of the E4M3 codes of x (a tensor or array), row-major: byte j of word k is
    element 4k + j. Exact for E4M3 values, -0 and subnormals included."""
    codes = float_to_e4m3_bytes(torch.as_tensor(x)).contiguous().numpy()
    return codes.view("<u4").ravel().tolist()


def e4m3_bytes_to_float(x: torch.Tensor) -> torch.Tensor:
    lut = _E4M3_FLOAT_LUT.to(x.device)
    return lut[(x.to(torch.int64) & 0xFF)]


def decode_model_output_bits(out_bits: torch.Tensor, out_fmt_sel: OutputFmtSel) -> torch.Tensor:
    if out_fmt_sel is OutputFmtSel.OutBF16:
        return torch_bf16_bits_to_float(out_bits.to(torch.int32))
    e4m3_bytes = (out_bits & 0xFF).to(torch.uint8)
    return e4m3_bytes_to_float(e4m3_bytes)


class IPTLinearRTLFunction:
    """y = pop(b + x @ w^T), as a program computes it on mxu1.

    x [..., K] and w [N, K] are rounded to E4M3 (exact for E4M3 values); K is padded with zeros
    to whole 32-wide tiles. A bias b [N], also E4M3, is the initial accumulator row
    (vmatpush.acc.fp8); without it the first tile starts from +0 (vmatmul). The pop is BF16_RNE,
    or for OutE4M3 E4M3_RN_SATFINITE(acc * 2^scale_exp) (e-register 127 + scale_exp). The
    result holds the popped values as float32.
    """

    def __init__(self, out_fmt_sel: OutputFmtSel = OutputFmtSel.OutBF16):
        self.out_fmt_sel = out_fmt_sel

    def __call__(
        self,
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        b_q: Optional[torch.Tensor] = None,
        scale_exp: int = 0,
    ) -> torch.Tensor:
        k, n = x_q.shape[-1], w_q.shape[0]
        pad = (0, -k % 32)
        x = np.pad(float_to_e4m3_bytes(x_q.reshape(-1, k)).cpu().numpy(), ((0, 0), pad))
        w = np.pad(float_to_e4m3_bytes(w_q).cpu().numpy(), ((0, 0), pad))
        acc = np.zeros((x.shape[0], n), np.uint32)
        if b_q is not None:
            acc[:] = cvt.e4m3_to_f32(float_to_e4m3_bytes(b_q).cpu().numpy())
        for k0 in range(0, x.shape[1], 32):
            acc = fp8_mma.matmul(x[:, k0:k0 + 32], w[:, k0:k0 + 32], acc)
        if self.out_fmt_sel is OutputFmtSel.OutBF16:
            y = cvt.bf16_rne(acc).astype(np.uint32) << 16
        else:
            y = cvt.e4m3_to_f32(cvt.e4m3_rn_satfinite(acc, 127 + scale_exp))
        return torch.from_numpy(y.view(np.float32)).reshape(*x_q.shape[:-1], n).to(x_q.device)

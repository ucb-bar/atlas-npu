"""Number formats and matmul arithmetic of the format study (docs/nvfp4-accuracy.md).

A quantizer maps an FP32 tensor to the FP32 values of its quantized form, with the blocks along
the last axis (K). A matmul of those values in FP32 then applies every scale as the hardware
does, up to the rounding of the sum. The caller turns TF32 off.
"""
import math

import torch

from ditflow import quant

E2M1_MAX, E4M3_MAX = 6.0, 448.0
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
E2M1_UP = (0.75, 1.75, 3.5)  # ties that round up to the even code


def exact(x: torch.Tensor, amax: float = 0.0) -> torch.Tensor:
    return x


def bf16(x: torch.Tensor, amax: float = 0.0) -> torch.Tensor:
    return x.bfloat16().float()


def e4m3(x: torch.Tensor) -> torch.Tensor:
    """E4M3, round to nearest even, saturate at 448, subnormals kept."""
    return x.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def e2m1(x: torch.Tensor) -> torch.Tensor:
    """E2M1, round to nearest even, saturate at 6 (ModelOpt's fp4_round_magnitude)."""
    a = x.abs().contiguous()
    i = torch.bucketize(a, torch.tensor(E2M1_BOUNDS, device=x.device))
    i = i + torch.isin(a, torch.tensor(E2M1_UP, device=x.device))
    return torch.tensor(E2M1, device=x.device)[i].copysign(x)


def global_scale(amax: float, pow2: bool) -> float:
    """NVFP4's per-tensor scale amax / (6 x 448); pow2 rounds it up to a power of two."""
    g = amax / (E2M1_MAX * E4M3_MAX)
    return 2.0 ** math.ceil(math.log2(g)) if pow2 else g


def nvfp4_scales(x: torch.Tensor, g: float) -> torch.Tensor:
    """The E4M3 scale of each 16-element block: amax / (6 g), saturated at 448; 0 zeroes the block."""
    return e4m3(x.unflatten(-1, (-1, 16)).abs().amax(-1) / (E2M1_MAX * g))


def nvfp4(x: torch.Tensor, amax: float, pow2: bool = False) -> torch.Tensor:
    """NVFP4 (ModelOpt nvfp4.yaml): E2M1 elements, an E4M3 scale per 16 along K, scale g per tensor.

    The block scales follow x (dynamic); amax is the tensor's calibrated or own maximum.
    """
    g = global_scale(amax, pow2)
    sg = nvfp4_scales(x, g)[..., None] * g
    b = x.unflatten(-1, (-1, 16))
    return torch.where(sg == 0, 0.0, e2m1(b / torch.where(sg == 0, 1.0, sg)) * sg).flatten(-2)


def mx(x: torch.Tensor, elem, elem_max: float) -> torch.Tensor:
    """OCP MX: an E8M0 scale per 32 along K, 2^ceil(log2(amax / elem_max)) as ModelOpt and TE set it."""
    b = x.unflatten(-1, (-1, 32))
    s = torch.exp2(torch.log2(b.abs().amax(-1, keepdim=True) / elem_max).ceil().clamp(-127, 127))
    return (elem(b / s) * s).flatten(-2)


def mxfp4(x: torch.Tensor, amax: float = 0.0) -> torch.Tensor:
    return mx(x, e2m1, E2M1_MAX)


def mxfp8(x: torch.Tensor, amax: float = 0.0) -> torch.Tensor:
    return mx(x, e4m3, E4M3_MAX)


def fp8(x: torch.Tensor, amax: float, margin: float = 1.0, flush: bool = False) -> torch.Tensor:
    """E4M3 with one power-of-two scale per tensor (ditflow.quant); flush: the v1 mxu1 flushes subnormals."""
    a = quant.exponent_for(amax, margin)
    y = e4m3(x * 2.0 ** -a)
    return (torch.where(y.abs() < 2.0 ** -6, 0.0, y) if flush else y) * 2.0 ** a


def fp8_split(x: torch.Tensor, amax: float, margin: float = 1.0, shift: int = 0) -> torch.Tensor:
    """BF16(x) as two E4M3 terms of one power-of-two scale 2^a (ditflow.quant): hi = E4M3(x / 2^a),
    lo = E4M3((x / 2^a - hi) 2^shift) / 2^shift. With shift = 0, both terms take the same weights
    into one accumulator on Atlas (two FP8 matmuls); shift > 0 gives lo a scale of its own."""
    a = quant.exponent_for(amax, margin)
    s = bf16(x) * 2.0 ** -a
    hi = e4m3(s)
    return (hi + e4m3((s - hi) * 2.0 ** shift) * 2.0 ** -shift) * 2.0 ** a


def mm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """a @ w^T with an FP32 accumulator (w holds one output column per row)."""
    return a @ w.mT


def mm_ipt(a: torch.Tensor, w: torch.Tensor, tile: int = 32) -> torch.Tensor:
    """Atlas mxu1 before ISA 2.0: each 32-wide K tile is added to a BF16 accumulator, rounding
    once per tile.

    The partial sum of a tile is exact in FP32 for FP8 operands of nearby exponents; scale
    exponents commute with the rounding. ISA 2.0 mxu1 accumulates in FP32 (Blackwell FP8, mm).
    """
    acc = torch.zeros(*a.shape[:-1], w.shape[-2], device=a.device)
    for k in range(0, a.shape[-1], tile):
        acc = (acc + a[..., k:k + tile] @ w[..., k:k + tile].mT).bfloat16().float()
    return acc

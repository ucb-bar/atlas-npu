"""FP8 (E4M3) quantization with per-tensor power-of-two scales, as Atlas evaluates it.

A tensor with scale exponent a is stored as FP8(x / 2**a). Atlas applies 2**a exactly: in a
scale register for vmatpop.fp8 (raw value 127 - a), or as a multiply by a power-of-two BF16 tile
after a matmul. Its MXU treats FP8 subnormals as zero, so the quantizer flushes them too.

Weights get their exponent from their own maximum; activations get it from calibration: the
largest magnitude each matmul input reaches over the sampler's trajectory on a few random
inputs, with a margin for inputs calibration did not see.
"""
import math

import torch

from . import model as M

FP8_MAX = 448.0
FP8_MIN_NORMAL = 2.0 ** -6


def exponent_for(amax: float, margin: float = 1.0) -> int:
    """The smallest a with amax * margin / 2**a <= FP8_MAX."""
    if amax <= 0:
        return 0
    return math.ceil(math.log2(amax * margin / FP8_MAX))


def to_fp8(x: torch.Tensor, a: int) -> torch.Tensor:
    """FP8 codes (uint8) of x / 2**a: round to nearest even, saturate at 448, flush subnormals."""
    y = (x.float() * 2.0 ** -a).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    codes = y.view(torch.uint8)
    return torch.where((codes & 0x78) == 0, torch.zeros_like(codes), codes)


def from_fp8(codes: torch.Tensor, a: int) -> torch.Tensor:
    return codes.view(torch.float8_e4m3fn).float() * 2.0 ** a


def fake(x: torch.Tensor, a: int) -> torch.Tensor:
    return from_fp8(to_fp8(x, a), a)


def calibrate(p: dict, d: M.DiTDims, seeds=(101, 102, 103, 104)) -> dict[str, float]:
    """Largest |value| of each FP8 activation over full sampler trajectories (FP32)."""
    amax: dict[str, float] = {}

    def observe(name, v):
        amax[name] = max(amax.get(name, 0.0), float(v.abs().max()))
        return v

    with torch.no_grad():
        for seed in seeds:
            x0, cond = random_inputs(d, seed)
            M.sample(p, x0, cond, d, q=observe)
    return amax


def random_inputs(d: M.DiTDims, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian noise for the chunk and a Gaussian stand-in for the observation encoding."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, d.horizon, d.action_dim, generator=g), torch.randn(1, d.cond_dim, generator=g)


class QuantPlan:
    """Scale exponents for every FP8 tensor of a model, and the FP8-rounded weights."""

    def __init__(self, p: dict, d: M.DiTDims, act_amax: dict[str, float], margin: float = 2.0):
        self.d = d
        self.act = {name: exponent_for(act_amax[name], margin) for name in M.act_names(d)}
        self.weight = {name: exponent_for(float(p[name].abs().max())) for name in M.weight_names(d)}
        self.codes = {name: to_fp8(p[name], self.weight[name]) for name in M.weight_names(d)}
        self.p = dict(p)
        for name in M.weight_names(d):                 # the weights Atlas multiplies with
            self.p[name] = from_fp8(self.codes[name], self.weight[name])

    def q(self, name: str, v: torch.Tensor) -> torch.Tensor:
        return fake(v, self.act[name])

    def to_dict(self) -> dict:
        return {"act": self.act, "weight": self.weight}

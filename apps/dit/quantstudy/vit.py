"""A pretrained DINOv3 ViT (transformers) with its matmuls in a format under test.

wrap() replaces the linears (q, k, v, o, up, down) and the patch embedding (a linear on 16 x 16
patches, as Atlas runs it) with QLinear, and routes attention through attention(), whose
QK^T and PV are FP32 or quantized. LayerNorm, RoPE, softmax, GELU, LayerScale and the residuals
stay FP32. A Run holds the current setting of a wrapped model.
"""
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AttentionInterface, DINOv3ViTModel

from . import formats

KINDS = ("pe", "qkv", "o", "up", "down")
MODELS = {"vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
          "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m"}


@dataclass(frozen=True)
class Fmt:
    weight: Callable    # (w, amax) -> quantized values
    act: Callable       # (x, calibrated amax) -> quantized values
    mm: Callable        # (a, w) -> a @ w^T


@dataclass
class Run:
    fmt: dict = field(default_factory=dict)     # kind: the format of those linears; absent is FP32
    attn: Fmt | None = None                     # QK^T and PV; None is FP32
    amax: dict = field(default_factory=dict)    # matmul input: its calibrated maximum
    observe: Callable | None = None             # (name, tensor): every matmul input, before quantization
    probe: Callable | None = None               # (QLinear, input)


class QLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, name: str, kind: str, act: str, wmax: float,
                 run: Run):
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias.detach(), requires_grad=False)
        self.name, self.kind, self.act, self.wmax, self.run = name, kind, act, wmax, run
        self.cache = {}

    def wq(self, fmt: Fmt) -> torch.Tensor:
        if fmt not in self.cache:
            self.cache[fmt] = fmt.weight(self.weight, self.wmax)
        return self.cache[fmt]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.run
        if r.observe:
            r.observe(self.act, x)
        if r.probe:
            r.probe(self, x)
        fmt = r.fmt.get(self.kind)
        if fmt is None:
            return F.linear(x, self.weight, self.bias)
        y = fmt.mm(fmt.act(x, r.amax[self.act]), self.wq(fmt))
        return y if self.bias is None else y + self.bias


class PatchEmbed(QLinear):
    """The patch convolution as a linear on flattened patches, returning the convolution's layout."""

    def __init__(self, conv: nn.Conv2d, run: Run):
        w = conv.weight.flatten(1)
        super().__init__(w, conv.bias, "pe", "pe", "pe", float(w.abs().max()), run)
        self.patch = conv.kernel_size[0]

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        p = self.patch
        y = super().forward(F.unfold(pixels, p, stride=p).mT)
        return y.mT.unflatten(-1, (pixels.shape[-2] // p, pixels.shape[-1] // p))


def attention(module, q, k, v, mask, scaling=None, dropout=0.0, **kwargs):
    r, n = module.run, module.qname
    a = r.attn
    mm = a.mm if a else formats.mm

    def operand(name, t):
        if r.observe:
            r.observe(f"{n}.{name}", t)
        return a.act(t, r.amax[f"{n}.{name}"]) if a else t

    s = mm(operand("q", q * scaling), operand("k", k))
    if mask is not None:
        s = s + mask
    out = mm(operand("p", torch.softmax(s, -1)), operand("v", v).mT)
    return out.transpose(1, 2).contiguous(), None


AttentionInterface.register("quantstudy", attention)


def load(model: str, cache_dir: str | None, device: str = "cuda") -> DINOv3ViTModel:
    return DINOv3ViTModel.from_pretrained(MODELS[model], cache_dir=cache_dir).to(device).eval().requires_grad_(False)


def wrap(vit: DINOv3ViTModel) -> Run:
    run = Run()
    for l, layer in enumerate(vit.model.layer):
        att, mlp = layer.attention, layer.mlp
        att.run, att.qname = run, str(l)
        qkv = [getattr(att, f"{s}_proj") for s in "qkv"]
        wmax = max(float(m.weight.abs().max()) for m in qkv)      # q, k, v are one GEMM on Atlas
        for s, m in zip("qkv", qkv):
            setattr(att, f"{s}_proj", QLinear(m.weight, m.bias, f"{l}.{s}", "qkv", f"{l}.qkv", wmax, run))
        for parent, attr, kind in ((att, "o_proj", "o"), (mlp, "up_proj", "up"), (mlp, "down_proj", "down")):
            m = getattr(parent, attr)
            setattr(parent, attr, QLinear(m.weight, m.bias, f"{l}.{kind}", kind, f"{l}.{kind}",
                                          float(m.weight.abs().max()), run))
    vit.embeddings.patch_embeddings = PatchEmbed(vit.embeddings.patch_embeddings, run)
    vit.set_attn_implementation("quantstudy")
    return run


def linears(vit: DINOv3ViTModel) -> list[QLinear]:
    return [m for m in vit.modules() if isinstance(m, QLinear)]


@torch.no_grad()
def calibrate(vit: DINOv3ViTModel, run: Run, batches) -> dict[str, float]:
    """Largest |value| of every matmul input over the calibration images (FP32 model)."""
    amax: dict[str, float] = {}

    def observe(name, t):
        amax[name] = max(amax.get(name, 0.0), float(t.abs().max()))

    saved = run.fmt, run.attn, run.observe
    run.fmt, run.attn, run.observe = {}, None, observe
    for x in batches:
        vit(pixel_values=x)
    run.fmt, run.attn, run.observe = saved
    return amax

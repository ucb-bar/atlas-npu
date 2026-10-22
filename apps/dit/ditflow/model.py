"""The DiT noise predictor of specialist_lerobot's ``specialist_dit`` policy, and its sampler.

specialist_dit (github.com/specialist-ai/specialist_lerobot) predicts with LeRobot's
``multi_task_dit`` DiffusionTransformer and integrates its flow-matching velocity with Euler
steps. This module builds that exact LeRobot module with random weights (``build``), runs
LeRobot's sampler arithmetic in FP32 (``sample_reference``), and re-states the forward pass
operation by operation (``forward``) so that every matmul input can be observed (calibration)
or rounded to FP8 the way Atlas rounds it (fake quantization).
"""
import math
from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DiTDims:
    horizon: int = 50          # action steps per chunk (tokens)
    action_dim: int = 10       # position 3 + rot6d 6 + gripper opening 1
    hidden: int = 256
    heads: int = 4
    layers: int = 2
    tdim: int = 128            # timestep_embed_dim
    cond_dim: int = 267        # observation encoder output: state 11 + 2 views x 4 queries x 32
    steps: int = 100           # num_integration_steps (Euler)
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def to_dict(self) -> dict:
        return asdict(self)


# specialist_dit's defaults: hidden 512, 8 heads, 4 layers, timestep embedding 256, and a
# conditioning vector of state 11 + 2 views x 8 queries x 64 = 1035. About 29 MB of FP8 weights.
FULL = DiTDims(hidden=512, heads=8, layers=4, tdim=256, cond_dim=1035)
# The same architecture scaled to fit the standalone VCU118's 4 MiB of DRAM (~3 MB of weights).
SMALL = DiTDims()
PRESETS = {"small": SMALL, "full": FULL}


def lerobot_config(d: DiTDims) -> SimpleNamespace:
    """The MultiTaskDiTConfig fields DiffusionTransformer and FlowMatchingObjective read."""
    return SimpleNamespace(
        action_feature=SimpleNamespace(shape=(d.action_dim,)), horizon=d.horizon, hidden_dim=d.hidden,
        num_layers=d.layers, num_heads=d.heads, dropout=0.0, use_rope=True, rope_base=d.rope_base,
        timestep_embed_dim=d.tdim, use_positional_encoding=False, num_integration_steps=d.steps,
        integration_method="euler", sigma_min=0.0)


def build(d: DiTDims, seed: int = 0) -> nn.Module:
    """LeRobot's DiffusionTransformer with every Linear randomly initialized.

    LeRobot zero-initializes the adaLN projections (adaLN-Zero), which would make every block an
    identity; they get the same default Linear initialization as the other layers here.
    """
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import DiffusionTransformer
    torch.manual_seed(seed)
    model = DiffusionTransformer(lerobot_config(d), conditioning_dim=d.cond_dim)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                bound = 1.0 / math.sqrt(module.in_features)
                module.weight.copy_((torch.rand(module.weight.shape, generator=gen) * 2 - 1) * bound)
                module.bias.copy_((torch.rand(module.bias.shape, generator=gen) * 2 - 1) * bound)
    return model.eval()


def time_grid(d: DiTDims) -> torch.Tensor:
    return torch.linspace(0, 1, d.steps + 1)


def sample_reference(model: nn.Module, x0: torch.Tensor, cond: torch.Tensor, d: DiTDims):
    """FlowMatchingObjective._euler_integrate in FP32. Returns (x_final, [velocity per step])."""
    grid = time_grid(d)
    x, velocities = x0.clone(), []
    with torch.no_grad():
        for i in range(d.steps):
            t = torch.full((x.shape[0],), grid[i].item(), dtype=x.dtype)
            v = model(x, t, conditioning_vec=cond)
            velocities.append(v)
            x = x + (grid[i + 1] - grid[i]).item() * v
    return x, velocities


# --------------------------------------------------------------------------------------------
# The forward pass, operation by operation
# --------------------------------------------------------------------------------------------

Quant = Callable[[str, torch.Tensor], torch.Tensor]


def params(model: nn.Module, d: DiTDims) -> dict:
    """The parameters forward() reads, by name. adaLN is split into its timestep and
    conditioning columns: SiLU acts elementwise on cat(time, cond), so the product separates."""
    p = {
        "time1.w": model.time_mlp[1].weight, "time1.b": model.time_mlp[1].bias,
        "time2.w": model.time_mlp[3].weight, "time2.b": model.time_mlp[3].bias,
        "in.w": model.input_proj.weight, "in.b": model.input_proj.bias,
        "out.w": model.output_proj.weight, "out.b": model.output_proj.bias,
    }
    for l, blk in enumerate(model.transformer_blocks):
        ada = blk.adaLN_modulation[1]
        p.update({
            f"{l}.ada_t.w": ada.weight[:, :d.tdim], f"{l}.ada_c.w": ada.weight[:, d.tdim:], f"{l}.ada.b": ada.bias,
            f"{l}.qkv.w": blk.attn.qkv_proj.weight, f"{l}.qkv.b": blk.attn.qkv_proj.bias,
            f"{l}.o.w": blk.attn.out_proj.weight, f"{l}.o.b": blk.attn.out_proj.bias,
            f"{l}.up.w": blk.mlp[0].weight, f"{l}.up.b": blk.mlp[0].bias,
            f"{l}.down.w": blk.mlp[2].weight, f"{l}.down.b": blk.mlp[2].bias,
        })
    return {k: v.detach().float() for k, v in p.items()}


def weight_names(d: DiTDims) -> list[str]:
    names = ["time1.w", "time2.w", "in.w", "out.w"]
    for l in range(d.layers):
        names += [f"{l}.{n}.w" for n in ("ada_t", "ada_c", "qkv", "o", "up", "down")]
    return names


def act_names(d: DiTDims) -> list[str]:
    """Every FP8 activation, in the order forward() quantizes it."""
    names = ["temb", "t1", "ada_t", "ada_c", "x"]
    for l in range(d.layers):
        names += [f"{l}.{n}" for n in ("ln1", "q", "k", "v", "p", "attn", "ln2", "gelu")]
    return names + ["h"]


def sinusoidal_freqs(d: DiTDims) -> torch.Tensor:
    half = d.tdim // 2
    return torch.exp(torch.arange(half) * -(math.log(10000) / (half - 1)))


def rope_tables(d: DiTDims) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of RotaryPositionalEmbedding, [horizon, head_dim / 2] (both halves are equal)."""
    inv_freq = 1.0 / (d.rope_base ** (torch.arange(0, d.head_dim, 2).float() / d.head_dim))
    freqs = torch.outer(torch.arange(d.horizon, dtype=inv_freq.dtype), inv_freq)
    return freqs.cos(), freqs.sin()


def layer_norm(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), eps=1e-6)


def forward(p: dict, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, d: DiTDims,
            q: Quant = lambda name, v: v, gelu_tanh_time: bool = False) -> torch.Tensor:
    """DiffusionTransformer.forward for a batch of one. q(name, value) sees every matmul input
    (an identity for the FP32 reference, FP8 rounding for fake quantization).
    gelu_tanh_time swaps the time MLP's exact GELU for the tanh form Atlas evaluates."""
    lin = lambda v, w, b: v @ w.T + b
    tgelu = (lambda v: F.gelu(v, approximate="tanh")) if gelu_tanh_time else F.gelu
    emb = t[:, None] * sinusoidal_freqs(d)[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    t1 = tgelu(lin(q("temb", emb), p["time1.w"], p["time1.b"]))
    tf = tgelu(lin(q("t1", t1), p["time2.w"], p["time2.b"]))
    s_t, s_c = q("ada_t", F.silu(tf)), q("ada_c", F.silu(cond))

    h = lin(q("x", x[0]), p["in.w"], p["in.b"])
    cos, sin = rope_tables(d)
    T, H, D = d.horizon, d.hidden, d.head_dim
    for l in range(d.layers):
        mod = s_t @ p[f"{l}.ada_t.w"].T + s_c @ p[f"{l}.ada_c.w"].T + p[f"{l}.ada.b"]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod[0].chunk(6)
        a_in = q(f"{l}.ln1", layer_norm(h) * (1 + scale_msa) + shift_msa)
        qkv = lin(a_in, p[f"{l}.qkv.w"], p[f"{l}.qkv.b"]).reshape(T, 3, d.heads, D).permute(1, 2, 0, 3)
        qh, kh, vh = qkv[0], qkv[1], qkv[2]                      # [heads, T, D]

        def rope(v):
            v1, v2 = v[..., :D // 2], v[..., D // 2:]
            return torch.cat([v1 * cos - v2 * sin, v2 * cos + v1 * sin], dim=-1)

        scores = q(f"{l}.q", rope(qh) / math.sqrt(D)) @ q(f"{l}.k", rope(kh)).transpose(-1, -2)
        probs = torch.softmax(scores, dim=-1)
        o = q(f"{l}.p", probs) @ q(f"{l}.v", vh)                 # [heads, T, D]
        o = o.transpose(0, 1).reshape(T, H)
        h = h + gate_msa * lin(q(f"{l}.attn", o), p[f"{l}.o.w"], p[f"{l}.o.b"])
        m_in = q(f"{l}.ln2", layer_norm(h) * (1 + scale_mlp) + shift_mlp)
        up = F.gelu(lin(m_in, p[f"{l}.up.w"], p[f"{l}.up.b"]), approximate="tanh")
        h = h + gate_mlp * lin(q(f"{l}.gelu", up), p[f"{l}.down.w"], p[f"{l}.down.b"])
    return lin(q("h", h), p["out.w"], p["out.b"])[None]


def sample(p: dict, x0: torch.Tensor, cond: torch.Tensor, d: DiTDims, **kw):
    """The Euler sampler over forward(). Returns (x_final, [velocity per step])."""
    grid = time_grid(d)
    x, velocities = x0.clone(), []
    for i in range(d.steps):
        v = forward(p, x, grid[i:i + 1], cond, d, **kw)
        velocities.append(v)
        x = x + (grid[i + 1] - grid[i]).item() * v
    return x, velocities

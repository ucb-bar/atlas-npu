"""specialist_dit's observation encoder: DINOv3 ViT patch tokens pooled by learned queries.

The reference is specialist_lerobot's own PooledVision (DINOv3ViTModel from transformers,
specialist's QueryPool, then a projection) built at a scaled size with random weights, and its
ObservationEncoder's arithmetic: conditioning = cat(state, pooled(wide view), pooled(zoom view)).
At inference the crops are centered, so the input here is the two crops themselves, already
normalized by the policy's processor.

forward() restates PooledVision operation by operation, with every matmul input passed through
q(name, value) as in model.forward(), for calibration and fake quantization.
"""
import math
from dataclasses import dataclass, asdict
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VisionDims:
    views: tuple = (("wide", 336), ("zoom", 224))   # specialist_dit's vision_crops, in order
    patch: int = 16
    hidden: int = 192
    heads: int = 3
    layers: int = 4
    mlp: int = 768
    registers: int = 4
    queries: int = 4            # vision_queries
    token_dim: int = 32         # vision_token_dim
    state_dim: int = 11         # (n_obs_steps - 1) x 10 + 1
    rope_theta: float = 100.0
    eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def prefix(self) -> int:
        return 1 + self.registers

    def patches(self, size: int) -> int:
        return (size // self.patch) ** 2

    @property
    def cond_dim(self) -> int:
        return self.state_dim + len(self.views) * self.queries * self.token_dim

    def to_dict(self) -> dict:
        return asdict(self)


SMALL = VisionDims()


def dinov3_config(v: VisionDims):
    from transformers import DINOv3ViTConfig
    return DINOv3ViTConfig(
        hidden_size=v.hidden, num_attention_heads=v.heads, num_hidden_layers=v.layers,
        intermediate_size=v.mlp, patch_size=v.patch, image_size=v.views[0][1],
        num_register_tokens=v.registers, rope_theta=v.rope_theta, layer_norm_eps=v.eps)


def build(v: VisionDims, seed: int = 0) -> nn.Module:
    """specialist's PooledVision, its backbone a randomly initialized scaled DINOv3 ViT."""
    from transformers import DINOv3ViTModel
    from specialist_lerobot.policies.modeling_specialist_dit import PooledVision
    torch.manual_seed(seed)
    cfg = dinov3_config(v)
    with mock.patch.object(DINOv3ViTModel, "from_pretrained", lambda name, **kw: DINOv3ViTModel(cfg)):
        enc = PooledVision("facebook/dinov3-scaled-random", v.queries, v.token_dim, False)
    g = torch.Generator().manual_seed(seed)
    uni = lambda shape, bound: (torch.rand(shape, generator=g) * 2 - 1) * bound
    with torch.no_grad():
        for m in enc.modules():
            if isinstance(m, nn.Linear):
                b = 1.0 / math.sqrt(m.in_features)
                m.weight.copy_(uni(m.weight.shape, b))
                if m.bias is not None:
                    m.bias.copy_(uni(m.bias.shape, b))
            elif isinstance(m, nn.LayerNorm):
                m.weight.copy_(1 + uni(m.weight.shape, 0.2))
                m.bias.copy_(uni(m.bias.shape, 0.1))
        att = enc.pool.attention
        b = 1.0 / math.sqrt(v.hidden)
        att.in_proj_weight.copy_(uni(att.in_proj_weight.shape, b))
        att.in_proj_bias.copy_(uni(att.in_proj_bias.shape, b))
        pe = enc.backbone.embeddings.patch_embeddings
        b = 1.0 / math.sqrt(pe.weight[0].numel())
        pe.weight.copy_(uni(pe.weight.shape, b))
        pe.bias.copy_(uni(pe.bias.shape, b))
        enc.backbone.embeddings.cls_token.copy_(torch.randn(enc.backbone.embeddings.cls_token.shape, generator=g))
        enc.backbone.embeddings.register_tokens.copy_(
            torch.randn(enc.backbone.embeddings.register_tokens.shape, generator=g))
        for layer in enc.backbone.model.layer:
            layer.layer_scale1.lambda1.copy_(0.5 + torch.rand(v.hidden, generator=g))
            layer.layer_scale2.lambda1.copy_(0.5 + torch.rand(v.hidden, generator=g))
        enc.pool.query.copy_(torch.randn(enc.pool.query.shape, generator=g))
    return enc.eval()


def random_views(v: VisionDims, seed: int) -> list[torch.Tensor]:
    """Stand-ins for the processor's normalized crops: (1, 3, size, size) each."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 3, s, s, generator=g) for _, s in v.views]


def encode_reference(enc: nn.Module, state: torch.Tensor, views: list[torch.Tensor]) -> torch.Tensor:
    """ObservationEncoder.encode for one hand at inference: cat(state, pooled view, ...)."""
    with torch.no_grad():
        return torch.cat([state] + [enc(img) for img in views], dim=-1)


# --------------------------------------------------------------------------------------------
# Operation by operation
# --------------------------------------------------------------------------------------------

def patchify(img: torch.Tensor, patch: int) -> torch.Tensor:
    """(1, 3, H, W) -> (H/p * W/p, 3 * p * p), rows in raster order, columns as the conv weight's."""
    _, c, h, w = img.shape
    t = img[0].reshape(c, h // patch, patch, w // patch, patch).permute(1, 3, 0, 2, 4)
    return t.reshape((h // patch) * (w // patch), c * patch * patch)


def rope_tables(v: VisionDims, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """DINOv3ViTRopePositionEmbedding at eval: cos and sin [patches, head_dim / 2] (halves equal)."""
    n = size // v.patch
    inv_freq = 1 / v.rope_theta ** torch.arange(0, 1, 4 / v.head_dim, dtype=torch.float32)
    coords = (torch.arange(0.5, n, dtype=torch.float32) / n)
    coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    angles = (2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]).flatten(1, 2)
    return torch.cos(angles), torch.sin(angles)


def params(enc: nn.Module, v: VisionDims) -> dict:
    bb, pool = enc.backbone, enc.pool
    h = v.hidden
    p = {
        "pe.w": bb.embeddings.patch_embeddings.weight.reshape(h, -1), "pe.b": bb.embeddings.patch_embeddings.bias,
        "cls": bb.embeddings.cls_token[0, 0], "reg": bb.embeddings.register_tokens[0],
        "norm.w": bb.norm.weight, "norm.b": bb.norm.bias,
        "pool.query": pool.query[0],
        "pool.q.w": pool.attention.in_proj_weight[:h], "pool.q.b": pool.attention.in_proj_bias[:h],
        "pool.kv.w": pool.attention.in_proj_weight[h:], "pool.kv.b": pool.attention.in_proj_bias[h:],
        "pool.out.w": pool.attention.out_proj.weight, "pool.out.b": pool.attention.out_proj.bias,
        "pool.ln.w": pool.norm.weight, "pool.ln.b": pool.norm.bias,
        "pool.up.w": pool.mlp[0].weight, "pool.up.b": pool.mlp[0].bias,
        "pool.down.w": pool.mlp[2].weight, "pool.down.b": pool.mlp[2].bias,
        "proj.w": enc.project.weight, "proj.b": enc.project.bias,
    }
    for l, layer in enumerate(bb.model.layer):
        a = layer.attention
        p.update({
            f"{l}.ln1.w": layer.norm1.weight, f"{l}.ln1.b": layer.norm1.bias,
            f"{l}.ln2.w": layer.norm2.weight, f"{l}.ln2.b": layer.norm2.bias,
            f"{l}.qkv.w": torch.cat([a.q_proj.weight, a.k_proj.weight, a.v_proj.weight]),
            f"{l}.qkv.b": torch.cat([a.q_proj.bias, torch.zeros(h), a.v_proj.bias]),
            f"{l}.o.w": a.o_proj.weight, f"{l}.o.b": a.o_proj.bias,
            f"{l}.ls1": layer.layer_scale1.lambda1, f"{l}.ls2": layer.layer_scale2.lambda1,
            f"{l}.up.w": layer.mlp.up_proj.weight, f"{l}.up.b": layer.mlp.up_proj.bias,
            f"{l}.down.w": layer.mlp.down_proj.weight, f"{l}.down.b": layer.mlp.down_proj.bias,
        })
    return {k: t.detach().float() for k, t in p.items()}


def weight_names(v: VisionDims) -> list[str]:
    names = ["pe.w", "pool.q.w", "pool.kv.w", "pool.out.w", "pool.up.w", "pool.down.w", "proj.w"]
    for l in range(v.layers):
        names += [f"{l}.{n}.w" for n in ("qkv", "o", "up", "down")]
    return names


def act_names(v: VisionDims) -> list[str]:
    names = ["pe_in"]
    for l in range(v.layers):
        names += [f"{l}.{n}" for n in ("ln1", "q", "k", "v", "p", "attn", "ln2", "gelu")]
    return names + ["pool.qin", "pool.kvin", "pool.q", "pool.k", "pool.v", "pool.p", "pool.attn",
                    "pool.ln", "pool.gelu", "proj_in"]


def ln(x, w, b, eps):
    return F.layer_norm(x, (x.shape[-1],), w, b, eps)


def attention(qh, kh, vh, D, q, tag):
    """softmax(q k^T / sqrt(D)) v per head, with the FP8 points of the Atlas kernel."""
    s = q(f"{tag}.q", qh / math.sqrt(D)) @ q(f"{tag}.k", kh).transpose(-1, -2)
    return q(f"{tag}.p", torch.softmax(s, -1)) @ q(f"{tag}.v", vh)


def forward(p: dict, img: torch.Tensor, v: VisionDims, q=lambda name, t: t, gelu_tanh: bool = False) -> torch.Tensor:
    """PooledVision.forward for one view: (1, 3, S, S) -> (1, queries * token_dim)."""
    act = (lambda t: F.gelu(t, approximate="tanh")) if gelu_tanh else F.gelu
    lin = lambda t, w, b: t @ w.T + b
    h, H, D = v.hidden, v.heads, v.head_dim
    size = img.shape[-1]
    x = lin(q("pe_in", patchify(img, v.patch)), p["pe.w"], p["pe.b"])
    x = torch.cat([p["cls"][None], p["reg"], x])                     # (prefix + patches, hidden)
    cos, sin = rope_tables(v, size)
    n = x.shape[0]
    P = v.prefix

    def rope(t):                                                     # patch tokens only
        tp = t[:, P:]
        rot = torch.cat([tp[..., :D // 2] * cos - tp[..., D // 2:] * sin,
                         tp[..., D // 2:] * cos + tp[..., :D // 2] * sin], -1)
        return torch.cat([t[:, :P], rot], 1)

    for l in range(v.layers):
        a = q(f"{l}.ln1", ln(x, p[f"{l}.ln1.w"], p[f"{l}.ln1.b"], v.eps))
        qkv = lin(a, p[f"{l}.qkv.w"], p[f"{l}.qkv.b"]).reshape(n, 3, H, D).permute(1, 2, 0, 3)
        o = attention(rope(qkv[0]), rope(qkv[1]), qkv[2], D, q, str(l)).transpose(0, 1).reshape(n, h)
        x = x + p[f"{l}.ls1"] * lin(q(f"{l}.attn", o), p[f"{l}.o.w"], p[f"{l}.o.b"])
        m = q(f"{l}.ln2", ln(x, p[f"{l}.ln2.w"], p[f"{l}.ln2.b"], v.eps))
        x = x + p[f"{l}.ls2"] * lin(q(f"{l}.gelu", act(lin(m, p[f"{l}.up.w"], p[f"{l}.up.b"]))),
                                    p[f"{l}.down.w"], p[f"{l}.down.b"])
    tokens = ln(x, p["norm.w"], p["norm.b"], v.eps)[P:]              # patch tokens

    qp = lin(q("pool.qin", p["pool.query"]), p["pool.q.w"], p["pool.q.b"])
    kv = lin(q("pool.kvin", tokens), p["pool.kv.w"], p["pool.kv.b"])
    k, val = kv[:, :h], kv[:, h:]
    split = lambda t: t.reshape(t.shape[0], H, D).transpose(0, 1)
    o = attention(split(qp), split(k), split(val), D, q, "pool").transpose(0, 1).reshape(-1, h)
    pooled = lin(q("pool.attn", o), p["pool.out.w"], p["pool.out.b"])
    mm = q("pool.ln", ln(pooled, p["pool.ln.w"], p["pool.ln.b"], 1e-5))
    pooled = pooled + lin(q("pool.gelu", act(lin(mm, p["pool.up.w"], p["pool.up.b"]))), p["pool.down.w"], p["pool.down.b"])
    return lin(q("proj_in", pooled), p["proj.w"], p["proj.b"]).flatten()[None]


def encode(p: dict, state: torch.Tensor, views: list[torch.Tensor], v: VisionDims, **kw) -> torch.Tensor:
    return torch.cat([state] + [forward(p, img, v, **kw) for img in views], dim=-1)

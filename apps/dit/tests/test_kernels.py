"""Each kernel in the functional model against PyTorch on the same FP8 operands."""
import math

import pytest
import torch
import torch.nn.functional as F

from ditflow import layout as L
from ditflow import quant as Q
from ditflow.harness import KernelRun
from ditflow.kernels import Linear


def rel(a, r):
    return float((a.float() - r.float()).norm() / r.float().norm())


def operands(rows, K, N, seed):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(rows, K, generator=g)
    W = (torch.rand(N, K, generator=g) * 2 - 1) / math.sqrt(K)
    bias = torch.randn(N, generator=g) * 0.1
    a_x, a_w = Q.exponent_for(float(X.abs().max()), 2.0), Q.exponent_for(float(W.abs().max()))
    return Q.to_fp8(X, a_x), a_x, Q.to_fp8(W, a_w), a_w, bias


@pytest.mark.parametrize("rows,K,N,resident", [(50, 256, 96, True), (50, 1024, 64, False), (32, 128, 256, True)])
def test_linear(rows, K, N, resident):
    Xq, a_x, Wq, a_w, bias = operands(rows, K, N, seed=K + N)
    kr = KernelRun()
    xq = kr.input_fp8("xq", Xq)
    lin = Linear("w", kr.b.dram.add("w", L.weight_blocks(Wq, bias)), L.tiles(N), L.tiles(K), True, a_w)
    y = kr.tensor("y", rows, N, "bf16")
    kr.k.linear(lin, xq, a_x + a_w, kr.k.epi_store(y), resident=resident)
    kr.output(y)
    got = kr.run().read("y")
    ref = Q.from_fp8(Xq, a_x) @ Q.from_fp8(Wq, a_w).T + bias
    assert rel(got, ref) < 0.01


def test_linear_gelu_quant_epilogue():
    Xq, a_x, Wq, a_w, bias = operands(50, 256, 128, seed=5)
    ref = F.gelu(Q.from_fp8(Xq, a_x) @ Q.from_fp8(Wq, a_w).T + bias, approximate="tanh")
    a_g = Q.exponent_for(float(ref.abs().max()), 2.0)
    kr = KernelRun()
    xq = kr.input_fp8("xq", Xq)
    lin = Linear("w", kr.b.dram.add("w", L.weight_blocks(Wq, bias)), 4, 8, True, a_w)
    g = kr.tensor("g", 50, 128, "fp8")
    kr.k.linear(lin, xq, a_x + a_w, kr.k.epi_quant(g, a_g + 1, pre=lambda o, r: kr.k.gelu2(o)))
    kr.output(g)
    got = Q.from_fp8(kr.run().read("g"), a_g)
    assert rel(got, ref) < 0.05


def test_layernorm_modulate():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(50, 256, generator=g) * 2 + 0.5
    scale1, shift = 1 + torch.randn(256, generator=g) * 0.3, torch.randn(256, generator=g) * 0.3
    ref = F.layer_norm(x.to(torch.bfloat16).float(), (256,), eps=1e-6) * scale1 + shift
    a = Q.exponent_for(float(ref.abs().max()), 2.0)
    kr = KernelRun()
    xt = kr.input_bf16("x", x)
    s1, sh = kr.input_bf16("s1", L.replicated(scale1)), kr.input_bf16("sh", L.replicated(shift))
    out = kr.tensor("out", 50, 256, "fp8")
    kr.k.ln_modulate_quant(xt, s1, sh, out, a)
    kr.output(out)
    got = Q.from_fp8(kr.run().read("out"), a)
    assert rel(got, Q.fake(ref, a)) < 0.04 and rel(got, ref) < 0.06


def test_attention():
    from ditflow import model as M
    d = M.SMALL
    g = torch.Generator().manual_seed(4)
    T, H, D = d.horizon, d.hidden, d.head_dim
    qkv = torch.randn(T, 3 * H, generator=g) * 1.5
    cos, sin = M.rope_tables(d)
    qh, kh, vh = qkv.reshape(T, 3, d.heads, D).permute(1, 2, 0, 3)

    def rope(v):
        return torch.cat([v[..., :D // 2] * cos - v[..., D // 2:] * sin, v[..., D // 2:] * cos + v[..., :D // 2] * sin], -1)

    qs, ks = rope(qh) / math.sqrt(D), rope(kh)
    amax = lambda v: float(v.abs().max())
    a_q, a_k, a_v, a_p = (Q.exponent_for(amax(v), 2.0) for v in (qs, ks, vh, torch.ones(1)))
    probs = torch.softmax(Q.fake(qs, a_q) @ Q.fake(ks, a_k).transpose(-1, -2), -1)
    ref = (Q.fake(probs, a_p) @ Q.fake(vh, a_v)).transpose(0, 1).reshape(T, H)
    a_o = Q.exponent_for(amax(ref), 2.0)

    kr = KernelRun()
    qkv_t = kr.input_bf16("qkv", qkv)
    ct = kr.input_bf16("cos", cos)
    st = kr.input_bf16("sin", sin)
    mask = torch.zeros(32, 32)
    mask[:, T - 32:] = -30000.0
    mt = kr.input_bf16("mask", mask)
    out = kr.tensor("attn", T, H, "fp8")
    kr.k.attention(qkv_t, ct, st, mt, out, d.heads, a_q, a_k, a_v, a_p, a_o)
    kr.output(out)
    got = Q.from_fp8(kr.run().read("attn"), a_o)
    # FP8 attention probabilities dominate the error; Atlas adds BF16 softmax rounding on top.
    exact = (torch.softmax(qs @ ks.transpose(-1, -2), -1) @ vh).transpose(0, 1).reshape(T, H)
    assert rel(got, exact) < 1.2 * rel(Q.fake(ref, a_o), exact) + 0.01

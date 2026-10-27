"""Looped kernels (loopk.py) in the functional model against PyTorch."""
import math

import pytest
import torch
import torch.nn.functional as F

from ditflow import layout as L
from ditflow import quant as Q
from ditflow.harness import KernelRun
from ditflow.kernels import Linear
from ditflow.loopk import LoopKernels, Stream


def rel(a, r):
    return float((a.float() - r.float()).norm() / r.float().norm())


@pytest.mark.parametrize("rows,K,N", [(128, 192, 192), (100, 768, 96), (64, 192, 160)])
def test_linear_rows(rows, K, N):
    g = torch.Generator().manual_seed(rows + K)
    X = torch.randn(rows, K, generator=g)
    W = (torch.rand(N, K, generator=g) * 2 - 1) / math.sqrt(K)
    bias = torch.randn(N, generator=g) * 0.1
    a_x, a_w = Q.exponent_for(float(X.abs().max()), 2.0), Q.exponent_for(float(W.abs().max()))
    Xq, Wq = Q.to_fp8(X, a_x), Q.to_fp8(W, a_w)
    res = torch.randn(rows, N, generator=g)
    scale = 0.5 + torch.rand(N, generator=g)
    kr = KernelRun()
    lk = LoopKernels(kr.b, kr.k)
    xq = kr.input_fp8("xq", L.pad2(Xq, -(-rows // 64) * 64, K))
    lin = Linear("w", kr.b.dram.add("w", L.weight_blocks(Wq, bias)), L.tiles(N), L.tiles(K), True, a_w)
    x = kr.input_bf16("x", L.pad2(res, xq.rows, N))
    sc = kr.input_bf16("sc", L.replicated(scale))
    lk.linear_rows(lin, lin.addr, xq, a_x + a_w, [Stream("x", x), Stream("sc", sc, per_row=False)],
                   lk.epi_residual_scale("x", "sc"))
    kr.output(x)
    got = kr.run().read("x")[:rows]
    ref = res.to(torch.bfloat16).float() + scale * (Q.from_fp8(Xq, a_x) @ Q.from_fp8(Wq, a_w).T + bias)
    assert rel(got, ref) < 0.01


def test_linear_rows_split():
    """Input and weights as two FP8 terms each (W_hi x_hi + W_hi x_lo + W_lo x_hi): the error of a
    BF16 matmul, not of an FP8 one."""
    g = torch.Generator().manual_seed(8)
    rows, K, N = 64, 768, 64
    X = torch.randn(rows, K, generator=g)
    W = (torch.rand(N, K, generator=g) * 2 - 1) / math.sqrt(K)
    a_x, a_w = Q.exponent_for(float(X.abs().max()), 2.0), Q.exponent_for(float(W.abs().max()))
    Wq = Q.to_fp8(W, a_w)
    Wlo = Q.to_fp8(W - Q.from_fp8(Wq, a_w), a_w)
    kr = KernelRun(stage_bytes=48 * 1024)
    lk = LoopKernels(kr.b, kr.k)
    x = kr.input_bf16("x", X)
    hi, lo = kr.tensor("hi", rows, K, "fp8"), kr.tensor("lo", rows, K, "fp8")
    kr.k.quantize(x, hi, a_x, lo)
    lin = Linear("w", kr.b.dram.add("w", L.weight_blocks(Wq, None, Wlo)), L.tiles(N), L.tiles(K), False, a_w, 2)
    y = kr.tensor("y", rows, N, "bf16")
    lk.linear_rows(lin, lin.addr, (hi, lo), a_x + a_w, [Stream("y", y)], lk.epi_store("y"))
    kr.output(y)
    got = kr.run().read("y")
    ref = X @ W.T
    assert rel(got, ref) < 0.006 < rel(Q.fake(X, a_x) @ Q.from_fp8(Wq, a_w).T, ref)


def test_layernorm_rows():
    g = torch.Generator().manual_seed(9)
    x = torch.randn(128, 192, generator=g) * 3 + 1
    w, b = 1 + torch.randn(192, generator=g) * 0.2, torch.randn(192, generator=g) * 0.1
    ref = F.layer_norm(x.to(torch.bfloat16).float(), (192,), w, b, 1e-5)
    a = Q.exponent_for(float(ref.abs().max()), 2.0)
    kr = KernelRun()
    lk = LoopKernels(kr.b, kr.k)
    xt, wt, bt = kr.input_bf16("x", x), kr.input_bf16("w", L.replicated(w)), kr.input_bf16("b", L.replicated(b))
    out = kr.tensor("out", 128, 192, "fp8")
    lk.layernorm_rows(xt, wt, bt, out, a, 1e-5)
    kr.output(out)
    got = Q.from_fp8(kr.run().read("out"), a)
    assert rel(got, Q.fake(ref, a)) < 0.04


def _scratch(kr, rq, rk):
    return {"qbuf": kr.tensor("qbuf", rq * 32, 64, "fp8"), "kbuf": kr.tensor("kbuf", rk * 32, 64, "fp8"),
            "vtbuf": kr.tensor("vtbuf", 64, rk * 32, "fp8"), "pbuf": kr.tensor("pbuf", 32, rk * 32, "fp8"),
            "sbuf": kr.tensor("sbuf", 32, rk * 32, "bf16")}


@pytest.mark.parametrize("cross", [False, True])
def test_attention_rows(cross):
    g = torch.Generator().manual_seed(11 + cross)
    T, H, D, heads, valid = 128, 192, 64, 3, 101
    kv = torch.randn(T, 3 * H, generator=g)
    q_src = torch.randn(32, H, generator=g) if cross else kv[:, :H]
    nq = 4 if cross else T
    ang = torch.rand(T, D // 2, generator=g) * 6
    cos, sin = ang.cos(), ang.sin()
    split = lambda t: t.reshape(t.shape[0], heads, D).transpose(0, 1)
    qh, kh, vh = split(q_src[:nq]), split(kv[:, H:2 * H]), split(kv[:, 2 * H:])
    if not cross:
        rot = lambda t: torch.cat([t[..., :32] * cos - t[..., 32:] * sin, t[..., 32:] * cos + t[..., :32] * sin], -1)
        qh, kh = rot(qh), rot(kh)
    qs = qh / 8
    amax = lambda t: float(t.abs().max())
    a_q, a_k, a_v, a_p = (Q.exponent_for(amax(t), 2.0) for t in (qs, kh, vh, torch.ones(1)))
    s = Q.fake(qs, a_q) @ Q.fake(kh, a_k).transpose(-1, -2)
    s[..., valid:] = -1e9
    exact_s = qs @ kh.transpose(-1, -2)
    exact_s[..., valid:] = -1e9
    exact = (torch.softmax(exact_s, -1) @ vh).transpose(0, 1).reshape(nq, H)
    ref = (Q.fake(torch.softmax(s, -1), a_p) @ Q.fake(vh, a_v)).transpose(0, 1).reshape(nq, H)
    a_o = Q.exponent_for(amax(ref), 2.0)
    masks = torch.zeros(32, T)
    masks[:, valid:] = -30000.0
    kr = KernelRun()
    lk = LoopKernels(kr.b, kr.k)
    kvt = kr.input_bf16("kv", kv)
    qt = kr.input_bf16("q", q_src) if cross else kvt
    ct, st = kr.input_bf16("cos", cos), kr.input_bf16("sin", sin)
    mt = kr.input_bf16("masks", masks)
    out = kr.tensor("attn", qt.rows, H, "fp8")
    lk.attention_rows(qt, 0, kvt, H // 32, 2 * H // 32, heads, None if cross else (ct, st), mt, out,
                      a_q, a_k, a_v, a_p, a_o, _scratch(kr, qt.rt, kvt.rt))
    kr.output(out)
    got = Q.from_fp8(kr.run().read("attn"), a_o)[:nq]
    assert rel(got, exact) < 1.2 * rel(Q.fake(ref, a_o), exact) + 0.01

"""The study's quantizers against the published recipes, and its matmuls against the bit-exact models."""
import math

import numpy as np
import torch

from ditflow import quant
from npu_model.configs import fp8_mma, nvfp4
from quantstudy import formats as F
from quantstudy.study import e2m1_codes, nv_operand


def _round_e2m1(v: float) -> float:
    """ModelOpt's fp4_round_magnitude, literally."""
    a = abs(v)
    q = (0.0 if a <= 0.25 else 0.5 if a < 0.75 else 1.0 if a <= 1.25 else 1.5 if a < 1.75 else
         2.0 if a <= 2.5 else 3.0 if a < 3.5 else 4.0 if a <= 5.0 else 6.0)
    return math.copysign(q, v)


def test_e2m1_grid_ties_and_saturation():
    x = torch.tensor([0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, 0.3, 2.9, 5.01, 100, -0.75, -5, -7] + list(F.E2M1))
    assert F.e2m1(x).tolist() == [_round_e2m1(v) for v in x.tolist()]
    assert F.e2m1(x)[:8].tolist() == [0, 0, 1, 1, 2, 2, 4, 4]


def test_nvfp4_matches_the_recipe():
    """x = E2M1 x E4M3(amax_block / (6 g)) x g, g = amax / (6 x 448), from the formulas of
    ModelOpt (nvfp4_quant.fp8_quantize_scale, nvfp4_scalar_quant) and TE's NVFP4 recipe."""
    x = torch.randn(8, 64) * torch.exp2(torch.randint(-12, 4, (8, 1)).float())
    x[0, :16] = 0
    amax = float(x.abs().max()) * 1.5
    for pow2 in (False, True):
        g = amax / 2688 if not pow2 else 2.0 ** math.ceil(math.log2(amax / 2688))
        out = F.nvfp4(x, amax, pow2).tolist()
        for r in range(8):
            for b in range(4):
                blk = x[r, 16 * b:16 * b + 16].tolist()
                s = float(torch.tensor(min(max(map(abs, blk)) / (6 * g), 448.0)).to(torch.float8_e4m3fn).float())
                sg = float(torch.tensor(s * g, dtype=torch.float32))
                ref = [0.0 if s == 0 else _round_e2m1(float(torch.tensor(v / sg, dtype=torch.float32))) * sg for v in blk]
                assert np.allclose(out[r][16 * b:16 * b + 16], ref, rtol=1e-6, atol=0), (pow2, r, b)


def test_mx_scale_rule():
    x = torch.randn(4, 64) * 10
    for fn, emax in ((F.mxfp4, 6.0), (F.mxfp8, 448.0)):
        q = fn(x).unflatten(-1, (-1, 32))
        b = x.unflatten(-1, (-1, 32))
        s = torch.exp2(torch.ceil(torch.log2(b.abs().amax(-1, keepdim=True) / emax)))
        assert torch.all(q.abs() <= emax * s)
        assert torch.equal(q / s, F.e2m1(b / s) if emax == 6.0 else F.e4m3(b / s))


def test_fp8_flushes_only_on_atlas():
    x = torch.tensor([2.0 ** -8, 448.0, 1000.0])
    assert F.fp8(x, 448.0, flush=True).tolist() == [0.0, 448.0, 448.0]
    assert F.fp8(x, 448.0).tolist() == [2.0 ** -8, 448.0, 448.0]


def test_fp32_emulation_against_blackwell_fp8_and_atlas_nvfp4():
    """An FP32 matmul of the dequantized operands is within FP32 rounding of the bit-exact dot
    products (Blackwell FP8: fp8_mma; Atlas NVFP4 MXU: nvfp4), relative to sum |a w|."""
    g = torch.Generator().manual_seed(4)
    a = torch.randn(4, 128, generator=g) * 3
    w = torch.randn(8, 128, generator=g) * 0.1
    ca = (a * 2.0 ** -quant.exponent_for(float(a.abs().max()))).to(torch.float8_e4m3fn)
    cw = (w * 2.0 ** -quant.exponent_for(float(w.abs().max()))).to(torch.float8_e4m3fn)
    acc = np.zeros((4, 8), np.uint32)
    for k in range(0, 128, 32):
        acc = fp8_mma.dot(ca.view(torch.uint8).numpy()[:, None, k:k + 32], cw.view(torch.uint8).numpy()[None, :, k:k + 32], acc)
    exact = torch.from_numpy(acc.view(np.float32).copy())
    bound = ca.float().abs() @ cw.float().abs().T * 2.0 ** -16  # K u + the cuts, K = 128
    assert torch.all((ca.float() @ cw.float().T - exact).abs() <= bound)

    qa, sa, ga = nv_operand(a, float(a.abs().max()))
    qw, sw, gw = nv_operand(w, float(w.abs().max()))
    exact = torch.tensor([[0] * 8 for _ in range(4)])
    for k in range(2):
        exact = torch.tensor([[nvfp4.dot(qa[r][64 * k:64 * k + 64], qw[j][64 * k:64 * k + 64], sa[r][4 * k:4 * k + 4],
                                         sw[j][4 * k:4 * k + 4], int(exact[r, j])) for j in range(8)] for r in range(4)])
    exact = exact.to(torch.int32).view(torch.float32) * ga * gw
    da, dw = F.nvfp4(a, float(a.abs().max())), F.nvfp4(w, float(w.abs().max()))
    assert torch.all((da @ dw.T - exact).abs() <= da.abs() @ dw.abs().T * 2.0 ** -16)
    assert e2m1_codes(torch.tensor(F.E2M1 + tuple(-v for v in F.E2M1))).tolist() == list(range(8)) + [0] + list(range(9, 16))

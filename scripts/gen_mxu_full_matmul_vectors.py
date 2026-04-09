#!/usr/bin/env python3
"""
gen_mxu_full_matmul_vectors.py: generate full-matmul MXU test vectors using
the SA (systolic array) and IPT functional models.

Both models compute F.linear: y = x @ w^T
Both accept full matrices and handle tiling internally.
Test data is generated using the model's own E4M3 LUT to ensure
the float→byte round-trip is consistent.

Produces two files:
  1. sa_<base>.txt   – expected values from SARTLLinearFunction
  2. ipt_<base>.txt  – expected values from IPTLinearRTLFunction

Usage:
    python3 scripts/gen_mxu_full_matmul_vectors.py
"""

import argparse, struct, os, sys
import numpy as np
from numpy.random import default_rng
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baremetal", "generators"))
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.ipt_rtl_linear import (
    IPTLinearRTLFunction,
    _E4M3_FLOAT_LUT,
)
from software_models.mxu1_ipt.fp_formats import OutputFmtSel

# ── Build pool of valid E4M3 (byte, float) pairs from the model's own LUT ──

def build_e4m3_pool():
    valid_bytes = []
    valid_floats = []
    for b in range(256):
        f = _E4M3_FLOAT_LUT[b].item()
        if b == 0 or f != 0.0:
            valid_bytes.append(b)
            valid_floats.append(f)
    return np.array(valid_bytes, dtype=np.uint8), np.array(valid_floats, dtype=np.float32)

POOL_BYTES, POOL_FLOATS = build_e4m3_pool()

# ── helpers ────────────────────────────────────────────────────────────────

def float32_to_bf16_hex(x):
    return f"{(struct.unpack('>I', struct.pack('>f', float(x)))[0] >> 16) & 0xFFFF:04x}"

def sample_e4m3(rng, n):
    """Return (hex_list, float_tensor) for n random E4M3 values."""
    idxs = rng.integers(0, len(POOL_BYTES), size=n)
    bytes_arr = POOL_BYTES[idxs]
    floats_arr = POOL_FLOATS[idxs]
    hex_list = [f"{b:02x}" for b in bytes_arr]
    return hex_list, torch.from_numpy(floats_arr.copy())

# ── compute expected using full matrices ───────────────────────────────────

def compute_expected(model, ta_f32, tw_f32, args):
    """
    Build full [num_rows, inner_dim] and [num_lanes, inner_dim] matrices
    from per-tile float tensors, pass to model which handles tiling internally.

    Both models compute F.linear: y = x @ w^T
      x = A_full:  [num_rows, inner_dim]
      w = W_full:  [num_lanes, inner_dim]
      y:           [num_rows, num_lanes]
    """
    nt = len(ta_f32)

    # Full weight matrix: [num_lanes, inner_dim]
    w_full = torch.zeros(args.num_lanes, args.inner_dim)
    for t in range(nt):
        k0 = t * args.vec_len
        w_full[:, k0:k0 + args.vec_len] = tw_f32[t]

    # Full activation matrix: [num_rows, inner_dim]
    a_full = torch.zeros(args.num_rows, args.inner_dim)
    for t in range(nt):
        k0 = t * args.vec_len
        a_full[:, k0:k0 + args.vec_len] = ta_f32[t]

    # Both models: y = x @ w^T, same call convention
    # Invalidate model weight cache to prevent stale data — the model
    # caches E4M3-encoded weights keyed by (data_ptr, shape, _version),
    # which can collide when PyTorch reuses memory addresses.
    if hasattr(model, '_w_cache_key'):
        model._w_cache_key = None
    if hasattr(model, '_prepared_cache'):
        model._prepared_cache = None

    out = model(a_full, w_full, b_q=None, scale_exp=0)  # [num_rows, num_lanes]

    exp_hex = []
    for i in range(args.num_rows):
        exp_hex.append([float32_to_bf16_hex(out[i, r].item()) for r in range(args.num_lanes)])
    return exp_hex

# ── write output file ─────────────────────────────────────────────────────

def write_vectors(path, cases):
    with open(path, "w") as f:
        for case in cases:
            cid = case["cid"]
            num_rows = case["num_rows"]
            nt = case["num_tiles"]
            tw_hex = case["tw_hex"]
            ta_hex = case["ta_hex"]
            exp = case["exp"]
            f.write(f"# {cid} full_matmul\nnum_rows {num_rows}\nnum_tiles {nt}\n")
            for t in range(nt):
                for r, row in enumerate(tw_hex[t]):
                    f.write(f"wgt {t} {r} {' '.join(row)}\n")
                for i, row in enumerate(ta_hex[t]):
                    f.write(f"act {t} {i} {' '.join(row)}\n")
            for i, row in enumerate(exp):
                f.write(f"exp {i} {' '.join(row)}\n")
            f.write("\n")
    print(f"Wrote {len(cases)} cases to {path}")

# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="src/test/resources/mxu_test_vectors/mxu_full_matmul_vectors")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--num-rows", type=int, default=32)
    ap.add_argument("--vec-len", type=int, default=32)
    ap.add_argument("--num-lanes", type=int, default=32)
    ap.add_argument("--inner-dim", type=int, default=16384)
    args = ap.parse_args()

    assert args.inner_dim % args.vec_len == 0
    nt = args.inner_dim // args.vec_len
    rng = default_rng(args.seed)

    print(f"Generating {args.num} cases: [{args.num_rows}x{args.inner_dim}] x "
          f"[{args.inner_dim}x{args.num_lanes}], {nt} tiles")

    # SA model — matches doc 21: rows=32, cols=32
    sa_model = SARTLLinearFunction(
        rows=32, cols=32,
        out_fmt_sel=OutputFmtSel.OutBF16,
    )
    # IPT model — matches doc 20
    ipt_model = IPTLinearRTLFunction(
        vec_len=32, num_lanes=32, pipeline_depth=1,
        out_fmt_sel=OutputFmtSel.OutBF16,
    )

    sa_cases = []
    ipt_cases = []

    for cid in range(args.num):
        tw_hex = []
        ta_hex = []
        tw_f32 = []
        ta_f32 = []

        for _ in range(nt):
            tile_wgt_hex = []
            tile_wgt_f32 = []
            for _ in range(args.num_lanes):
                wh, wf = sample_e4m3(rng, args.vec_len)
                tile_wgt_hex.append(wh)
                tile_wgt_f32.append(wf)
            tw_hex.append(tile_wgt_hex)
            tw_f32.append(torch.stack(tile_wgt_f32, dim=0))

            tile_act_hex = []
            tile_act_f32 = []
            for _ in range(args.num_rows):
                ah, af = sample_e4m3(rng, args.vec_len)
                tile_act_hex.append(ah)
                tile_act_f32.append(af)
            ta_hex.append(tile_act_hex)
            ta_f32.append(torch.stack(tile_act_f32, dim=0))

        exp_sa = compute_expected(sa_model, ta_f32, tw_f32, args)
        exp_ipt = compute_expected(ipt_model, ta_f32, tw_f32, args)

        sa_cases.append({"cid": cid, "num_rows": args.num_rows, "num_tiles": nt,
                         "tw_hex": tw_hex, "ta_hex": ta_hex, "exp": exp_sa})
        ipt_cases.append({"cid": cid, "num_rows": args.num_rows, "num_tiles": nt,
                          "tw_hex": tw_hex, "ta_hex": ta_hex, "exp": exp_ipt})

    stripped = args.out.removesuffix(".txt")
    outdir = os.path.dirname(stripped)
    base = os.path.basename(stripped)

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    def outpath(name):
        return os.path.join(outdir, name) if outdir else name

    write_vectors(outpath(f"sa_{base}.txt"), sa_cases)
    write_vectors(outpath(f"ipt_{base}.txt"), ipt_cases)

if __name__ == "__main__":
    main()
    
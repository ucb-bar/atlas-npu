#!/usr/bin/env python3
"""
gen_mxu_vectors.py: generate MXU test vectors using SA and IPT functional models,
then convert expected outputs to E4M3 with scaling.

Test data is generated using the IPT model's own E4M3 LUT so that
the float→byte round-trip inside the model is consistent with the hex values
written to the test vector files.

Produces four files:
  1. sa_<base>.txt          – BF16-expected vectors (SA model)
  2. ipt_<base>.txt         – BF16-expected vectors (IPT model)
  3. sa_<base>_e4m3.txt     – E4M3-converted (SA model)
  4. ipt_<base>_e4m3.txt    – E4M3-converted (IPT model)

Usage:
    python3 scripts/gen_mxu_vectors.py
"""

import argparse, math, struct, os, sys
import numpy as np
from numpy.random import default_rng
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baremetal", "generators"))
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.ipt_rtl_linear import (
    IPTLinearRTLFunction,
    _E4M3_FLOAT_LUT,
)
from software_models.mxu1_ipt.converters import bf16_scale_to_e4m3
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

def float32_to_bf16_hex(x: np.float32) -> str:
    fp32_bits = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    return f"{(fp32_bits >> 16) & 0xFFFF:04x}"

def bf16_hex_to_float(h: str) -> float:
    return struct.unpack(">f", struct.pack(">I", (int(h, 16) & 0xFFFF) << 16))[0]

def float_to_bf16_bits(x: float) -> int:
    f32 = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    return ((f32 + 0x7FFF + ((f32 >> 16) & 1)) >> 16) & 0xFFFF

def sample_e4m3(rng, n):
    """Return (hex_list, float_tensor) for n random E4M3 values."""
    idxs = rng.integers(0, len(POOL_BYTES), size=n)
    bytes_arr = POOL_BYTES[idxs]
    floats_arr = POOL_FLOATS[idxs]
    hex_list = [f"{b:02x}" for b in bytes_arr]
    return hex_list, torch.from_numpy(floats_arr.copy())

# ── compute expected for all lanes via functional model ────────────────────

def compute_expected_row(model, act_f32, wgt_f32, sel, bias_f32):
    """
    Both models use F.linear convention: y = x @ w^T + b
      act_f32:  [1, vec_len]
      wgt_f32:  [num_lanes, vec_len]
      bias_f32: [num_lanes] (for sel=1) or None
    """
    if sel == 1:
        out = model(act_f32, wgt_f32, b_q=bias_f32, scale_exp=0).squeeze(0)
    else:
        out = model(act_f32, wgt_f32, b_q=None, scale_exp=0).squeeze(0)

    result = out.cpu().numpy().astype(np.float32)
    return [float32_to_bf16_hex(np.float32(v)) for v in result]

# ── generate BF16-expected vectors ─────────────────────────────────────────

def generate_vectors(path, model, args):
    rng = default_rng(args.seed)
    case_types = ["basic", "bias"]
    with open(path, "w") as f:
        for i in range(args.num):
            ct = case_types[i % 2]

            act_hex, act_f32 = sample_e4m3(rng, args.vec_len)
            act_f32 = act_f32.unsqueeze(0)  # [1, vec_len]

            wgt_hex_all = []
            wgt_f32_rows = []
            for _ in range(args.num_lanes):
                wh, wf = sample_e4m3(rng, args.vec_len)
                wgt_hex_all.append(wh)
                wgt_f32_rows.append(wf)
            wgt_f32 = torch.stack(wgt_f32_rows, dim=0)  # [num_lanes, vec_len]

            sel = 0
            bias_hex = ["00"] * args.num_lanes
            bias_f32 = None

            if ct == "bias":
                sel = 1
                bh, bf = sample_e4m3(rng, args.num_lanes)
                bias_hex = bh
                bias_f32 = bf  # [num_lanes]

            exp_hex = compute_expected_row(model, act_f32, wgt_f32, sel, bias_f32)

            f.write(f"# {i} {ct}\nsel {sel}\nact {' '.join(act_hex)}\n")
            for r, row in enumerate(wgt_hex_all):
                f.write(f"wgt {r} {' '.join(row)}\n")
            f.write(f"bias {' '.join(bias_hex)}\nexp {' '.join(exp_hex)}\n\n")
    print(f"Wrote {args.num} vectors to {path}")

# ── convert exp lines to E4M3 with scaling ─────────────────────────────────

def decode_e4m3_hw(b: int) -> float:
    return float(_E4M3_FLOAT_LUT[b & 0xFF].item())

def choose_dequant_scale(vals):
    mx = max(abs(v) for v in vals) if vals else 0.0
    if mx == 0.0:
        return 1.0
    raw = mx / 448.0
    return 1.0 if raw <= 1.0 else 2.0 ** math.ceil(math.log2(raw))

def convert_file(inp: str, outp: str):
    with open(inp) as f:
        lines = f.readlines()
    out = []
    for line in lines:
        s = line.strip()
        if not s.startswith("exp "):
            out.append(line)
            continue
        hexes = s.split()[1:]
        vals = [bf16_hex_to_float(h) for h in hexes]
        ds = choose_dequant_scale(vals)
        qs = 1.0 / ds
        se = int(round(math.log2(qs)))
        # Match the hardware pop path exactly: BF16 -> scaled E4M3 with
        # underflow flushed to zero and no E4M3 subnormal outputs.
        qb = [bf16_scale_to_e4m3(float_to_bf16_bits(v), se) for v in vals]
        out.append(f"scale_exp {(se & 0xFF):02x}    # {se:+d}  (quant_scale={qs:.9g}, dequant_scale={ds:.9g})\n")
        out.append("exp_e4m3 " + " ".join(f"{b:02x}" for b in qb) + "\n")
        recon = [decode_e4m3_hw(b) * ds for b in qb]
        out.append("recon_bf16 " + " ".join(f"{float_to_bf16_bits(x):04x}" for x in recon) + "\n")
    with open(outp, "w") as f:
        f.writelines(out)
    print(f"Wrote E4M3-converted vectors to {outp}")

# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="src/test/resources/mxu_test_vectors/mxu_vectors")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--vec-len", type=int, default=32)
    ap.add_argument("--num-lanes", type=int, default=32)
    args = ap.parse_args()

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

    stripped = args.out.removesuffix(".txt")
    outdir = os.path.dirname(stripped)
    base = os.path.basename(stripped)

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    def outpath(name):
        return os.path.join(outdir, name) if outdir else name

    sa_bf16_path = outpath(f"sa_{base}.txt")
    ipt_bf16_path = outpath(f"ipt_{base}.txt")
    sa_e4m3_path = outpath(f"sa_{base}_e4m3.txt")
    ipt_e4m3_path = outpath(f"ipt_{base}_e4m3.txt")

    generate_vectors(sa_bf16_path, sa_model, args)
    generate_vectors(ipt_bf16_path, ipt_model, args)

    convert_file(sa_bf16_path, sa_e4m3_path)
    convert_file(ipt_bf16_path, ipt_e4m3_path)

if __name__ == "__main__":
    main()

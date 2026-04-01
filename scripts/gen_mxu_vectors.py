#!/usr/bin/env python3
"""
gen_mxu_vectors.py: generate MXU test vectors with PyTorch-based reference

Updated for refactored MXU: default num_lanes=32, vec_len=32

Output format (one block per test):
    # <id> <type>
    sel <0|1|2>
    act <hex> <hex> ...
    wgt <r> <hex> <hex> ...
    bias <hex> <hex> ...
    psum <hex> <hex> ...
    exp <hex> <hex> ...

Usage:
    python3 scripts/gen_mxu_vectors.py --ref fp32_accum --out src/test/resources/ipt_test_vectors/mxu_vectors.txt --num 50 --seed 42
"""

import argparse, struct
import numpy as np
from numpy.random import default_rng
import torch, torch.nn.functional as F

def float32_to_bf16_hex(x: np.float32) -> str:
    fp32_bits = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    return f"{(fp32_bits >> 16) & 0xFFFF:04x}"

def bf16_hex_to_float32(h: str) -> np.float32:
    return np.float32(struct.unpack(">f", struct.pack(">I", (int(h,16)&0xFFFF)<<16))[0])

def float_to_e4m3_hex(x: float, fp8_dtype: torch.dtype) -> str:
    return f"{torch.tensor([x],dtype=torch.float32).to(fp8_dtype).view(torch.uint8).item():02x}"

def all_fp8_values(fp8_dtype: torch.dtype) -> np.ndarray:
    u = torch.arange(256, dtype=torch.uint8)
    exp = (u >> 3) & 0x0F
    man = u & 0x07
    q = u.view(fp8_dtype)
    f = q.to(torch.float32)
    mask = (~torch.isnan(f)) & (~torch.isinf(f)) & (~((exp==0)&(man!=0)))
    return np.unique(f[mask].cpu().numpy().astype(np.float32))

def dot_expected(act_hex, wgt_hex, sel, addend_hex, fp8_dtype, ref):
    act_u8 = torch.tensor([int(h,16)&0xFF for h in act_hex], dtype=torch.uint8)
    wgt_u8 = torch.tensor([int(h,16)&0xFF for h in wgt_hex], dtype=torch.uint8)
    act_fp8 = act_u8.view(fp8_dtype)
    wgt_fp8 = wgt_u8.view(fp8_dtype)
    dt = torch.bfloat16 if ref == "bf16_accum" else torch.float32
    act = act_fp8.to(dt).unsqueeze(0)
    wgt = wgt_fp8.to(dt).unsqueeze(0)
    bias = None
    if sel == 1:
        b_u8 = torch.tensor([int(addend_hex,16)&0xFF], dtype=torch.uint8)
        bias = b_u8.view(fp8_dtype).to(dt)
    elif sel == 2:
        bias = torch.tensor([float(bf16_hex_to_float32(addend_hex))], dtype=dt)
    acc = F.linear(act, wgt, bias).squeeze(0).squeeze(0)
    out = acc.to(torch.bfloat16).to(torch.float32).cpu().numpy().astype(np.float32).item()
    return float32_to_bf16_hex(np.float32(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mxu_vectors.txt")
    ap.add_argument("--num", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--vec-len", type=int, default=32)
    ap.add_argument("--num-lanes", type=int, default=32)
    ap.add_argument("--fp8", choices=["e4m3fn","e4m3fnuz"], default="e4m3fn")
    ap.add_argument("--ref", choices=["bf16_accum","fp32_accum"], default="bf16_accum")
    args = ap.parse_args()
    fp8_dtype = torch.float8_e4m3fn if args.fp8 == "e4m3fn" else torch.float8_e4m3fnuz
    rng = default_rng(args.seed)
    pool = all_fp8_values(fp8_dtype)
    case_types = ["basic", "bias", "psum"]
    with open(args.out, "w") as f:
        for i in range(args.num):
            ct = case_types[i % 3]
            act_vals = pool[rng.integers(0,len(pool),size=args.vec_len)]
            act_hex = [float_to_e4m3_hex(float(v), fp8_dtype) for v in act_vals]
            wgt_hex = [[float_to_e4m3_hex(float(v), fp8_dtype) for v in pool[rng.integers(0,len(pool),size=args.vec_len)]] for _ in range(args.num_lanes)]
            sel = 0
            bias_hex = ["00"]*args.num_lanes
            psum_hex = ["0000"]*args.num_lanes
            if ct == "bias":
                sel = 1
                bias_hex = [float_to_e4m3_hex(float(v), fp8_dtype) for v in pool[rng.integers(0,len(pool),size=args.num_lanes)]]
            elif ct == "psum":
                sel = 2
                ps = torch.from_numpy(rng.uniform(-100,100,size=args.num_lanes).astype(np.float32)).to(torch.bfloat16).to(torch.float32).cpu().numpy()
                psum_hex = [float32_to_bf16_hex(v) for v in ps]
            exp_hex = [dot_expected(act_hex, wgt_hex[r], sel, bias_hex[r] if sel==1 else psum_hex[r], fp8_dtype, args.ref) for r in range(args.num_lanes)]
            f.write(f"# {i} {ct}\nsel {sel}\nact {' '.join(act_hex)}\n")
            for r, row in enumerate(wgt_hex): f.write(f"wgt {r} {' '.join(row)}\n")
            f.write(f"bias {' '.join(bias_hex)}\npsum {' '.join(psum_hex)}\nexp {' '.join(exp_hex)}\n\n")
    print(f"Wrote {args.num} vectors to {args.out} (lanes={args.num_lanes}, ref={args.ref})")

if __name__ == "__main__": 
    main()

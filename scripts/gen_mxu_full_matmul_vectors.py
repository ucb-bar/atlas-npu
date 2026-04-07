#!/usr/bin/env python3
"""
gen_mxu_full_matmul.py: generate full-matmul MXU test vectors

Updated: default num_lanes=32, num_rows=32 (matching tileRows)

Usage:
    python3 scripts/gen_mxu_full_matmul.py --ref fp32_accum --inner-dim 16384 \
      --out src/test/resources/mxu_test_vectors/mxu_full_matmul_vectors.txt
"""

import argparse, struct
import numpy as np
from numpy.random import default_rng
import torch, torch.nn.functional as F

def float32_to_bf16_hex(x): 
    return f"{(struct.unpack('>I',struct.pack('>f',float(x)))[0]>>16)&0xFFFF:04x}"

def bf16_hex_to_float32(h): 
    return np.float32(struct.unpack(">f",struct.pack(">I",(int(h,16)&0xFFFF)<<16))[0])

def float_to_e4m3_hex(x, dt): 
    return f"{torch.tensor([x],dtype=torch.float32).to(dt).view(torch.uint8).item():02x}"

def all_fp8_values(dt):
    u=torch.arange(256,dtype=torch.uint8)
    exp=(u>>3)&0x0F
    man=u&0x07
    q=u.view(dt)
    f=q.to(torch.float32)
    mask=(~torch.isnan(f))&(~torch.isinf(f))&(~((exp==0)&(man!=0)))
    return np.unique(f[mask].cpu().numpy().astype(np.float32))

def dot_bf16(act_hex, wgt_hex, sel, addend_hex, dt8, ref):
    au=torch.tensor([int(h,16)&0xFF for h in act_hex],dtype=torch.uint8)
    wu=torch.tensor([int(h,16)&0xFF for h in wgt_hex],dtype=torch.uint8)
    t=torch.bfloat16 if ref=="bf16_accum" else torch.float32
    a=au.view(dt8).to(t).unsqueeze(0)
    w=wu.view(dt8).to(t).unsqueeze(0)
    b=None
    if sel==1: 
        bu=torch.tensor([int(addend_hex,16)&0xFF],dtype=torch.uint8)
        b=bu.view(dt8).to(t)
    elif sel==2: 
        b=torch.tensor([float(bf16_hex_to_float32(addend_hex))],dtype=t)
    r=F.linear(a,w,b).squeeze(0).squeeze(0).to(torch.bfloat16).to(torch.float32).cpu().numpy().item()
    return float32_to_bf16_hex(np.float32(r))

def sample(rng, n, pool, dt8): 
    return [float_to_e4m3_hex(float(v),dt8) for v in pool[rng.integers(0,len(pool),size=n)]]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="mxu_full_matmul_vectors.txt")
    ap.add_argument("--num",type=int,default=5)
    ap.add_argument("--seed",type=int,default=12345)
    ap.add_argument("--num-rows",type=int,default=32)
    ap.add_argument("--vec-len",type=int,default=32)
    ap.add_argument("--num-lanes",type=int,default=32)
    ap.add_argument("--inner-dim",type=int,default=64)
    ap.add_argument("--fp8",choices=["e4m3fn","e4m3fnuz"],default="e4m3fn")
    ap.add_argument("--ref",choices=["bf16_accum","fp32_accum"],default="bf16_accum")
    args=ap.parse_args()
    assert args.inner_dim%args.vec_len==0
    nt=args.inner_dim//args.vec_len
    dt8=torch.float8_e4m3fn if args.fp8=="e4m3fn" else torch.float8_e4m3fnuz
    rng=default_rng(args.seed)
    pool=all_fp8_values(dt8)
    print(f"Generating {args.num} cases: [{args.num_rows}x{args.inner_dim}] x [{args.inner_dim}x{args.num_lanes}], {nt} tiles")
    with open(args.out,"w") as f:
        for cid in range(args.num):
            tw=[]
            ta=[]
            for _ in range(nt):
                tw.append([sample(rng,args.vec_len,pool,dt8) for _ in range(args.num_lanes)])
                ta.append([sample(rng,args.vec_len,pool,dt8) for _ in range(args.num_rows)])
            ps=[["0000"]*args.num_lanes for _ in range(args.num_rows)]
            for t in range(nt):
                sel=0 if t==0 else 2
                for i in range(args.num_rows):
                    ro=[]
                    for r in range(args.num_lanes):
                        ad="00" if sel==0 else ps[i][r]
                        ro.append(dot_bf16(ta[t][i],tw[t][r],sel,ad,dt8,args.ref))
                    ps[i]=ro
            f.write(f"# {cid} full_matmul\nnum_rows {args.num_rows}\nnum_tiles {nt}\n")
            for t in range(nt):
                for r,row in enumerate(tw[t]): f.write(f"wgt {t} {r} {' '.join(row)}\n")
                for i,row in enumerate(ta[t]): f.write(f"act {t} {i} {' '.join(row)}\n")
            for i,row in enumerate(ps): f.write(f"exp {i} {' '.join(row)}\n")
            f.write("\n")
    print(f"Wrote {args.num} cases to {args.out}")

if __name__=="__main__": 
    main()

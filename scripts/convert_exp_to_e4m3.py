#!/usr/bin/env python3
"""
convert_exp_to_e4m3.py: convert BF16 expected outputs to E4M3 with scaling

Usage:
  python3 scripts/convert_exp_to_e4m3.py \
    src/test/resources/mxu_vectors.txt src/test/resources/ipt_test_vectors/mxu_vectors_e4m3.txt
"""

import argparse, math, struct
from typing import List

def bf16_hex_to_float(h):
    return struct.unpack(">f", struct.pack(">I", (int(h,16)&0xFFFF)<<16))[0]

def float_to_bf16_bits(x):
    f32 = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    return ((f32 + 0x7FFF + ((f32>>16)&1)) >> 16) & 0xFFFF

def decode_e4m3(b):
    b &= 0xFF
    sign = -1.0 if (b&0x80) else 1.0
    exp = (b>>3)&0xF
    mant = b&0x7
    if exp==0: return (sign*(mant/8.0)*(2.0**-6)) if mant else (-0.0 if sign<0 else 0.0)
    if exp==0xF and mant==0x7: return float("nan")
    return sign*(1.0+mant/8.0)*(2.0**(exp-7))

FINITE = [(b, decode_e4m3(b)) for b in range(256) if not math.isnan(decode_e4m3(b))]

def encode_e4m3_nearest(x):
    if math.isnan(x): return 0x7F
    if x == 0.0: return 0x80 if math.copysign(1.0,x)<0 else 0x00
    best_b, best_err = 0, float("inf")
    for b, v in FINITE:
        err = abs(v-x)
        if err < best_err:
            best_err=err
            best_b=b
        elif err == best_err and (b&1)==0 and (best_b&1)!=0: 
            best_b=b
    return best_b

def choose_dequant_scale(vals):
    mx = max(abs(v) for v in vals) if vals else 0.0
    if mx == 0.0: return 1.0
    raw = mx / 448.0
    return 1.0 if raw <= 1.0 else 2.0**math.ceil(math.log2(raw))

def convert_file(inp, outp):
    with open(inp) as f: lines = f.readlines()
    out = []
    for line in lines:
        s = line.strip()
        if not s.startswith("exp "): out.append(line); continue
        hexes = s.split()[1:]
        vals = [bf16_hex_to_float(h) for h in hexes]
        ds = choose_dequant_scale(vals)
        qs = 1.0/ds
        se = int(round(math.log2(qs)))
        qb = [encode_e4m3_nearest(v*qs) for v in vals]
        out.append(f"scale_exp {(se&0xFF):02x}    # {se:+d}  (quant_scale={qs:.9g}, dequant_scale={ds:.9g})\n")
        out.append("exp_e4m3 " + " ".join(f"{b:02x}" for b in qb) + "\n")
        recon = [decode_e4m3(b)*ds for b in qb]
        out.append("recon_bf16 " + " ".join(f"{float_to_bf16_bits(x):04x}" for x in recon) + "\n")
    with open(outp, "w") as f: f.writelines(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt")
    ap.add_argument("output_txt")
    args = ap.parse_args()
    convert_file(args.input_txt, args.output_txt)

if __name__ == "__main__": 
    main()

#!/usr/bin/env python3
"""
gen_mxu_vectors.py: generate MXU test vectors with PyTorch-based reference

π0-like default:
- Decode stored FP8 E4M3 inputs (act/weights/bias) into torch.float8 tensors,
- Cast to BF16, do MAC in BF16, and output BF16.

Output format (one block per test):
    # <id> <type>
    sel <0|1|2>
    act <hex> <hex> ...
    wgt <r> <hex> <hex> ...
    bias <hex> <hex> ...
    psum <hex> <hex> ...
    exp <hex> <hex> ...

Usage:
    python3 scripts/gen_mxu_vectors.py --out src/test/resources/mxu_vectors.txt
    python3 scripts/gen_mxu_vectors.py --out src/test/resources/mxu_vectors.txt --num 50 --seed 42
Optional:
    --ref fp32_accum   (fp32 accumulate then cast to bf16)
    --ref bf16_accum   (bf16 accumulate; DEFAULT, π0-like)
    --fp8 e4m3fn|e4m3fnuz
"""

import argparse
import struct
import numpy as np
from numpy.random import default_rng
import torch


# BF16 hex helpers
def float32_to_bf16_hex(x: np.float32) -> str:
    """Take the top 16 bits of FP32 as BF16 bits (rounding should have happened in torch)."""
    fp32_bits = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    bf16_bits = (fp32_bits >> 16) & 0xFFFF
    return f"{bf16_bits:04x}"

def bf16_hex_to_float32(h: str) -> np.float32:
    bf16_bits = int(h, 16) & 0xFFFF
    fp32_bits = (bf16_bits << 16) & 0xFFFFFFFF
    return np.float32(struct.unpack(">f", struct.pack(">I", fp32_bits))[0])


# FP8 E4M3 hex helpers via torch

def float_to_e4m3_hex_via_torch(x: float, fp8_dtype: torch.dtype) -> str:
    t = torch.tensor([x], dtype=torch.float32)
    q = t.to(fp8_dtype)
    u8 = q.view(torch.uint8).item()
    return f"{u8:02x}"

def all_fp8_values(fp8_dtype: torch.dtype) -> np.ndarray:
    """All representable values excluding NaN, Inf, and FP8 subnormals."""
    u = torch.arange(256, dtype=torch.uint8)

    # Bitfields for E4M3: s eeee mmm
    exp = (u >> 3) & 0x0F
    man = u & 0x07

    is_subnormal = (exp == 0) & (man != 0)

    q = u.view(fp8_dtype)
    f = q.to(torch.float32)

    mask = (~torch.isnan(f)) & (~torch.isinf(f)) & (~is_subnormal)

    vals = f[mask].cpu().numpy().astype(np.float32)
    vals = np.unique(vals)
    vals.sort()
    return vals


# Reference dot products

def dot_expected_bf16_hex(
    act_hex: list[str],
    wgt_hex: list[str],
    sel: int,
    addend_hex: str,              # bias: 1-byte fp8 hex when sel=1, psum: 2-byte bf16 hex when sel=2
    fp8_dtype: torch.dtype,
    ref: str,                     # "bf16_accum" (default) or "fp32_accum"
) -> str:
    # reinterpret stored FP8 bit patterns exactly.
    act_u8 = torch.tensor([int(h, 16) & 0xFF for h in act_hex], dtype=torch.uint8)
    wgt_u8 = torch.tensor([int(h, 16) & 0xFF for h in wgt_hex], dtype=torch.uint8)
    act_fp8 = act_u8.view(fp8_dtype)
    wgt_fp8 = wgt_u8.view(fp8_dtype)

    if ref == "bf16_accum":
        # π0-like: we want the numerical behavior to look like BF16 MACs, even though the backend may secretly use higher precision fused implementations.
        act = act_fp8.to(torch.bfloat16)
        wgt = wgt_fp8.to(torch.bfloat16)

        acc = (act * wgt).sum(dtype=torch.bfloat16)

        if sel == 1:
            b_u8 = torch.tensor([int(addend_hex, 16) & 0xFF], dtype=torch.uint8)
            b = b_u8.view(fp8_dtype).to(torch.bfloat16).squeeze(0)
            acc = (acc + b).to(torch.bfloat16)
        elif sel == 2:
            psum_fp32 = bf16_hex_to_float32(addend_hex)
            psum_bf16 = torch.tensor(float(psum_fp32), dtype=torch.bfloat16)
            acc = (acc + psum_bf16).to(torch.bfloat16)

        out_bf16 = acc.to(torch.bfloat16)

    elif ref == "fp32_accum":
        # baseline: fp32 accumulate, then cast to bf16.
        act = act_fp8.to(torch.float32)
        wgt = wgt_fp8.to(torch.float32)

        acc = (act * wgt).sum(dtype=torch.float32)

        if sel == 1:
            b_u8 = torch.tensor([int(addend_hex, 16) & 0xFF], dtype=torch.uint8)
            b = b_u8.view(fp8_dtype).to(torch.float32).squeeze(0)
            acc = acc + b
        elif sel == 2:
            psum_fp32 = bf16_hex_to_float32(addend_hex)
            acc = acc + torch.tensor(float(psum_fp32), dtype=torch.float32)

        out_bf16 = acc.to(torch.bfloat16)

    else:
        raise ValueError(f"Unknown --ref {ref}")

    # export BF16 bits: convert bf16 -> fp32 (exact), then take top 16 bits.
    out_fp32 = out_bf16.to(torch.float32).cpu().numpy().astype(np.float32).item()
    return float32_to_bf16_hex(np.float32(out_fp32))


# Test vector generation
def gen_vector_from_pool(rng, n, pool_np: np.ndarray) -> np.ndarray:
    idx = rng.integers(0, len(pool_np), size=n)
    return pool_np[idx].astype(np.float32)

def generate_test_case(
    rng,
    vec_len=32,
    num_lanes=16,
    case_type="basic",
    fp8_dtype=torch.float8_e4m3fn,
    ref="bf16_accum",
):
    pool = all_fp8_values(fp8_dtype)

    # sample as float32, quantize to fp8, store fp8 bits as hex
    act_vals = gen_vector_from_pool(rng, vec_len, pool)
    act_hex = [float_to_e4m3_hex_via_torch(float(v), fp8_dtype) for v in act_vals]

    wgt_hex = []
    for _ in range(num_lanes):
        row_vals = gen_vector_from_pool(rng, vec_len, pool)
        row_hex = [float_to_e4m3_hex_via_torch(float(v), fp8_dtype) for v in row_vals]
        wgt_hex.append(row_hex)

    sel = 0
    bias_hex = ["00"] * num_lanes
    psum_hex = ["0000"] * num_lanes

    if case_type == "bias":
        sel = 1
        bias_vals = gen_vector_from_pool(rng, num_lanes, pool)
        bias_hex = [float_to_e4m3_hex_via_torch(float(v), fp8_dtype) for v in bias_vals]
    elif case_type == "psum":
        sel = 2
        # generate psum in fp32, cast to bf16 (torch does rounding), export bits
        ps = (rng.uniform(-100, 100, size=num_lanes)).astype(np.float32)
        ps_bf16 = torch.from_numpy(ps).to(torch.bfloat16)
        ps_fp32 = ps_bf16.to(torch.float32).cpu().numpy().astype(np.float32)
        psum_hex = [float32_to_bf16_hex(v) for v in ps_fp32]

    expected_hex = []
    for r in range(num_lanes):
        addend = "00"
        if sel == 1:
            addend = bias_hex[r]
        elif sel == 2:
            addend = psum_hex[r]

        expected_hex.append(
            dot_expected_bf16_hex(
                act_hex,
                wgt_hex[r],
                sel=sel,
                addend_hex=addend,
                fp8_dtype=fp8_dtype,
                ref=ref,
            )
        )

    return {
        "act": act_hex,
        "weights": wgt_hex,
        "bias": bias_hex,
        "psum": psum_hex,
        "sel": sel,
        "expected": expected_hex,
    }

def write_case(f, case_id, case_type, case):
    f.write(f"# {case_id} {case_type}\n")
    f.write(f"sel {case['sel']}\n")
    f.write(f"act {' '.join(case['act'])}\n")
    for r, row in enumerate(case["weights"]):
        f.write(f"wgt {r} {' '.join(row)}\n")
    f.write(f"bias {' '.join(case['bias'])}\n")
    f.write(f"psum {' '.join(case['psum'])}\n")
    f.write(f"exp {' '.join(case['expected'])}\n\n")


# main

def main():
    parser = argparse.ArgumentParser(description="Generate MXU test vectors (PyTorch FP8/BF16)")
    parser.add_argument("--out", default="mxu_vectors.txt", help="Output file path")
    parser.add_argument("--num", type=int, default=30, help="Number of test cases")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--vec-len", type=int, default=32)
    parser.add_argument("--num-lanes", type=int, default=16)

    parser.add_argument(
        "--fp8",
        choices=["e4m3fn", "e4m3fnuz"],
        default="e4m3fn",
        help="Torch FP8 E4M3 variant used for quantization/bit patterns",
    )
    parser.add_argument(
        "--ref",
        choices=["bf16_accum", "fp32_accum"],
        default="bf16_accum",
        help="Reference math: bf16_accum (π0-like default) or fp32_accum baseline",
    )

    args = parser.parse_args()

    fp8_dtype = torch.float8_e4m3fn if args.fp8 == "e4m3fn" else torch.float8_e4m3fnuz

    rng = default_rng(args.seed)
    case_types = ["basic", "bias", "psum"]

    with open(args.out, "w") as f:
        for i in range(args.num):
            ct = case_types[i % len(case_types)]
            case = generate_test_case(
                rng,
                vec_len=args.vec_len,
                num_lanes=args.num_lanes,
                case_type=ct,
                fp8_dtype=fp8_dtype,
                ref=args.ref,
            )
            write_case(f, i, ct, case)

    print(f"Wrote {args.num} test vectors to {args.out} (fp8={args.fp8}, ref={args.ref}, out=bf16)")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_mxu_full_matmul.py: generate MXU test vectors for full tensor-register matrix multiply

Architecture:
  - Weight tile: vec_len × num_lanes (default 32×16), weight-stationary
  - Activation tensor register: num_rows × vec_len (default 64×32)
  - Per cycle:  [1×32] act × [32×16] wgt = [1×16] BF16 partial sums
  - Full op:    [num_rows × inner_dim] × [inner_dim × num_lanes] = [num_rows × num_lanes]
  - When inner_dim > vec_len, the multiply is tiled:
      num_tiles = inner_dim / vec_len
    Each tile's output feeds back as psum into the next tile.

Example:
  inner_dim=32  → 1 tile,   no psum accumulation
  inner_dim=64  → 2 tiles,  1 round of psum feedback
  inner_dim=256 → 8 tiles,  7 rounds of psum feedback (more rounding error)

Output format:
    # <id> full_matmul
    num_rows <R>
    num_tiles <T>
    wgt <tile> <lane> <hex> ...
    act <tile> <row>  <hex> ...
    exp <row> <hex> ...

Usage:
    # Single tile, no accumulation (simplest)
    python3 scripts/gen_mxu_full_matmul.py --out src/test/resources/mxu_full_matmul_vectors.txt --inner-dim 32

    # 2 tiles, tests psum feedback
    python3 scripts/gen_mxu_full_matmul.py --out src/test/resources/mxu_full_matmul_vectors.txt --inner-dim 64

    # 8 tiles, stress psum accumulation
    python3 scripts/gen_mxu_full_matmul.py --out src/test/resources/mxu_full_matmul_vectors.txt --inner-dim 256 --num 1

    # RUN THIS: Simulate large layer (16384-wide)
    python3 scripts/gen_mxu_full_matmul.py --out src/test/resources/mxu_full_matmul_vectors.txt --ref fp32_accum --inner-dim 16384 --num 1
"""

import argparse
import struct
import numpy as np
from numpy.random import default_rng
import torch


# ── BF16 hex helpers ──────────────────────────────────────────────────────────

def float32_to_bf16_hex(x: np.float32) -> str:
    fp32_bits = struct.unpack(">I", struct.pack(">f", float(x)))[0]
    bf16_bits = (fp32_bits >> 16) & 0xFFFF
    return f"{bf16_bits:04x}"

def bf16_hex_to_float32(h: str) -> np.float32:
    bf16_bits = int(h, 16) & 0xFFFF
    fp32_bits = (bf16_bits << 16) & 0xFFFFFFFF
    return np.float32(struct.unpack(">f", struct.pack(">I", fp32_bits))[0])


# ── FP8 E4M3 helpers ─────────────────────────────────────────────────────────

def float_to_e4m3_hex(x: float, fp8_dtype: torch.dtype) -> str:
    t = torch.tensor([x], dtype=torch.float32)
    q = t.to(fp8_dtype)
    return f"{q.view(torch.uint8).item():02x}"

def all_fp8_values(fp8_dtype: torch.dtype) -> np.ndarray:
    u = torch.arange(256, dtype=torch.uint8)
    exp = (u >> 3) & 0x0F
    man = u & 0x07
    is_subnormal = (exp == 0) & (man != 0)
    q = u.view(fp8_dtype)
    f = q.to(torch.float32)
    mask = (~torch.isnan(f)) & (~torch.isinf(f)) & (~is_subnormal)
    vals = f[mask].cpu().numpy().astype(np.float32)
    return np.unique(vals)


# ── Reference dot product ─────────────────────────────────────────────────────

def dot_bf16_hex(
    act_hex: list[str],
    wgt_hex: list[str],
    sel: int,
    addend_hex: str,
    fp8_dtype: torch.dtype,
    ref: str,
) -> str:
    act_u8 = torch.tensor([int(h, 16) & 0xFF for h in act_hex], dtype=torch.uint8)
    wgt_u8 = torch.tensor([int(h, 16) & 0xFF for h in wgt_hex], dtype=torch.uint8)
    act_fp8 = act_u8.view(fp8_dtype)
    wgt_fp8 = wgt_u8.view(fp8_dtype)

    if ref == "bf16_accum":
        act = act_fp8.to(torch.bfloat16)
        wgt = wgt_fp8.to(torch.bfloat16)
        acc = (act * wgt).sum(dtype=torch.bfloat16)

        if sel == 1:
            b_u8 = torch.tensor([int(addend_hex, 16) & 0xFF], dtype=torch.uint8)
            b = b_u8.view(fp8_dtype).to(torch.bfloat16).squeeze(0)
            acc = (acc + b).to(torch.bfloat16)
        elif sel == 2:
            psum_bf16 = torch.tensor(float(bf16_hex_to_float32(addend_hex)), dtype=torch.bfloat16)
            acc = (acc + psum_bf16).to(torch.bfloat16)

        out_bf16 = acc.to(torch.bfloat16)

    elif ref == "fp32_accum":
        act = act_fp8.to(torch.float32)
        wgt = wgt_fp8.to(torch.float32)
        acc = (act * wgt).sum(dtype=torch.float32)

        if sel == 1:
            b_u8 = torch.tensor([int(addend_hex, 16) & 0xFF], dtype=torch.uint8)
            b = b_u8.view(fp8_dtype).to(torch.float32).squeeze(0)
            acc = acc + b
        elif sel == 2:
            acc = acc + torch.tensor(float(bf16_hex_to_float32(addend_hex)), dtype=torch.float32)

        out_bf16 = acc.to(torch.bfloat16)
    else:
        raise ValueError(f"Unknown --ref {ref}")

    out_fp32 = out_bf16.to(torch.float32).cpu().numpy().astype(np.float32).item()
    return float32_to_bf16_hex(np.float32(out_fp32))


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_fp8_hex(rng, n: int, pool: np.ndarray, fp8_dtype: torch.dtype) -> list[str]:
    idx = rng.integers(0, len(pool), size=n)
    vals = pool[idx].astype(np.float32)
    return [float_to_e4m3_hex(float(v), fp8_dtype) for v in vals]


# ── Full matmul generation ────────────────────────────────────────────────────

def generate_full_matmul(
    rng,
    num_rows: int,
    vec_len: int,
    num_lanes: int,
    num_tiles: int,
    fp8_dtype: torch.dtype,
    ref: str,
):
    """
    Compute:  C[num_rows × num_lanes] = Σ_t  A_t[num_rows × vec_len] × W_t[vec_len × num_lanes]

    Tile 0:  sel=0 (no addend, just dot product)
    Tile 1+: sel=2 (add previous tile's output via psum feedback)
    """
    pool = all_fp8_values(fp8_dtype)

    tiles_wgt = []   # [tile][lane][col]
    tiles_act = []   # [tile][row][col]

    for _ in range(num_tiles):
        wgt = [sample_fp8_hex(rng, vec_len, pool, fp8_dtype) for _ in range(num_lanes)]
        act = [sample_fp8_hex(rng, vec_len, pool, fp8_dtype) for _ in range(num_rows)]
        tiles_wgt.append(wgt)
        tiles_act.append(act)

    # Compute expected: accumulate tile by tile
    psum_hex = [["0000"] * num_lanes for _ in range(num_rows)]

    for t in range(num_tiles):
        sel = 0 if t == 0 else 2
        for i in range(num_rows):
            row_out = []
            for r in range(num_lanes):
                addend = "00" if sel == 0 else psum_hex[i][r]
                row_out.append(
                    dot_bf16_hex(
                        tiles_act[t][i],
                        tiles_wgt[t][r],
                        sel=sel,
                        addend_hex=addend,
                        fp8_dtype=fp8_dtype,
                        ref=ref,
                    )
                )
            psum_hex[i] = row_out

    return tiles_wgt, tiles_act, psum_hex


# ── File I/O ──────────────────────────────────────────────────────────────────

def write_case(f, case_id, num_rows, num_tiles, tiles_wgt, tiles_act, expected):
    f.write(f"# {case_id} full_matmul\n")
    f.write(f"num_rows {num_rows}\n")
    f.write(f"num_tiles {num_tiles}\n")

    for t in range(num_tiles):
        for r, row in enumerate(tiles_wgt[t]):
            f.write(f"wgt {t} {r} {' '.join(row)}\n")
        for i, row in enumerate(tiles_act[t]):
            f.write(f"act {t} {i} {' '.join(row)}\n")

    for i, row in enumerate(expected):
        f.write(f"exp {i} {' '.join(row)}\n")
    f.write("\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate full-matmul MXU test vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single tile, no psum accumulation (simplest)
  python3 %(prog)s --inner-dim 32

  # 2 tiles, tests psum feedback
  python3 %(prog)s --inner-dim 64

  # 8 tiles, stress psum accumulation
  python3 %(prog)s --inner-dim 256

  # Simulate large FFN layer
  python3 %(prog)s --inner-dim 8192 --num 1
""",
    )
    ap.add_argument("--out",       default="mxu_full_matmul_vectors.txt")
    ap.add_argument("--num",       type=int, default=5,     help="Number of test cases")
    ap.add_argument("--seed",      type=int, default=12345)
    ap.add_argument("--num-rows",  type=int, default=64,    help="Rows in activation tensor register")
    ap.add_argument("--vec-len",   type=int, default=32,    help="Hardware dot-product width")
    ap.add_argument("--num-lanes", type=int, default=16,    help="Number of output lanes")
    ap.add_argument("--inner-dim", type=int, default=64,
                    help="Full inner (shared) dimension of the matmul. "
                         "Must be a multiple of vec-len. "
                         "Tiles = inner_dim / vec_len. "
                         "e.g. 32 → 1 tile, 256 → 8 tiles")
    ap.add_argument("--fp8",  choices=["e4m3fn", "e4m3fnuz"], default="e4m3fn")
    ap.add_argument("--ref",  choices=["bf16_accum", "fp32_accum"], default="bf16_accum")
    args = ap.parse_args()

    if args.inner_dim % args.vec_len != 0:
        ap.error(f"--inner-dim ({args.inner_dim}) must be a multiple of --vec-len ({args.vec_len})")

    num_tiles = args.inner_dim // args.vec_len
    fp8_dtype = torch.float8_e4m3fn if args.fp8 == "e4m3fn" else torch.float8_e4m3fnuz
    rng = default_rng(args.seed)

    print(f"Generating {args.num} test case(s):")
    print(f"  Matmul:  [{args.num_rows}×{args.inner_dim}] × [{args.inner_dim}×{args.num_lanes}]")
    print(f"  Tiling:  {num_tiles} tile(s) of [{args.num_rows}×{args.vec_len}] × [{args.vec_len}×{args.num_lanes}]")
    print(f"  Psum feedback passes: {max(0, num_tiles - 1)}")

    with open(args.out, "w") as f:
        for case_id in range(args.num):
            wgt, act, exp = generate_full_matmul(
                rng,
                num_rows=args.num_rows,
                vec_len=args.vec_len,
                num_lanes=args.num_lanes,
                num_tiles=num_tiles,
                fp8_dtype=fp8_dtype,
                ref=args.ref,
            )
            write_case(f, case_id, args.num_rows, num_tiles, wgt, act, exp)
            if (case_id + 1) % 10 == 0 or case_id == args.num - 1:
                print(f"  ... {case_id + 1}/{args.num} cases done")

    print(f"\nWrote {args.num} test cases to {args.out}")
    print(f"  fp8={args.fp8}  ref={args.ref}  output=bf16")


if __name__ == "__main__":
    main()

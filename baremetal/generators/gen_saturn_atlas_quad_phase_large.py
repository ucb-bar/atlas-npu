#!/usr/bin/env python3
"""Golden generator for tests/saturn_atlas_quad_phase_large.c.

Multi-tile Saturn<->Atlas pipeline with three Atlas phases interleaved with
two Saturn RVV BF16 interludes. Two of the three Atlas phases are JAL chains
across IMEM halves (lower-half kernel JALRs to upper-half kernel without
halting); the middle phase is a single MXU0 matmul.

Per tile (8 by default):

  Phase A1 (Atlas, JAL chain):
    lower (smolvla_matmul_mxu1_to_high):  Y1 = A1_A_fp8 @ A1_B_fp8 (MXU1/IPT)
    upper (smolvla_reduction_sum_at_2000): SUM1 = row_reduce_broadcast(A1_X)
  Interlude S1 (Saturn RVV BF16):
    Y1' = trunc_bf16(Y1 * SCALE1)  (per-row broadcast scale, write to fresh slot)
  Phase A2 (Atlas, single kernel):
    smolvla_matmul_mxu0:           Y2 = A2_A_fp8 @ A2_B_fp8 (MXU0/SA)
  Interlude S2 (Saturn RVV BF16):
    Y2' = trunc_bf16(Y2 * SCALE2)
  Phase A3 (Atlas, JAL chain):
    lower (smolvla_elementwise_add_to_high):    Z3 = A3_A_add + A3_B_add (BF16)
    upper (smolvla_matmul_mxu1_at_2000):        Y3 = A3_A_mm @ A3_B_mm (MXU1/IPT)

Per-tile counts: 3 ECALL boundaries, 2 in-IMEM JALR transitions, 2 Saturn
intervals.  N_TILES = 8 default → 24 ECALLs + 16 JALRs.

Both MXUs covered: MXU1 (IPT, single-round aligned tree) in A1 + A3,
MXU0 (SA, per-PE rounding) in A2.

DRAM layout per tile (host re-stages between phases):
  Lower-half kernel I/O (matches stock smolvla_*):
    A1 lower: [0x0000-0x07FF] in (FP8 A,B), [0x0800-0x0FFF] out (BF16 Y1)
    A3 lower: [0x0000-0x0FFF] in (BF16 A,B), [0x1000-0x17FF] out (BF16 Z3)
  Upper-half kernel I/O (offsets shifted by +0x2000 to avoid trampling lower):
    A1 upper: [0x2000-0x27FF] in (BF16 X), [0x2800-0x2FFF] out (BF16 SUM1)
    A3 upper: [0x2000-0x27FF] in (FP8 A,B), [0x2800-0x2FFF] out (BF16 Y3)
  A2 (single kernel, native offsets):
    [0x0000-0x07FF] in (FP8 A,B), [0x0800-0x0FFF] out (BF16 Y2)
  Saturn output slots (host writes from RVV BF16):
    [0x4000-0x47FF]   Y1' (post-S1)
    [0x4800-0x4FFF]   Y2' (post-S2)

All tiles use pseudo-random seeds intentionally. BF16 truncation edge cases
are covered by gen_saturn_atlas_tri_phase_large.py; this test focuses on
multi-MXU and repeated JAL-chain coverage.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (
    quantize_fp8,
    fp8_e4m3_bits_to_float,
    float_to_bf16_bits,
)
from software_models.mxu0_sa.systolic_array_rtl_linear import SARTLLinearFunction
from software_models.mxu1_ipt.fp_formats import OutputFmtSel
from software_models.mxu1_ipt.ipt_rtl_linear import IPTLinearRTLFunction
from software_models.vpu.bf16_utils import (
    bf16_upper_half_of_fp32_bits,
    fp32_bits_from_bf16,
)
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    run_binary_rows,
    run_row_reduce_tensor,
)


GOLDEN_N_TILES = 8
THROUGHPUT_BASE_SEED = 200

TILE_ROWS = ROWS_PER_REGISTER       # 32
TILE_COLS = 2 * BF16_PER_BEAT       # 32
TOTAL_ROWS = ROWS_PER_TENSOR        # 64 (bank0 + bank1)
LANES = BF16_PER_BEAT               # 16


# ── Helpers ────────────────────────────────────────────────────────────────


def fp32_to_e4m3_byte(x: float, _lut: dict[int, float] = {}) -> int:
    """Encode an FP32 value (already FP8-quantized) to its canonical E4M3 byte."""
    if not _lut:
        for b in range(256):
            _lut[b] = fp8_e4m3_bits_to_float(b)
    arr = np.array([[x]], dtype=np.float32)
    qf = float(quantize_fp8(arr).astype(np.float32)[0, 0])
    for b, v in _lut.items():
        if v == qf or (np.isnan(v) and np.isnan(qf)):
            return b
    raise ValueError(f"no E4M3 byte encodes {qf}")


def fp32_matrix_to_fp8_bytes(mat: np.ndarray) -> np.ndarray:
    rows, cols = mat.shape
    out = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = fp32_to_e4m3_byte(float(mat[r, c]))
    return out


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> list[list[int]]:
    if tile.shape != (TILE_ROWS, TILE_COLS):
        raise ValueError(f"expected {TILE_ROWS}x{TILE_COLS} tile, got {tuple(tile.shape)}")
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bank0 = tile[:, :LANES].contiguous().view(torch.int16)
    bank1 = tile[:, LANES:].contiguous().view(torch.int16)
    rows: list[list[int]] = []
    for half in (bank0, bank1):
        for i in range(TILE_ROWS):
            rows.append([int(v) & 0xFFFF for v in half[i].tolist()])
    return rows


def fp32_to_bf16_bank_rows(mat: np.ndarray) -> list[list[int]]:
    """Convert a (32, 32) FP32 matrix to bank-split BF16 rows."""
    bits = np.vectorize(float_to_bf16_bits)(mat).astype(np.uint16)
    rows: list[list[int]] = []
    for r in range(TILE_ROWS):
        rows.append([int(v) for v in bits[r, :LANES]])
    for r in range(TILE_ROWS):
        rows.append([int(v) for v in bits[r, LANES:TILE_COLS]])
    return rows


def saturn_interlude(rows: list[list[int]], scale_row_bf16: list[int]) -> list[list[int]]:
    """Saturn RVV BF16 row-broadcast scale (matches tri_phase_large semantics)."""
    if len(rows) != TOTAL_ROWS:
        raise ValueError(f"expected {TOTAL_ROWS} rows, got {len(rows)}")
    if len(scale_row_bf16) != TILE_ROWS:
        raise ValueError(f"scale must have {TILE_ROWS} entries, got {len(scale_row_bf16)}")
    out_rows: list[list[int]] = []
    for row_idx, row in enumerate(rows):
        tile_row = row_idx % TILE_ROWS
        s_bits = scale_row_bf16[tile_row] & 0xFFFF
        s_fp32 = fp32_bits_from_bf16(s_bits)
        out_row: list[int] = []
        for lane_bits in row:
            a_fp32 = fp32_bits_from_bf16(lane_bits & 0xFFFF)
            a = struct.unpack("<f", struct.pack("<I", a_fp32))[0]
            s = struct.unpack("<f", struct.pack("<I", s_fp32))[0]
            p = a * s
            try:
                p_bits = struct.unpack("<I", struct.pack("<f", p))[0]
            except OverflowError:
                p_bits = 0xFF800000 if p < 0 else 0x7F800000
            out_row.append(bf16_upper_half_of_fp32_bits(p_bits))
        out_rows.append(out_row)
    return out_rows


def flatten_rows(rows: list[list[int]]) -> list[int]:
    flat: list[int] = []
    for r in rows:
        flat.extend(r)
    return flat


# ── MXU goldens ────────────────────────────────────────────────────────────


def mxu1_matmul(a_fp8_q: np.ndarray, b_fp8_q: np.ndarray) -> np.ndarray:
    """Run MXU1 (IPT) matmul software model. Inputs are FP8-quantized FP32
    arrays; output is FP32 holding BF16-rounded values."""
    ipt = IPTLinearRTLFunction(
        vec_len=TILE_ROWS, num_lanes=TILE_ROWS, pipeline_depth=1,
        out_fmt_sel=OutputFmtSel.OutBF16,
    )
    a_t = torch.from_numpy(a_fp8_q)
    w_t = torch.from_numpy(b_fp8_q.T.copy())
    return ipt(a_t, w_t, scale_exp=0).numpy().astype(np.float32)


def mxu0_matmul(a_fp8_q: np.ndarray, b_fp8_q: np.ndarray) -> np.ndarray:
    sa = SARTLLinearFunction(rows=TILE_ROWS, cols=TILE_ROWS, out_fmt_sel=OutputFmtSel.OutBF16)
    a_t = torch.from_numpy(a_fp8_q)
    w_t = torch.from_numpy(b_fp8_q.T.copy())
    return sa(a_t, w_t, scale_exp=0).numpy().astype(np.float32)


# ── Header emission ────────────────────────────────────────────────────────


def emit_u16_array_3d(name: str, shape: tuple[int, int, int], vals: list[int]) -> str:
    tiles, rows, cols = shape
    assert len(vals) == tiles * rows * cols, (name, shape, len(vals))
    tile_blocks = []
    for t in range(tiles):
        row_lines = []
        base = t * rows * cols
        for r in range(rows):
            row_vals = vals[base + r * cols : base + (r + 1) * cols]
            row_body = ", ".join(f"0x{v & 0xFFFF:04x}" for v in row_vals)
            row_lines.append(f"        {{ {row_body} }}")
        tile_blocks.append("    {\n" + ",\n".join(row_lines) + "\n    }")
    body = ",\n".join(tile_blocks)
    return f"static const uint16_t {name}[{tiles}][{rows}][{cols}] = {{\n{body}\n}};\n"


def emit_u16_array_2d(name: str, shape: tuple[int, int], vals: list[int]) -> str:
    rows, cols = shape
    assert len(vals) == rows * cols, (name, shape, len(vals))
    lines = []
    for r in range(rows):
        row_vals = vals[r * cols : (r + 1) * cols]
        row_body = ", ".join(f"0x{v & 0xFFFF:04x}" for v in row_vals)
        lines.append(f"    {{ {row_body} }}")
    body = ",\n".join(lines)
    return f"static const uint16_t {name}[{rows}][{cols}] = {{\n{body}\n}};\n"


def emit_u8_array_3d(name: str, shape: tuple[int, int, int], vals: list[int]) -> str:
    tiles, rows, cols = shape
    assert len(vals) == tiles * rows * cols, (name, shape, len(vals))
    tile_blocks = []
    for t in range(tiles):
        row_lines = []
        base = t * rows * cols
        for r in range(rows):
            row_vals = vals[base + r * cols : base + (r + 1) * cols]
            row_body = ", ".join(f"0x{v & 0xFF:02x}" for v in row_vals)
            row_lines.append(f"        {{ {row_body} }}")
        tile_blocks.append("    {\n" + ",\n".join(row_lines) + "\n    }")
    body = ",\n".join(tile_blocks)
    return f"static const uint8_t {name}[{tiles}][{rows}][{cols}] = {{\n{body}\n}};\n"


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-header", default=None,
                    help="Write C header to this path (default: stdout)")
    ap.add_argument("--n-tiles", type=int, default=GOLDEN_N_TILES,
                    help=f"Override golden tile count (default {GOLDEN_N_TILES}).")
    args = ap.parse_args()
    n_tiles = args.n_tiles

    # Per-tile flat collectors
    in_a1_a_fp8: list[int] = []   # FP8 bytes (A1 lower input A)
    in_a1_b_fp8: list[int] = []   # FP8 bytes (A1 lower input B, pre-transposed)
    in_a1_x_bf16: list[int] = []  # BF16 (A1 upper input X), bank-split

    in_a2_a_fp8: list[int] = []
    in_a2_b_fp8: list[int] = []

    in_a3_a_add_bf16: list[int] = []   # bank-split
    in_a3_b_add_bf16: list[int] = []
    in_a3_a_mm_fp8: list[int] = []
    in_a3_b_mm_fp8: list[int] = []

    scale1_bits: list[int] = []   # 32 BF16 per tile
    scale2_bits: list[int] = []

    expect_y1: list[int] = []
    expect_sum1: list[int] = []
    expect_y1p: list[int] = []
    expect_y2: list[int] = []
    expect_y2p: list[int] = []
    expect_z3: list[int] = []
    expect_y3: list[int] = []

    for tile in range(n_tiles):
        torch.manual_seed(THROUGHPUT_BASE_SEED + tile)
        np_seed = THROUGHPUT_BASE_SEED + tile

        # ── Inputs ─────────────────────────────────────────────────────────
        rng = np.random.RandomState(np_seed)
        a1_a_fp32 = rng.uniform(-1.25, 1.25, size=(TILE_ROWS, TILE_ROWS)).astype(np.float32)
        a1_b_fp32 = rng.uniform(-1.25, 1.25, size=(TILE_ROWS, TILE_ROWS)).astype(np.float32)
        a1_a_q = quantize_fp8(a1_a_fp32).astype(np.float32)
        a1_b_q = quantize_fp8(a1_b_fp32).astype(np.float32)
        # Match gen_smolvla_matmul_mxu{0,1}.py: DRAM holds raw B. The Atlas
        # kernel VTRPOSE.XLUs B before push; mxu*_matmul() compensates for
        # F.linear by passing w = B.T to the software model.
        a1_b_preload = a1_b_q.copy()

        a1_x_tile = torch.randn(TILE_ROWS, TILE_COLS, dtype=torch.bfloat16)
        scale1_tile = torch.randn(TILE_ROWS, dtype=torch.bfloat16)

        a2_a_fp32 = rng.uniform(-1.25, 1.25, size=(TILE_ROWS, TILE_ROWS)).astype(np.float32)
        a2_b_fp32 = rng.uniform(-1.25, 1.25, size=(TILE_ROWS, TILE_ROWS)).astype(np.float32)
        a2_a_q = quantize_fp8(a2_a_fp32).astype(np.float32)
        a2_b_q = quantize_fp8(a2_b_fp32).astype(np.float32)
        a2_b_preload = a2_b_q.copy()

        scale2_tile = torch.randn(TILE_ROWS, dtype=torch.bfloat16)

        a3_a_add_tile = torch.randn(TILE_ROWS, TILE_COLS, dtype=torch.bfloat16)
        a3_b_add_tile = torch.randn(TILE_ROWS, TILE_COLS, dtype=torch.bfloat16)

        a3_a_mm_fp32 = rng.uniform(-1.25, 1.25, size=(TILE_ROWS, TILE_ROWS)).astype(np.float32)
        a3_b_mm_fp32 = rng.uniform(-1.25, 1.25, size=(TILE_ROWS, TILE_ROWS)).astype(np.float32)
        a3_a_mm_q = quantize_fp8(a3_a_mm_fp32).astype(np.float32)
        a3_b_mm_q = quantize_fp8(a3_b_mm_fp32).astype(np.float32)
        a3_b_mm_preload = a3_b_mm_q.copy()

        # ── Goldens ────────────────────────────────────────────────────────
        # A1 lower: MXU1 matmul Y1 = A1_A @ A1_B (BF16 from IPT model)
        y1_fp32 = mxu1_matmul(a1_a_q, a1_b_preload)
        y1_rows = fp32_to_bf16_bank_rows(y1_fp32)

        # A1 upper: row-sum broadcast over A1_X (BF16)
        x_rows = bf16_tile_to_bank_rows(a1_x_tile)
        sum1_rows = run_row_reduce_tensor("rsum", x_rows)

        # S1: Saturn rescales Y1 by scale1 → Y1' (per-row broadcast)
        scale1_bits_tile = [int(v) & 0xFFFF for v in scale1_tile.view(torch.int16).tolist()]
        y1p_rows = saturn_interlude(y1_rows, scale1_bits_tile)

        # A2: MXU0 matmul Y2 = A2_A @ A2_B (BF16 from SA model)
        y2_fp32 = mxu0_matmul(a2_a_q, a2_b_preload)
        y2_rows = fp32_to_bf16_bank_rows(y2_fp32)

        # S2: Saturn rescales Y2 by scale2 → Y2'
        scale2_bits_tile = [int(v) & 0xFFFF for v in scale2_tile.view(torch.int16).tolist()]
        y2p_rows = saturn_interlude(y2_rows, scale2_bits_tile)

        # A3 lower: BF16 elementwise add
        a3_a_add_rows = bf16_tile_to_bank_rows(a3_a_add_tile)
        a3_b_add_rows = bf16_tile_to_bank_rows(a3_b_add_tile)
        z3_rows = run_binary_rows("add", a3_a_add_rows, a3_b_add_rows)

        # A3 upper: MXU1 matmul Y3 = A3_A_mm @ A3_B_mm (BF16 from IPT model)
        y3_fp32 = mxu1_matmul(a3_a_mm_q, a3_b_mm_preload)
        y3_rows = fp32_to_bf16_bank_rows(y3_fp32)

        # ── FP8 byte conversion for kernel inputs ──────────────────────────
        a1_a_bytes = fp32_matrix_to_fp8_bytes(a1_a_q).reshape(-1).tolist()
        a1_b_bytes = fp32_matrix_to_fp8_bytes(a1_b_preload).reshape(-1).tolist()
        a2_a_bytes = fp32_matrix_to_fp8_bytes(a2_a_q).reshape(-1).tolist()
        a2_b_bytes = fp32_matrix_to_fp8_bytes(a2_b_preload).reshape(-1).tolist()
        a3_a_mm_bytes = fp32_matrix_to_fp8_bytes(a3_a_mm_q).reshape(-1).tolist()
        a3_b_mm_bytes = fp32_matrix_to_fp8_bytes(a3_b_mm_preload).reshape(-1).tolist()

        # ── Append to per-tile collectors ──────────────────────────────────
        in_a1_a_fp8.extend(a1_a_bytes)
        in_a1_b_fp8.extend(a1_b_bytes)
        in_a1_x_bf16.extend(flatten_rows(x_rows))

        in_a2_a_fp8.extend(a2_a_bytes)
        in_a2_b_fp8.extend(a2_b_bytes)

        in_a3_a_add_bf16.extend(flatten_rows(a3_a_add_rows))
        in_a3_b_add_bf16.extend(flatten_rows(a3_b_add_rows))
        in_a3_a_mm_fp8.extend(a3_a_mm_bytes)
        in_a3_b_mm_fp8.extend(a3_b_mm_bytes)

        scale1_bits.extend(scale1_bits_tile)
        scale2_bits.extend(scale2_bits_tile)

        expect_y1.extend(flatten_rows(y1_rows))
        expect_sum1.extend(flatten_rows(sum1_rows))
        expect_y1p.extend(flatten_rows(y1p_rows))
        expect_y2.extend(flatten_rows(y2_rows))
        expect_y2p.extend(flatten_rows(y2p_rows))
        expect_z3.extend(flatten_rows(z3_rows))
        expect_y3.extend(flatten_rows(y3_rows))

        print(f"  tile {tile:2d} (seed={np_seed}) done", file=sys.stderr)

    header: list[str] = []
    header.append("/* AUTO-GENERATED by generators/sp26-atlas-acc/baremetal/generators/")
    header.append(" *   gen_saturn_atlas_quad_phase_large.py")
    header.append(f" * GOLDEN_N_TILES = {n_tiles}")
    header.append(f" * THROUGHPUT_BASE_SEED = {THROUGHPUT_BASE_SEED}")
    header.append(" * Pipeline: Atlas(A1=MXU1+reduce JAL chain) -> Saturn S1 ->")
    header.append(" *           Atlas(A2=MXU0 single) -> Saturn S2 ->")
    header.append(" *           Atlas(A3=add+MXU1 JAL chain)")
    header.append(" * Do not edit by hand. */")
    header.append("#ifndef SATURN_ATLAS_QUAD_PHASE_LARGE_GOLDEN_H")
    header.append("#define SATURN_ATLAS_QUAD_PHASE_LARGE_GOLDEN_H")
    header.append("")
    header.append("#include <stdint.h>")
    header.append("")
    header.append(f"#define SAC_QUAD_GOLDEN_N_TILES {n_tiles}")
    header.append(f"#define SAC_QUAD_TILE_ROWS      {TILE_ROWS}")
    header.append(f"#define SAC_QUAD_TILE_COLS      {TILE_COLS}")
    header.append(f"#define SAC_QUAD_TOTAL_ROWS     {TOTAL_ROWS}")
    header.append(f"#define SAC_QUAD_LANES          {LANES}")
    header.append("")

    # Inputs
    header.append(emit_u8_array_3d ("SAC_QUAD_A1_A_FP8",    (n_tiles, TILE_ROWS, TILE_ROWS), in_a1_a_fp8))
    header.append(emit_u8_array_3d ("SAC_QUAD_A1_B_FP8",    (n_tiles, TILE_ROWS, TILE_ROWS), in_a1_b_fp8))
    header.append(emit_u16_array_3d("SAC_QUAD_A1_X_BF16",   (n_tiles, TOTAL_ROWS, LANES),    in_a1_x_bf16))
    header.append(emit_u8_array_3d ("SAC_QUAD_A2_A_FP8",    (n_tiles, TILE_ROWS, TILE_ROWS), in_a2_a_fp8))
    header.append(emit_u8_array_3d ("SAC_QUAD_A2_B_FP8",    (n_tiles, TILE_ROWS, TILE_ROWS), in_a2_b_fp8))
    header.append(emit_u16_array_3d("SAC_QUAD_A3_A_ADD_BF16", (n_tiles, TOTAL_ROWS, LANES),  in_a3_a_add_bf16))
    header.append(emit_u16_array_3d("SAC_QUAD_A3_B_ADD_BF16", (n_tiles, TOTAL_ROWS, LANES),  in_a3_b_add_bf16))
    header.append(emit_u8_array_3d ("SAC_QUAD_A3_A_MM_FP8",   (n_tiles, TILE_ROWS, TILE_ROWS), in_a3_a_mm_fp8))
    header.append(emit_u8_array_3d ("SAC_QUAD_A3_B_MM_FP8",   (n_tiles, TILE_ROWS, TILE_ROWS), in_a3_b_mm_fp8))

    # Saturn scales
    header.append(emit_u16_array_2d("SAC_QUAD_SCALE1",       (n_tiles, TILE_ROWS), scale1_bits))
    header.append(emit_u16_array_2d("SAC_QUAD_SCALE2",       (n_tiles, TILE_ROWS), scale2_bits))

    # Expected outputs
    header.append(emit_u16_array_3d("SAC_QUAD_EXPECT_Y1",    (n_tiles, TOTAL_ROWS, LANES), expect_y1))
    header.append(emit_u16_array_3d("SAC_QUAD_EXPECT_SUM1",  (n_tiles, TOTAL_ROWS, LANES), expect_sum1))
    header.append(emit_u16_array_3d("SAC_QUAD_EXPECT_Y1P",   (n_tiles, TOTAL_ROWS, LANES), expect_y1p))
    header.append(emit_u16_array_3d("SAC_QUAD_EXPECT_Y2",    (n_tiles, TOTAL_ROWS, LANES), expect_y2))
    header.append(emit_u16_array_3d("SAC_QUAD_EXPECT_Y2P",   (n_tiles, TOTAL_ROWS, LANES), expect_y2p))
    header.append(emit_u16_array_3d("SAC_QUAD_EXPECT_Z3",    (n_tiles, TOTAL_ROWS, LANES), expect_z3))
    header.append(emit_u16_array_3d("SAC_QUAD_EXPECT_Y3",    (n_tiles, TOTAL_ROWS, LANES), expect_y3))

    header.append("#endif")
    header.append("")

    text = "\n".join(header)
    if args.out_header:
        with open(args.out_header, "w") as f:
            f.write(text)
        print(f"Wrote {args.out_header}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

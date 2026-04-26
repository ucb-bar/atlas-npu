#!/usr/bin/env python3
"""Golden generator for tests/saturn_atlas_tri_phase_large.c.

Multi-tile variant of gen_saturn_atlas_tri_phase.py. Emits
GOLDEN_N_TILES = 16 tiles exercising the full four-stage pipeline:

  Phase 1 (Atlas):      Y   = A * B                    (smolvla_elementwise_mul)
  Interlude 1 (Saturn): Y'  = trunc_bf16(Y * scale)    (RVV, row-broadcast)
  Phase 2 (Atlas):      S   = row_reduce(Y')           (sac_reduction_sum_to_high
                                                         — JAL to upper IMEM, NO halt)
  Phase 3 (Atlas):      Z   = Y' + S                   (smolvla_elementwise_add)
  Interlude 2 (Saturn): Z'' = trunc_bf16(Z * post_scale) (RVV, row-broadcast)

Tile layout:
  Tile 0: subnormal-heavy BF16 interlude-1 scale row (a/b from seed 98).
          Stresses BF16 subnormal handling in Saturn's u16->f32 path.
  Tile 1: planted FP32-subnormal product at interlude 1, lane (0,0):
            a[0,0]=0x3E66, b[0,0]=0x3F2D, scale[0]=0x00C0.
          Forces Y*scale into the FP32 subnormal window; FTZ-on vs
          FTZ-off hardware diverges at upper-16 slice 0x001D vs 0x0000.
          Remaining lanes from seed 99.
  Tile 2..15: pure torch.randn with seed = 98 + i. Tile 2 uses seed
          100, bit-identical to the single-tile tri_phase's Y/Y'/S
          anchors (only phases 3 + interlude 2 are new vs compose).

post_scale is randn per tile (not edge-case planted); one edge-case
axis per tile is enough.

Semantics match gen_saturn_atlas_tri_phase.py; see its docstring.
Emits a C header with u16 arrays consumed by the test driver.
"""

from __future__ import annotations

import argparse
import struct
import sys

import torch

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


GOLDEN_N_TILES = 16
THROUGHPUT_BASE_SEED = 98   # tile 2 -> seed 100 (single-tile anchor)

TILE_ROWS = ROWS_PER_REGISTER       # 32
TILE_COLS = 2 * BF16_PER_BEAT       # 32
TOTAL_ROWS = ROWS_PER_TENSOR        # 64 (bank0 + bank1)
LANES = BF16_PER_BEAT               # 16


# ── Bit-level helpers ──────────────────────────────────────────────────────

def _u16_to_signed(bits: int) -> int:
    bits &= 0xFFFF
    return bits if bits < 0x8000 else bits - 0x10000


def _u16_list_to_bf16_tensor(vals: list[int]) -> torch.Tensor:
    signed = [_u16_to_signed(v) for v in vals]
    return torch.tensor(signed, dtype=torch.int16).view(torch.bfloat16).clone()


def _patch_bf16_bits_2d(tile: torch.Tensor, idx: tuple[int, int], bits: int) -> torch.Tensor:
    out = tile.clone().contiguous()
    out.view(torch.int16)[idx[0], idx[1]] = _u16_to_signed(bits)
    return out


def _patch_bf16_bits_1d(vec: torch.Tensor, i: int, bits: int) -> torch.Tensor:
    out = vec.clone().contiguous()
    out.view(torch.int16)[i] = _u16_to_signed(bits)
    return out


# ── Edge-case tile constructors (shared semantics with compose_large) ──────

def make_subnormal_scale_row() -> list[int]:
    row = [
        0x0001,  # +2^-133 (smallest positive BF16 subnormal)
        0x0040,  # exactly 2^-127 (mid subnormal)
        0x007F,  # largest positive BF16 subnormal
        0x8001, 0x8040, 0x807F,   # negative counterparts
        0x0000, 0x8000,           # +0, -0
    ]
    while len(row) < TILE_ROWS:
        row.append(0x3B2D if len(row) % 2 == 0 else 0xBB4B)
    assert len(row) == TILE_ROWS
    return row


def plant_subnormal_product_tile(
    a_tile: torch.Tensor, b_tile: torch.Tensor, s_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a_tile = _patch_bf16_bits_2d(a_tile, (0, 0), 0x3E66)
    b_tile = _patch_bf16_bits_2d(b_tile, (0, 0), 0x3F2D)
    s_scale = _patch_bf16_bits_1d(s_scale, 0, 0x00C0)
    return a_tile, b_tile, s_scale


# ── Pipeline helpers (identical to single-tile tri_phase) ──────────────────

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


def saturn_interlude(rows: list[list[int]], scale_row_bf16: list[int]) -> list[list[int]]:
    """Row-broadcast BF16 scale using Saturn's upper-16 truncation policy."""
    if len(rows) != TOTAL_ROWS:
        raise ValueError(f"expected {TOTAL_ROWS} rows, got {len(rows)}")
    if len(scale_row_bf16) != TILE_ROWS:
        raise ValueError(f"scale must have {TILE_ROWS} entries, got {len(scale_row_bf16)}")

    out_rows: list[list[int]] = []
    for row_idx, row in enumerate(rows):
        tile_row = row_idx % TILE_ROWS
        s_bits = scale_row_bf16[tile_row] & 0xFFFF
        s_fp32 = fp32_bits_from_bf16(s_bits)
        out_row = []
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


# ── Header emission ────────────────────────────────────────────────────────

def emit_u16_array_2d(name: str, shape: tuple[int, int], vals: list[int]) -> str:
    rows, cols = shape
    assert len(vals) == rows * cols, (name, shape, len(vals))
    lines = []
    for r in range(rows):
        row_vals = vals[r * cols:(r + 1) * cols]
        row_body = ", ".join(f"0x{v & 0xFFFF:04x}" for v in row_vals)
        lines.append(f"    {{ {row_body} }}")
    body = ",\n".join(lines)
    return f"static const uint16_t {name}[{rows}][{cols}] = {{\n{body}\n}};\n"


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


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-header", default=None,
                    help="Write C header to this path (default: stdout)")
    ap.add_argument("--n-tiles", type=int, default=GOLDEN_N_TILES,
                    help=f"Override golden tile count (default {GOLDEN_N_TILES}). "
                         "For CI use the default; smaller values are dev-only.")
    args = ap.parse_args()

    n_tiles = args.n_tiles

    a_all:          list[int] = []
    b_all:          list[int] = []
    scale_all:      list[int] = []
    post_scale_all: list[int] = []
    z2_all:         list[int] = []

    for i in range(n_tiles):
        torch.manual_seed(THROUGHPUT_BASE_SEED + i)
        a_tile     = torch.randn(TILE_ROWS, TILE_COLS, dtype=torch.bfloat16)
        b_tile     = torch.randn(TILE_ROWS, TILE_COLS, dtype=torch.bfloat16)
        s_scale    = torch.randn(TILE_ROWS, dtype=torch.bfloat16)
        post_scale = torch.randn(TILE_ROWS, dtype=torch.bfloat16)

        if i == 0:
            s_scale = _u16_list_to_bf16_tensor(make_subnormal_scale_row())
        elif i == 1:
            a_tile, b_tile, s_scale = plant_subnormal_product_tile(a_tile, b_tile, s_scale)

        a_rows = bf16_tile_to_bank_rows(a_tile)
        b_rows = bf16_tile_to_bank_rows(b_tile)
        scale_bits      = [int(v) & 0xFFFF for v in s_scale.view(torch.int16).tolist()]
        post_scale_bits = [int(v) & 0xFFFF for v in post_scale.view(torch.int16).tolist()]

        y_rows  = run_binary_rows("mul", a_rows, b_rows)
        yp_rows = saturn_interlude(y_rows, scale_bits)
        s_rows  = run_row_reduce_tensor("rsum", yp_rows)
        z_rows  = run_binary_rows("add", yp_rows, s_rows)
        z2_rows = saturn_interlude(z_rows, post_scale_bits)

        a_all.extend(flatten_rows(a_rows))
        b_all.extend(flatten_rows(b_rows))
        scale_all.extend(scale_bits)
        post_scale_all.extend(post_scale_bits)
        z2_all.extend(flatten_rows(z2_rows))

        print(f"  tile {i:2d} (seed={THROUGHPUT_BASE_SEED + i}) done",
              file=sys.stderr)

    header = []
    header.append("/* AUTO-GENERATED by generators/sp26-atlas-acc/baremetal/generators/")
    header.append(" *   gen_saturn_atlas_tri_phase_large.py")
    header.append(f" * GOLDEN_N_TILES = {n_tiles}")
    header.append(f" * THROUGHPUT_BASE_SEED = {THROUGHPUT_BASE_SEED}")
    header.append(" *   - tile 0: hand-built subnormal-heavy interlude-1 scale row (a,b from seed 98)")
    header.append(" *   - tile 1: planted FP32-subnormal product (a,b,scale patched; rest from seed 99)")
    header.append(" *   - tile i>=2: pure randn seeded at BASE + i (tile 2 -> seed 100 anchor)")
    header.append(" * Do not edit by hand. */")
    header.append("#ifndef SATURN_ATLAS_TRI_PHASE_LARGE_GOLDEN_H")
    header.append("#define SATURN_ATLAS_TRI_PHASE_LARGE_GOLDEN_H")
    header.append("")
    header.append("#include <stdint.h>")
    header.append("")
    header.append(f"#define SAC_TRI_L_GOLDEN_N_TILES {n_tiles}")
    header.append(f"#define SAC_TRI_L_TILE_ROWS      {TILE_ROWS}")
    header.append(f"#define SAC_TRI_L_TILE_COLS      {TILE_COLS}")
    header.append(f"#define SAC_TRI_L_TOTAL_ROWS     {TOTAL_ROWS}")
    header.append(f"#define SAC_TRI_L_LANES          {LANES}")
    header.append("")
    header.append(emit_u16_array_3d("SAC_TRI_L_A",           (n_tiles, TOTAL_ROWS, LANES), a_all))
    header.append(emit_u16_array_3d("SAC_TRI_L_B",           (n_tiles, TOTAL_ROWS, LANES), b_all))
    header.append(emit_u16_array_2d("SAC_TRI_L_SCALE",       (n_tiles, TILE_ROWS),         scale_all))
    header.append(emit_u16_array_2d("SAC_TRI_L_POST_SCALE",  (n_tiles, TILE_ROWS),         post_scale_all))
    header.append(emit_u16_array_3d("SAC_TRI_L_EXPECT_Z2",   (n_tiles, TOTAL_ROWS, LANES), z2_all))
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

#!/usr/bin/env python3
"""Golden generator for tests/saturn_atlas_tri_phase{,_native_bf16}.c.

Four-stage pipeline combining ECALL handshake (compose baseline), in-IMEM
JALR dispatch across IMEM halves (Stage 2 pattern), and Saturn cooperative
RVV compute:

  Phase 1 (Atlas):      Y   = A * B                  (smolvla_elementwise_mul)
  Interlude 1 (Saturn): Y'  = bf16(Y * scale)        (RVV row-broadcast)
  Phase 2 (Atlas):      S   = row_reduce(Y')         (sac_reduction_sum_to_high)
    -- JAL to IMEM upper half, no halt --
  Phase 3 (Atlas):      Z   = Y' + S                 (smolvla_elementwise_add)
  Interlude 2 (Saturn): Z'' = bf16(Z * post_scale)   (RVV row-broadcast)

Both interludes share the same --interlude-mode selector:

  truncate_upper16 (default; matches the legacy bit-hack RVV path):
      bf16_out = upper-16 of (bf16_to_f32(a) * bf16_to_f32(s))
                  -- bf16_upper_half_of_fp32_bits, no RNE pass

  native_bf16_rne (matches Saturn vfmul.vv altfmt=1 + vfncvtbf16.f.f.w):
      bf16_out = bf16_mul(a, s)  -- f32_to_bf16_bits_rne after FP32 mul

The default truncate mode preserves the existing saturn_atlas_tri_phase
oracle and serves as a Saturn FP32-subnormal regression anchor;
native_bf16_rne emits the sibling golden for
saturn_atlas_tri_phase_native_bf16. Atlas-side phases (mul, row-reduce,
add) route through VectorEngineModel via vpu_gen_utils in both modes.

Seed anchor: defaults to seed=100 (matches compose single-tile and
compose-large tile 2) so phases 1+2 are bit-identical to compose's
SAC_EXPECT_S through step 2 of the pipeline.

Emits a C header with u16 arrays consumed by the test driver.
"""

from __future__ import annotations

import argparse
import struct
import sys

import torch

from software_models.vpu.bf16_utils import (
    bf16_mul,
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


SEED = 100
TILE_ROWS = ROWS_PER_REGISTER       # 32
TILE_COLS = 2 * BF16_PER_BEAT       # 32
TOTAL_ROWS = ROWS_PER_TENSOR        # 64 (bank0 + bank1)
LANES = BF16_PER_BEAT               # 16


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


INTERLUDE_MODES = ("truncate_upper16", "native_bf16_rne")


def saturn_interlude(
    rows: list[list[int]],
    scale_row_bf16: list[int],
    mode: str = "truncate_upper16",
) -> list[list[int]]:
    """Row-broadcast BF16 scale.

    Two semantics, selected by `mode`:

      truncate_upper16 (default; matches the bit-hack RVV interlude):
          f32_a = bf16<<16; f32_s = scale<<16
          f32_p = f32_a * f32_s
          bf16_out = f32_p >> 16  (raw upper-16 slice, no second RNE pass)

      native_bf16_rne (matches Saturn `vfmul.vv` altfmt=1 + `vfncvtbf16`):
          bf16_out = bf16_mul(a_bits, s_bits)
                   = f32_to_bf16_bits_rne(bf16_to_f32(a) * bf16_to_f32(s))

    `scale_row_bf16` has length TILE_ROWS (32). Bank 0 rows [0..31] and
    bank 1 rows [0..31] each get scale[row_in_tile].
    """
    if mode not in INTERLUDE_MODES:
        raise ValueError(f"unknown interlude mode {mode!r}; expected one of {INTERLUDE_MODES}")
    if len(rows) != TOTAL_ROWS:
        raise ValueError(f"expected {TOTAL_ROWS} rows, got {len(rows)}")
    if len(scale_row_bf16) != TILE_ROWS:
        raise ValueError(f"scale must have {TILE_ROWS} entries, got {len(scale_row_bf16)}")

    out_rows: list[list[int]] = []
    for row_idx, row in enumerate(rows):
        tile_row = row_idx % TILE_ROWS  # bank0 rows first, then bank1
        s_bits = scale_row_bf16[tile_row] & 0xFFFF
        out_row = []
        if mode == "truncate_upper16":
            s_fp32 = fp32_bits_from_bf16(s_bits)
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
        else:  # native_bf16_rne
            for lane_bits in row:
                out_row.append(bf16_mul(lane_bits & 0xFFFF, s_bits))
        out_rows.append(out_row)
    return out_rows


def flatten_rows(rows: list[list[int]]) -> list[int]:
    flat: list[int] = []
    for r in rows:
        flat.extend(r)
    return flat


def emit_u16_array(name: str, shape: tuple[int, ...], flat_vals: list[int]) -> str:
    assert len(flat_vals) == _prod(shape), (name, shape, len(flat_vals))
    if len(shape) == 1:
        body = ",\n    ".join(_chunk_hex(flat_vals, 8))
        return f"static const uint16_t {name}[{shape[0]}] = {{\n    {body}\n}};\n"
    if len(shape) == 2:
        rows, cols = shape
        lines = []
        for r in range(rows):
            row_vals = flat_vals[r * cols:(r + 1) * cols]
            row_body = ", ".join(f"0x{v & 0xFFFF:04x}" for v in row_vals)
            lines.append(f"    {{ {row_body} }}")
        body = ",\n".join(lines)
        return f"static const uint16_t {name}[{rows}][{cols}] = {{\n{body}\n}};\n"
    raise ValueError(f"unsupported shape {shape}")


def _prod(shape: tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= s
    return out


def _chunk_hex(vals: list[int], per_line: int) -> list[str]:
    out = []
    for i in range(0, len(vals), per_line):
        out.append(", ".join(f"0x{v & 0xFFFF:04x}" for v in vals[i:i + per_line]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-header", default=None,
                    help="Write C header to this path (default: stdout)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--interlude-mode", choices=INTERLUDE_MODES,
                    default="truncate_upper16",
                    help="Saturn BF16 interlude semantics for BOTH interludes "
                         "(Y->Y' and Z->Z''). truncate_upper16 (default) matches "
                         "the legacy bit-hack RVV path; native_bf16_rne matches "
                         "Saturn vfmul.vv altfmt=1 + vfncvtbf16.f.f.w.")
    ap.add_argument("--guard-suffix", default=None,
                    help="Optional include-guard suffix. Default: none for "
                         "truncate, _NATIVE_BF16 for native.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    a_tile = torch.randn(TILE_ROWS, TILE_COLS, dtype=torch.bfloat16)
    b_tile = torch.randn(TILE_ROWS, TILE_COLS, dtype=torch.bfloat16)
    scale_tile = torch.randn(TILE_ROWS, dtype=torch.bfloat16)
    post_scale_tile = torch.randn(TILE_ROWS, dtype=torch.bfloat16)

    a_rows = bf16_tile_to_bank_rows(a_tile)
    b_rows = bf16_tile_to_bank_rows(b_tile)
    scale_bits = [int(v) & 0xFFFF for v in scale_tile.view(torch.int16).tolist()]
    post_scale_bits = [int(v) & 0xFFFF for v in post_scale_tile.view(torch.int16).tolist()]

    # Phase 1: Atlas mul (VectorEngineModel via run_binary_rows)
    y_rows = run_binary_rows("mul", a_rows, b_rows)

    # Interlude 1: Saturn-side BF16 scale, semantics chosen by --interlude-mode
    yp_rows = saturn_interlude(y_rows, scale_bits, mode=args.interlude_mode)

    # Phase 2: Atlas row-reduce (broadcasts scalar back to both banks)
    s_rows = run_row_reduce_tensor("rsum", yp_rows)

    # Phase 3: Atlas add (Y' + S), lane-wise. Both Y' and S are (64, 16).
    z_rows = run_binary_rows("add", yp_rows, s_rows)

    # Interlude 2: same mode as interlude 1 (a single test exercises one
    # native-vs-truncate policy end-to-end; mixing modes within one test
    # would muddy the semantic comparison).
    z2_rows = saturn_interlude(z_rows, post_scale_bits, mode=args.interlude_mode)

    if args.guard_suffix is not None:
        guard_suffix = args.guard_suffix
    elif args.interlude_mode == "native_bf16_rne":
        guard_suffix = "_NATIVE_BF16"
    else:
        guard_suffix = ""
    guard = f"SATURN_ATLAS_TRI_PHASE{guard_suffix}_GOLDEN_H"

    header = []
    header.append("/* AUTO-GENERATED by generators/sp26-atlas-acc/baremetal/generators/")
    header.append(" *   gen_saturn_atlas_tri_phase.py")
    header.append(f" * Seed: {args.seed}")
    header.append(f" * Interlude mode: {args.interlude_mode}")
    header.append(" * Do not edit by hand. */")
    header.append(f"#ifndef {guard}")
    header.append(f"#define {guard}")
    header.append("")
    header.append("#include <stdint.h>")
    header.append("")
    header.append(f"#define SAC_TRI_TILE_ROWS   {TILE_ROWS}")
    header.append(f"#define SAC_TRI_TILE_COLS   {TILE_COLS}")
    header.append(f"#define SAC_TRI_TOTAL_ROWS  {TOTAL_ROWS}   /* bank0 + bank1, 16 lanes each */")
    header.append(f"#define SAC_TRI_LANES       {LANES}")
    header.append("")
    header.append(emit_u16_array("SAC_TRI_A",           (TOTAL_ROWS, LANES), flatten_rows(a_rows)))
    header.append(emit_u16_array("SAC_TRI_B",           (TOTAL_ROWS, LANES), flatten_rows(b_rows)))
    header.append(emit_u16_array("SAC_TRI_SCALE",       (TILE_ROWS,),         scale_bits))
    header.append(emit_u16_array("SAC_TRI_POST_SCALE",  (TILE_ROWS,),         post_scale_bits))
    # Intermediates (for driver-side diagnostic checkpoints between phases).
    header.append(emit_u16_array("SAC_TRI_EXPECT_Y",    (TOTAL_ROWS, LANES), flatten_rows(y_rows)))
    header.append(emit_u16_array("SAC_TRI_EXPECT_YP",   (TOTAL_ROWS, LANES), flatten_rows(yp_rows)))
    header.append(emit_u16_array("SAC_TRI_EXPECT_S",    (TOTAL_ROWS, LANES), flatten_rows(s_rows)))
    header.append(emit_u16_array("SAC_TRI_EXPECT_Z",    (TOTAL_ROWS, LANES), flatten_rows(z_rows)))
    header.append(emit_u16_array("SAC_TRI_EXPECT_Z2",   (TOTAL_ROWS, LANES), flatten_rows(z2_rows)))
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

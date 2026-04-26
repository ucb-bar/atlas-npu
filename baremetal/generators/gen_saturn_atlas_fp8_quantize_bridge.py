#!/usr/bin/env python3
"""Golden generator for tests/saturn_atlas_fp8_quantize_bridge.c.

One-way Saturn -> Atlas FP8 handshake test. Two checkpoints:

  Checkpoint 1 (Saturn quantizer alignment):
      Saturn quantizes A_bf16 -> A_fp8 (E4M3) via native vfncvtbf16.f.f.w.
      Host compares Saturn-produced bytes against bf16_tile_to_e4m3_bytes
      (which mirrors Atlas's own VMATPUSH.ACC.BF16 + SELI 0x7F + VMATPOP.FP8
      roundtrip). If checkpoint 1 fails, Saturn and Atlas disagree on
      E4M3 bit-encoding for normal-finite inputs -- that is a real
      cross-engine FP8-format finding.

  Checkpoint 2 (end-to-end correctness):
      Atlas runs smolvla_fused_matmul_bias_v2_mxu1 consuming Saturn's
      A_fp8 + host-prequantized B_fp8 + BF16 bias. Host compares the
      BF16 output against `IPTLinearRTLFunction + run_binary_rows("add",
      ...)` chain through software_models. This catches matmul or
      bias-add bugs separately from the encoding question.

Atlas operates on a perfect-software assumption: no Inf/NaN/subnormal
inputs. The Saturn-quantized A_bf16 tile is restricted to |x| in
[0.25, 2.0) because Atlas's bf16_tile_to_e4m3_bytes flushes near-zero
values to +/-0 while Saturn emits E4M3 subnormals for the same inputs
(real cross-engine encoding divergence outside the well-defined
domain). B_bf16 and bias do NOT go through Saturn's quantizer, so they
use the unrestricted torch.randn distribution.

DRAM layout (matches smolvla_fused_matmul_bias_v2_mxu1.S kernel + a
Saturn staging slot at 0x2000):
    [0x0000..0x03FF]  A fp8                (32x32, 1024 B)   <- Saturn writes
    [0x0400..0x07FF]  B fp8                (32x32, 1024 B)   <- host seeds
    [0x0800..0x0FFF]  bias bf16            (32x32, 2048 B)   <- host seeds
    [0x1000..0x17FF]  out bf16             (32x32, 2048 B)   <- Atlas writes
    [0x2000..0x27FF]  A bf16  (Saturn src) (32x32, 2048 B)   <- host seeds

Emits a C header consumed by the test driver.
"""

from __future__ import annotations

import argparse
import sys

import torch

from mxu_fp8_utils import bf16_tile_to_e4m3_bytes
from software_models.mxu1_ipt.fp_formats import OutputFmtSel
from software_models.mxu1_ipt.ipt_rtl_linear import (
    IPTLinearRTLFunction,
    e4m3_bytes_to_float,
)
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    run_binary_rows,
)


SEED = 42
TILE_ROWS = ROWS_PER_REGISTER          # 32
TILE_COLS = 2 * BF16_PER_BEAT           # 32
TOTAL_BANK_ROWS = ROWS_PER_TENSOR       # 64 (bank0 + bank1, each 16 lanes)
LANES_BANK = BF16_PER_BEAT              # 16 BF16 lanes per bank


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> list[list[int]]:
    """Split a (32, 32) BF16 tile into 64 rows of 16 BF16 lanes (bank0 then bank1)."""
    if tile.shape != (TILE_ROWS, TILE_COLS):
        raise ValueError(f"expected {TILE_ROWS}x{TILE_COLS} tile, got {tuple(tile.shape)}")
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bank0 = tile[:, :LANES_BANK].contiguous().view(torch.int16)
    bank1 = tile[:, LANES_BANK:].contiguous().view(torch.int16)
    rows: list[list[int]] = []
    for half in (bank0, bank1):
        for i in range(TILE_ROWS):
            rows.append([int(v) & 0xFFFF for v in half[i].tolist()])
    return rows


def bf16_tile_to_flat_rows(tile: torch.Tensor) -> list[list[int]]:
    """Row-major (32, 32) BF16 tile as 32 rows of 32 BF16 lanes (no bank split).
    This matches the Saturn-side layout: each row is a flat 32-lane slab."""
    if tile.shape != (TILE_ROWS, TILE_COLS):
        raise ValueError(f"expected {TILE_ROWS}x{TILE_COLS} tile, got {tuple(tile.shape)}")
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bits = tile.contiguous().view(torch.int16)
    return [[int(v) & 0xFFFF for v in bits[i].tolist()] for i in range(TILE_ROWS)]


def fp8_tile_to_flat_rows(tile_uint8: torch.Tensor) -> list[list[int]]:
    """(32, 32) uint8 FP8 tile as 32 rows of 32 bytes (matches Atlas DRAM layout)."""
    if tile_uint8.shape != (TILE_ROWS, TILE_COLS):
        raise ValueError(f"expected {TILE_ROWS}x{TILE_COLS} fp8 tile, got {tuple(tile_uint8.shape)}")
    arr = tile_uint8.numpy()
    return [[int(v) & 0xFF for v in arr[i].tolist()] for i in range(TILE_ROWS)]


def _prod(shape: tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= s
    return out


def emit_u16_array(name: str, shape: tuple[int, ...], flat: list[int]) -> str:
    assert len(flat) == _prod(shape), (name, shape, len(flat))
    if len(shape) == 2:
        rows, cols = shape
        lines = []
        for r in range(rows):
            row_vals = flat[r * cols:(r + 1) * cols]
            row_body = ", ".join(f"0x{v & 0xFFFF:04x}" for v in row_vals)
            lines.append(f"    {{ {row_body} }}")
        body = ",\n".join(lines)
        return f"static const uint16_t {name}[{rows}][{cols}] = {{\n{body}\n}};\n"
    raise ValueError(f"unsupported u16 shape {shape}")


def emit_u8_array(name: str, shape: tuple[int, ...], flat: list[int]) -> str:
    assert len(flat) == _prod(shape), (name, shape, len(flat))
    if len(shape) == 2:
        rows, cols = shape
        lines = []
        for r in range(rows):
            row_vals = flat[r * cols:(r + 1) * cols]
            row_body = ", ".join(f"0x{v & 0xFF:02x}" for v in row_vals)
            lines.append(f"    {{ {row_body} }}")
        body = ",\n".join(lines)
        return f"static const uint8_t {name}[{rows}][{cols}] = {{\n{body}\n}};\n"
    raise ValueError(f"unsupported u8 shape {shape}")


def flatten_rows(rows: list[list[int]]) -> list[int]:
    flat: list[int] = []
    for r in rows:
        flat.extend(r)
    return flat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-header", default=None,
                    help="Write C header to this path (default: stdout)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    # ── Inputs (normal-finite BF16; respects Atlas's perfect-software domain) ──
    #
    # A_bf16 is the input to Saturn's vfncvtbf16 narrow. Atlas's reference
    # converter `bf16_tile_to_e4m3_bytes` flushes BF16 values whose scaled
    # unbiased exponent falls below -6 to +0/-0 (mxu_fp8_utils.py:7-9 — no
    # E4M3 subnormals emitted). Saturn's native vfncvtbf16 path *does*
    # emit E4M3 subnormals for the same inputs, so unbounded randn seeds
    # produce a known divergence for near-zero values. Per the Atlas
    # perfect-software invariant in the master plan, cooperative tests
    # must seed away from the underflow region. We use a uniform-magnitude
    # |x| in [0.25, 2.0) which sits well above E4M3's smallest normal
    # (2^-6 ≈ 0.0156) and inside Atlas's well-behaved domain.
    torch.manual_seed(args.seed)
    sign_a = torch.randint(0, 2, (TILE_ROWS, TILE_COLS)).float() * 2 - 1
    mag_a  = torch.rand(TILE_ROWS, TILE_COLS) * 1.75 + 0.25
    A_bf16 = (sign_a * mag_a).to(torch.bfloat16)                        # Saturn input
    # B and bias don't go through Saturn — B is host-quantized via the
    # same bf16_tile_to_e4m3_bytes, and the IPT golden decodes the same
    # bytes (no Saturn-vs-golden divergence path). Use unrestricted randn.
    B_bf16   = (torch.randn(TILE_ROWS, TILE_COLS) * 0.5).to(torch.bfloat16)
    bias_raw = torch.randn(TILE_ROWS, TILE_COLS, dtype=torch.bfloat16)

    # ── Saturn-side golden: BF16 -> E4M3 via the same converter Atlas uses ──
    A_fp8_expected = bf16_tile_to_e4m3_bytes(A_bf16, scale_exp=0)   # checkpoint 1
    B_fp8 = bf16_tile_to_e4m3_bytes(B_bf16, scale_exp=0)            # host preload

    # ── Atlas-side golden via IPT + VPU bias add (same chain as
    #    gen_smolvla_fused_matmul_bias_v2_mxu1.py) ────────────────────────────
    ipt_bf16 = IPTLinearRTLFunction(
        vec_len=TILE_ROWS, num_lanes=TILE_ROWS, pipeline_depth=1,
        out_fmt_sel=OutputFmtSel.OutBF16,
    )
    # e4m3_bytes_to_float wants torch tensors (uses .device).
    a_fp32 = e4m3_bytes_to_float(A_fp8_expected)
    w_fp32 = e4m3_bytes_to_float(B_fp8.T.contiguous())
    mat_bf16 = ipt_bf16(a_fp32, w_fp32, scale_exp=0).to(torch.bfloat16)

    mat_rows  = bf16_tile_to_bank_rows(mat_bf16)
    bias_rows = bf16_tile_to_bank_rows(bias_raw)
    out_rows  = run_binary_rows("add", mat_rows, bias_rows)         # checkpoint 2

    # ── Pack into header ─────────────────────────────────────────────────
    a_bf16_flat_rows = bf16_tile_to_flat_rows(A_bf16)
    a_fp8_flat_rows  = fp8_tile_to_flat_rows(A_fp8_expected)
    b_fp8_flat_rows  = fp8_tile_to_flat_rows(B_fp8)

    header: list[str] = []
    header.append("/* AUTO-GENERATED by generators/sp26-atlas-acc/baremetal/generators/")
    header.append(" *   gen_saturn_atlas_fp8_quantize_bridge.py")
    header.append(f" * Seed: {args.seed}")
    header.append(" * Do not edit by hand. */")
    header.append("#ifndef SATURN_ATLAS_FP8_QUANTIZE_BRIDGE_GOLDEN_H")
    header.append("#define SATURN_ATLAS_FP8_QUANTIZE_BRIDGE_GOLDEN_H")
    header.append("")
    header.append("#include <stdint.h>")
    header.append("")
    header.append(f"#define SAC_BR_TILE_ROWS    {TILE_ROWS}    /* 32 */")
    header.append(f"#define SAC_BR_TILE_COLS    {TILE_COLS}    /* 32 */")
    header.append(f"#define SAC_BR_TOTAL_BANK_ROWS  {TOTAL_BANK_ROWS}   /* bank0 + bank1, 16 lanes each */")
    header.append(f"#define SAC_BR_LANES_BANK   {LANES_BANK}   /* 16 BF16 lanes per bank */")
    header.append("")
    # Saturn input + checkpoint 1 golden (flat 32 rows x 32 lanes; matches
    # Saturn's per-row vle16/vfncvt/vse8 contract — no bank split).
    header.append(emit_u16_array("SAC_BR_A_BF16",
                                 (TILE_ROWS, TILE_COLS),
                                 flatten_rows(a_bf16_flat_rows)))
    header.append(emit_u8_array ("SAC_BR_EXPECT_A_FP8",
                                 (TILE_ROWS, TILE_COLS),
                                 flatten_rows(a_fp8_flat_rows)))
    header.append(emit_u8_array ("SAC_BR_B_FP8",
                                 (TILE_ROWS, TILE_COLS),
                                 flatten_rows(b_fp8_flat_rows)))
    # Bias + checkpoint 2 golden in bank-split layout (matches Atlas DRAM).
    header.append(emit_u16_array("SAC_BR_BIAS",
                                 (TOTAL_BANK_ROWS, LANES_BANK),
                                 flatten_rows(bias_rows)))
    header.append(emit_u16_array("SAC_BR_EXPECT_OUT",
                                 (TOTAL_BANK_ROWS, LANES_BANK),
                                 flatten_rows(out_rows)))
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

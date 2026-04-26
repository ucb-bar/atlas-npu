#!/usr/bin/env python3
"""Golden generator for tests/saturn_atlas_fp8_dequant_bridge.c.

One-way Atlas -> Saturn FP8 handshake test. Together with the partner
test saturn_atlas_fp8_quantize_bridge (Saturn -> Atlas direction), this
closes the bidirectional FP8 handoff story. "Handoff" rather than
"byte equivalence": each side is verified against its own correct
golden, but Atlas's scaled VFP8UNPACK and Saturn's no-scale
vfwcvtbf16 produce different bits by construction (the E8M0 scale
applies only to the Atlas path).

  stage 0   host:    seed A_fp8, B_fp8, scale byte in DRAM (same layout
                     mxu1_quant_relu_roundtrip.S expects)
  stage 1   Atlas:   run mxu1_quant_relu_roundtrip kernel:
                       MXU1 matmul (A_fp8 @ B_fp8 -> BF16 raw)
                       VPU ReLU on BF16 raw
                       VFP8PACK pack to FP8 (with E8M0 scale)
                       VFP8UNPACK roundtrip back to BF16
                     Atlas writes packed FP8 to DRAM[0x1800].

  ---- checkpoint 1 (Atlas FP8 production) ----
    compare Atlas's packed FP8 bytes at 0x1800 against the Python golden
    via IPTLinearRTLFunction + run_unary_rows("relu",...) + run_fp8_pack_rows.
    Same model the existing baremetal mxu1_quant_relu_roundtrip test uses,
    which is grounded in the actual VPU pack semantics.

  stage 2   Saturn:  load the same FP8 bytes via vle8.v, widen via native
                     vfwcvtbf16.f.f.v (E4M3 -> BF16, no scale), store BF16
                     to a fresh DRAM region (0x3000).

  ---- checkpoint 2 (Saturn FP8 decode) ----
    compare Saturn's decoded BF16 at 0x3000 against the Python golden via
    e4m3_bytes_to_float (canonical E4M3 -> FP32) cast to BF16. Saturn's
    vfwcvtbf16.f.f.v does no scale; this golden does no scale.

Atlas's own VFP8UNPACK output at 0x2000 applies the E8M0 scale (0x7E =
2^-1), so it is intentionally not compared against Saturn's no-scale
decode. This cooperative test verifies Atlas's packed FP8 bytes at
0x1800 and Saturn's no-scale decode at 0x3000; the existing
mxu1_quant_relu_roundtrip baremetal test covers Atlas's scaled unpack
output. The question this test answers is "do Atlas FP8 production
and Saturn FP8 decode each match their own model?" -- not "are
Atlas's and Saturn's decode semantics the same?".

DRAM layout:
    [0x0000..0x03FF]  A_fp8 (1024 B)            <- host seeds
    [0x0400..0x07FF]  B_fp8 (1024 B; B^T)       <- host seeds
    [0x0800..0x081F]  scale byte (32 B)         <- host seeds
    [0x1000..0x17FF]  Atlas raw BF16 output     <- kernel writes (not verified here)
    [0x1800..0x1BFF]  Atlas packed FP8 (1024 B) <- kernel writes; ckpt 1
    [0x2000..0x27FF]  Atlas unpacked BF16       <- kernel writes (not verified here)
    [0x3000..0x37FF]  Saturn decoded BF16       <- Saturn writes; ckpt 2

Emits a C header consumed by the test driver.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from gen_utils import (
    fp8_e4m3_bits_to_float,
    quantize_fp8,
    float_to_bf16_bits,
)
from software_models.mxu1_ipt.fp_formats import OutputFmtSel
from software_models.mxu1_ipt.ipt_rtl_linear import (
    IPTLinearRTLFunction,
    e4m3_bytes_to_float,
)
from vpu_gen_utils import run_fp8_pack_rows, run_unary_rows


SEED_A = 202
SEED_B = 303
TILE = 32
SCALE_E8M0 = 0x7E   # 2^-1; matches mxu1_quant_relu_roundtrip kernel
TIMEOUT = 500000


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


def fp32_to_e4m3_byte(x: float) -> int:
    """Quantize an FP32 value to a single E4M3 byte using gen_utils.quantize_fp8.
    quantize_fp8 takes an array; we go through it for consistency with the kernel.
    """
    arr = np.array([[x]], dtype=np.float32)
    qf = quantize_fp8(arr).astype(np.float32)
    # Encode the quantized FP32 back to E4M3 bytes: walk the LUT in reverse.
    # Simpler path: quantize_fp8 is idempotent and fp8_e4m3_bits_to_float is
    # bijective on canonical bytes, so we can search the LUT.
    bits_to_val = {b: fp8_e4m3_bits_to_float(b) for b in range(256)}
    target = float(qf[0, 0])
    for b, v in bits_to_val.items():
        if v == target or (np.isnan(v) and np.isnan(target)):
            return b
    raise ValueError(f"no E4M3 byte encodes {target}")


def fp32_matrix_to_fp8_bytes(mat: np.ndarray) -> np.ndarray:
    """Map an FP32 32x32 matrix (already FP8-quantized) to uint8 E4M3 bytes."""
    rows, cols = mat.shape
    out = np.zeros((rows, cols), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            out[r, c] = fp32_to_e4m3_byte(float(mat[r, c]))
    return out


def packed_fp8_rows_to_uint8_matrix(rows: list[list[int]]) -> np.ndarray:
    """run_fp8_pack_rows returns 32 rows x 16 uint16 slots (each slot = 2 FP8
    bytes in little-endian). Re-pack as a (32, 32) uint8 byte matrix."""
    if len(rows) != TILE:
        raise ValueError(f"expected {TILE} rows, got {len(rows)}")
    out = np.zeros((TILE, TILE), dtype=np.uint8)
    for r, row in enumerate(rows):
        if len(row) != 16:
            raise ValueError(f"expected 16 slots per row, got {len(row)}")
        for c, slot in enumerate(row):
            slot = int(slot) & 0xFFFF
            out[r, 2 * c]     = slot & 0xFF
            out[r, 2 * c + 1] = (slot >> 8) & 0xFF
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-header", default=None,
                    help="Write C header to this path (default: stdout)")
    args = ap.parse_args()

    # ── Seeds + FP8 quantize (matches mxu1_quant_relu_roundtrip's gen) ──
    rng_a = np.random.RandomState(SEED_A)
    rng_b = np.random.RandomState(SEED_B)
    A = rng_a.uniform(-1.25, 1.25, size=(TILE, TILE)).astype(np.float32)
    B = rng_b.uniform(-1.25, 1.25, size=(TILE, TILE)).astype(np.float32)
    A_q = quantize_fp8(A).astype(np.float32)
    B_q = quantize_fp8(B).astype(np.float32)
    B_preload = B_q.T.copy()                     # MXU pushes B directly as W

    # ── Atlas-side golden through software_models ──
    ipt = IPTLinearRTLFunction(out_fmt_sel=OutputFmtSel.OutBF16)
    C = ipt(torch.from_numpy(A_q), torch.from_numpy(B_preload)).numpy().astype(np.float32)

    # MXU1 BF16 result laid out as bank-pair rows: lo (cols 0:16), hi (cols 16:32).
    C_bits = np.vectorize(float_to_bf16_bits)(C).astype(np.uint16)
    raw_lo_rows = [list(C_bits[r, :16])  for r in range(TILE)]
    raw_hi_rows = [list(C_bits[r, 16:32]) for r in range(TILE)]
    raw_tensor_rows = raw_lo_rows + raw_hi_rows

    # ReLU + FP8 pack via VectorEngineModel (matches VPU semantics).
    relu_rows = run_unary_rows("relu", raw_tensor_rows)
    pack_rows = run_fp8_pack_rows(relu_rows[:TILE], relu_rows[TILE:2 * TILE], SCALE_E8M0)
    pack_bytes = packed_fp8_rows_to_uint8_matrix(pack_rows)   # (32, 32) uint8

    # ── Atlas inputs as uint8 byte matrices ──
    a_fp8_bytes = fp32_matrix_to_fp8_bytes(A_q)
    b_fp8_bytes = fp32_matrix_to_fp8_bytes(B_preload)

    # ── Saturn-side golden: e4m3_bytes_to_float -> BF16 cast ──
    # Saturn's vfwcvtbf16.f.f.v at SEW=E8 alt=0 produces BF16 directly from
    # E4M3 bytes. The reference path is e4m3_bytes_to_float (canonical
    # E4M3 -> FP32 LUT) followed by RNE narrow to BF16 via torch's cast.
    pack_bytes_t = torch.from_numpy(pack_bytes)
    pack_fp32 = e4m3_bytes_to_float(pack_bytes_t)
    saturn_bf16 = pack_fp32.to(torch.bfloat16).view(torch.int16).numpy().astype(np.uint16)

    # ── Header emission ──
    a_flat = a_fp8_bytes.reshape(-1).tolist()
    b_flat = b_fp8_bytes.reshape(-1).tolist()
    pack_flat = pack_bytes.reshape(-1).tolist()
    saturn_flat = saturn_bf16.reshape(-1).tolist()

    header = []
    header.append("/* AUTO-GENERATED by generators/sp26-atlas-acc/baremetal/generators/")
    header.append(" *   gen_saturn_atlas_fp8_dequant_bridge.py")
    header.append(f" * Seeds: A={SEED_A}, B={SEED_B}, scale=0x{SCALE_E8M0:02x}")
    header.append(" * Do not edit by hand. */")
    header.append("#ifndef SATURN_ATLAS_FP8_DEQUANT_BRIDGE_GOLDEN_H")
    header.append("#define SATURN_ATLAS_FP8_DEQUANT_BRIDGE_GOLDEN_H")
    header.append("")
    header.append("#include <stdint.h>")
    header.append("")
    header.append(f"#define SAC_FQ_TILE_ROWS    {TILE}    /* 32 */")
    header.append(f"#define SAC_FQ_TILE_COLS    {TILE}    /* 32 */")
    header.append(f"#define SAC_FQ_SCALE_BYTE   0x{SCALE_E8M0:02x}")
    header.append("")
    header.append(emit_u8_array ("SAC_FQ_A_FP8",            (TILE, TILE), a_flat))
    header.append(emit_u8_array ("SAC_FQ_B_FP8",            (TILE, TILE), b_flat))
    header.append(emit_u8_array ("SAC_FQ_EXPECT_PACK_FP8",  (TILE, TILE), pack_flat))
    header.append(emit_u16_array("SAC_FQ_EXPECT_SATURN_BF16",(TILE, TILE), saturn_flat))
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

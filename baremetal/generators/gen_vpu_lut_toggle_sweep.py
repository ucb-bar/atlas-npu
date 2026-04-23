#!/usr/bin/env python3
"""
gen_vpu_lut_toggle_sweep.py — golden generator for vpu_lut_toggle_sweep.S

This test targets the worst toggle-coverage leaves reported in
atlasTile_coverage_report.md:

    sqrt.lut     toggle  5.21%
    log2.lut     toggle  3.37%
    exp.lut      toggle 11.66%
    tanh.lut     toggle 11.66%
    rcp.lut      toggle  3.43%

Each of these LUTs is addressed by the input BF16 mantissa/exponent
bits. Existing unary-math tests feed CONSTANT tiles (all 1.0 or all
-2.0), so they only ever look up 1 or 2 LUT entries per op. This
generator constructs three deliberately-diverse 32x32 BF16 tiles whose
bit patterns span a large subset of all possible mantissa and exponent
values, then runs eight VPU unary ops on every tile.

We rely on the Atlas VPU software functional model (VectorEngineModel)
so that the expected outputs match hardware rounding exactly — the same
contract the existing gen_vpu_unary_{math,simple}.py scripts use.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generators"))

from gen_utils import (
    float_to_bf16_bits,
    bf16_bits_to_float,
    quantize_bf16,
)
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_TENSOR,
    pack_u16_le,
    run_unary_rows,
)

# ---------------------------------------------------------------------------
# DRAM layout (must match the assembly)
# ---------------------------------------------------------------------------

TILE_BYTES = 2048                     # 2 KiB = 32x32 BF16
TILE_BEATS = TILE_BYTES // 32         # 64 beats of 32 B

TILE_BASES_BYTES = [0x0000, 0x0800, 0x1000]
OUTPUT_BASE_BYTES = 0x2000
OUTPUT_STRIDE_BYTES = 0x0800          # 2 KiB per output pass

# Op order MUST match the assembly: VEXP, VEXP2, VSIN, VCOS, VTANH,
# VLOG2, VSQRT, VRECIP.BF16.
#
# The Atlas VectorEngineModel uses lowercase op names that match the
# existing gen_vpu_unary_*.py scripts.
OP_SEQUENCE = [
    ("exp",  "VEXP"),
    ("exp2", "VEXP2"),
    ("sin",  "VSIN"),
    ("cos",  "VCOS"),
    ("tanh", "VTANH"),
    ("log",  "VLOG2"),
    ("sqrt", "VSQRT"),
    ("rcp",  "VRECIP.BF16"),
]

TIMEOUT = 200000


# ---------------------------------------------------------------------------
# Tile constructors
# ---------------------------------------------------------------------------
#
# Each tile is a 32x32 float32 matrix that will be BF16-quantized before
# being packed into rows. We target value distributions that drive many
# different mantissa and exponent bits into each of the LUT-backed VPU
# lane boxes. The goal is purely toggle coverage, not a "useful" compute
# shape — but the oracle is still derived from the real VPU functional
# model, so every output beat is exact.

def build_tile0() -> np.ndarray:
    """Tile 0 — mantissa sweep, exponent clustered around 0.

    Values are all positive and span roughly [0.5, 3.99], which keeps
    every LUT-op input inside the "safe" domain for log2/rcp/sqrt
    (positive, non-subnormal). The mantissa field visits many distinct
    7-bit patterns across the 1024 lanes.
    """
    n = 32 * 32
    # A dense mantissa sweep in [1.0, 2.0), repeated across two octaves.
    mant_sweep = np.linspace(1.0, 1.99999, 512, dtype=np.float32)
    vals = np.concatenate([mant_sweep, 2.0 * mant_sweep])
    return vals.reshape(32, 32)


def build_tile1() -> np.ndarray:
    """Tile 1 — moderate magnitude, all positive.

    Values span [0.0625, 6.0] so that sin/cos/tanh/exp/exp2 receive
    inputs in their interesting ranges, while keeping VLOG2/VSQRT in
    the positive domain (all three LUT back-ends care about mantissa
    only, so we get no toggle benefit from sign bits). Evenly sampled
    plus a tiny periodic perturbation so adjacent lanes drive different
    mantissa bits into the LUT address buses.
    """
    n = 32 * 32
    vals = np.linspace(0.0625, 6.0, n, dtype=np.float32)
    perturb = 1e-3 * np.abs(np.sin(np.arange(n, dtype=np.float32) * 0.37))
    return (vals + perturb).reshape(32, 32)


def build_tile2() -> np.ndarray:
    """Tile 2 — wide positive exponent sweep.

    Values cover many octaves in [2^-6, 2^6] so the BF16 exponent field
    takes many distinct values and the exponent-indexed portion of each
    LUT toggles heavily. All-positive so VLOG2 and VSQRT stay in their
    real-valued domain.
    """
    n = 32 * 32
    vals = np.logspace(np.log10(0.015625), np.log10(60.0), n, dtype=np.float32)
    # Interleave two slightly different log sweeps so consecutive lanes
    # do not share identical mantissas.
    vals2 = np.logspace(np.log10(0.02), np.log10(50.0), n, dtype=np.float32)
    interleaved = np.empty(n, dtype=np.float32)
    interleaved[0::2] = vals[0::2]
    interleaved[1::2] = vals2[1::2]
    return interleaved.reshape(32, 32)


TILE_BUILDERS = [build_tile0, build_tile1, build_tile2]


# ---------------------------------------------------------------------------
# Packing helpers
# ---------------------------------------------------------------------------

def float_matrix_to_bf16_rows(mat: np.ndarray) -> list[list[int]]:
    """Convert a 32x32 float matrix to the row-of-16-BF16 layout the
    VPU model expects. Each source matrix supplies TWO 32x16 tensor
    banks (lo lanes and hi lanes), giving 64 BF16 rows in total —
    the standard ROWS_PER_TENSOR layout.
    """
    lo = mat[:, :BF16_PER_BEAT]
    hi = mat[:, BF16_PER_BEAT:]
    rows: list[list[int]] = []
    for r in range(mat.shape[0]):
        rows.append([float_to_bf16_bits(float(lo[r, c])) for c in range(BF16_PER_BEAT)])
    for r in range(mat.shape[0]):
        rows.append([float_to_bf16_bits(float(hi[r, c])) for c in range(BF16_PER_BEAT)])
    return rows


def rows_to_preload_entries(base_byte_off: int, rows: list[list[int]]) -> list[dict]:
    assert base_byte_off % 32 == 0
    base_beat = base_byte_off // 32
    assert len(rows) == ROWS_PER_TENSOR, (
        f"tile must have {ROWS_PER_TENSOR} BF16 rows, got {len(rows)}"
    )
    return [
        {"word_offset": base_beat + i, "data": pack_u16_le(row)}
        for i, row in enumerate(rows)
    ]


def rows_to_check_entries(base_byte_off: int, rows: list[list[int]]) -> list[dict]:
    assert base_byte_off % 32 == 0
    base_beat = base_byte_off // 32
    assert len(rows) == ROWS_PER_TENSOR, (
        f"output tile must have {ROWS_PER_TENSOR} BF16 rows, got {len(rows)}"
    )
    return [
        {"word_offset": base_beat + i, "expected": pack_u16_le(row)}
        for i, row in enumerate(rows)
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    preloads: list[dict] = []
    checks: list[dict] = []

    for tile_idx, builder in enumerate(TILE_BUILDERS):
        tile = quantize_bf16(builder().astype(np.float32))
        input_rows = float_matrix_to_bf16_rows(tile)
        preloads.extend(rows_to_preload_entries(TILE_BASES_BYTES[tile_idx], input_rows))

        for op_idx, (model_op, _asm_mnem) in enumerate(OP_SEQUENCE):
            output_rows = run_unary_rows(model_op, input_rows)
            pass_idx = tile_idx * len(OP_SEQUENCE) + op_idx
            out_off = OUTPUT_BASE_BYTES + pass_idx * OUTPUT_STRIDE_BYTES
            checks.extend(rows_to_check_entries(out_off, output_rows))

    sys.stderr.write(
        f"[gen_vpu_lut_toggle_sweep] tiles={len(TILE_BUILDERS)} "
        f"ops_per_tile={len(OP_SEQUENCE)} "
        f"preloads={len(preloads)} checks={len(checks)}\n"
    )

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

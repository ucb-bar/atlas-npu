#!/usr/bin/env python3
"""
gen_vpu_lut_bitflip_megasweep.py — golden generator for
vpu_lut_bitflip_megasweep.S.

This test layers on top of the existing vpu_lut_toggle_sweep and
vpu_raw_lut_class_stress suites. Those tests already feed mantissa-
diverse tiles, but the LUT-leaf toggle metric (`core.vpu.core.sqrt.lut`,
`log2.lut`, `rcp.lut`, `exp.lut`, `tanh.lut`) is still in the single
digits. The bottleneck is per-LANE per-CYCLE bit transitions in the
LUT result registers and in the lutFixedToBf16 conversion paths.

Strategy:
  Build four 32x32 BF16 input tiles where, in EVERY column, the 32
  consecutive values are deliberately chosen to drive maximally-
  toggling LUT inputs into a single hardware lane:

    Tile A "walk_pos"      All-positive mantissa zig-zag (0, 0x7F, 1,
                           0x7E, ...) XORed with a per-column mask, plus
                           an exponent whose LSB ALTERNATES every row
                           (drives SqrtLUT.oddExp), and a per-column
                           exponent band that sweeps the BF16 exponent
                           field. Output sign bit toggles for VLOG2
                           because some inputs are <1 and some >1.

    Tile B "walk_signed"   Same walk as A but with a sign pattern that
                           alternates every two rows AND every two
                           columns. Drives RcpLUT.neg toggle and forces
                           VRECIP / VEXP / VTANH to span both signs.
                           VSQRT / VLOG2 of negatives yield NaN, which
                           keeps the result reg's bit pattern flipping
                           between LUT-derived and 0x7FC1 (good toggle).

    Tile C "class_lattice" Special-class lattice: signed +/-0,
                           subnormals, +/-min normal, +/-1, +/-2, +/-4,
                           +/-8, +/-16, max finite, +/-inf, NaNs, plus
                           a few well-known fractional classes. Drives
                           the special-case branches in
                           lutFixedToBf16Sqrt/Rcp/Log so the upper
                           output bits (sign / exp / nanExp) flip across
                           NaN, +/-inf, and finite results.

    Tile D "magnitude_alt" Alternate large (exp=0x88 ~ 2^9) and small
                           (exp=0x70 ~ 2^-15) magnitudes per row, with
                           complementary mantissa patterns. Maximizes
                           cycle-to-cycle bit toggling in the LUT res
                           reg by forcing the BF16 output exp field to
                           swing widely on every cycle.

For each tile we run 8 unary ops:
    VSQRT, VLOG2, VRECIP.BF16, VEXP, VEXP2, VTANH, VSQUARE.BF16,
    VCUBE.BF16

That's 32 unary outputs (~64 KiB of golden checks). After the unary
sweep, a chain phase reuses Tile A through five LUT pipes
(VLOG2 + VSQRT + VEXP2 + VRECIP + VTANH) and combines the products
through VMUL.BF16 / VADD.BF16 / VSUB.BF16, VREDSUM.ROW.BF16,
VFP8PACK + VFP8UNPACK, and VREDSUM.BF16 (column reduce). This drives
the AddSubSum / ColAdd reduction trees and the FP8 datapath using
LUT-derived (not constant) operands.

The oracle is the Atlas VPU functional model (VectorEngineModel),
the same source used by gen_vpu_unary_*.py / gen_vpu_overlap_*.py.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data  # noqa: E402
from vpu_gen_utils import (  # noqa: E402
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    fp8_checks,
    pack_u16_le,
    run_binary_rows,
    run_col_reduce_tensor,
    run_fp8_pack_rows,
    run_fp8_unpack_rows,
    run_row_reduce_tensor,
    run_unary_rows,
    tensor_checks,
    tensor_preloads,
)


# ---------------------------------------------------------------------------
# Test parameters (must match the assembly DRAM layout)
# ---------------------------------------------------------------------------

TIMEOUT = 800000

TILE_BYTES = 2048                    # 2 KiB = 32x32 BF16
TILE_STRIDE_BYTES = 0x0800

# Inputs: 4 tiles starting at DRAM byte 0.
INPUT_BASES_BYTES = [0x0000, 0x0800, 0x1000, 0x1800]

# Unary outputs: 32 tiles (4 inputs x 8 ops) starting at 0x2000.
UNARY_OUTPUT_BASE_BYTES = 0x2000

# Op order MUST match the assembly. Lowercase keys map to model op names.
OP_SEQUENCE = [
    ("sqrt",   "VSQRT"),
    ("log",    "VLOG2"),
    ("rcp",    "VRECIP.BF16"),
    ("exp",    "VEXP"),
    ("exp2",   "VEXP2"),
    ("tanh",   "VTANH"),
    ("square", "VSQUARE.BF16"),
    ("cube",   "VCUBE.BF16"),
]

# Chain phase outputs (Tile A used as the chain input).
CHAIN_MUL_BASE     = 0x12000   # VMUL.BF16(VLOG2, VSQRT)
CHAIN_ADD_BASE     = 0x12800   # VADD.BF16(VEXP2, VRECIP.BF16)
CHAIN_SUB_BASE     = 0x13000   # VSUB.BF16(VTANH, MUL)
CHAIN_RSUM_BASE    = 0x13800   # VREDSUM.ROW.BF16(ADD)
CHAIN_PACK_BASE    = 0x14000   # VFP8PACK(ADD, scale=PACK_SCALE)
CHAIN_UNPACK_BASE  = 0x14400   # VFP8UNPACK(PACK, scale=UNPACK_SCALE)
CHAIN_CSUM_BASE    = 0x14C00   # VREDSUM.BF16(UNPACK)

PACK_SCALE_E8M0   = 0x7B
UNPACK_SCALE_E8M0 = 0x86


# ---------------------------------------------------------------------------
# BF16 helpers
# ---------------------------------------------------------------------------

def bf16(sign: int, exp: int, mant: int) -> int:
    """Compose a BF16 bit pattern from sign / exp / mantissa fields."""
    return (((sign & 1) << 15) | ((exp & 0xFF) << 7) | (mant & 0x7F)) & 0xFFFF


def _mant_zigzag(r: int) -> int:
    """Return a 7-bit mantissa where consecutive `r`s flip many bits.

    Sequence: 0, 0x7F, 1, 0x7E, 2, 0x7D, ..., 15, 0x70.
    Bit 6 (the high mantissa bit) toggles every row.
    """
    half = r >> 1
    return half if (r & 1) == 0 else (0x7F - half)


# Per-column BF16 exponent "high" base value (LSB will be added below). The
# 32 entries are chosen so the per-column exponent spans roughly 2^-32 to
# 2^32 across the 32 lanes, with widely separated bands so neighbouring
# columns don't collapse to the same LUT exponent input.
_EXP_BAND = [
    0x6E, 0x70, 0x72, 0x76, 0x78, 0x7A, 0x7C, 0x7E,
    0x80, 0x82, 0x84, 0x86, 0x88, 0x8A, 0x8C, 0x8E,
    0x66, 0x68, 0x6A, 0x6C, 0x90, 0x92, 0x94, 0x96,
    0x60, 0x62, 0x64, 0x98, 0x9A, 0x9C, 0x9E, 0xA0,
]


# ---------------------------------------------------------------------------
# Tile constructors (deterministic, must match assembly intent only — the
# assembly does not need to know the data, only the DRAM offsets).
# ---------------------------------------------------------------------------

def tile_a_value(r: int, c: int) -> int:
    """Tile A: all-positive mantissa zig-zag, exp-LSB alternation."""
    mant = (_mant_zigzag(r) ^ ((c * 11) & 0x7F)) & 0x7F
    exp_base = _EXP_BAND[c & 31]
    exp = ((exp_base & 0xFE) | (r & 1)) & 0xFF
    if exp < 1:
        exp = 1
    elif exp > 0xFE:
        exp = 0xFE
    return bf16(0, exp, mant)


def tile_b_value(r: int, c: int) -> int:
    """Tile B: walk + sign alternation (alternates every 2 rows / cols)."""
    mant = (_mant_zigzag(r) ^ ((c * 13 + 5) & 0x7F)) & 0x7F
    exp_base = _EXP_BAND[(c + 7) & 31]
    exp = ((exp_base & 0xFE) | (r & 1)) & 0xFF
    if exp < 1:
        exp = 1
    elif exp > 0xFE:
        exp = 0xFE
    sign = ((r >> 1) ^ (c >> 1)) & 1
    return bf16(sign, exp, mant)


_CLASS_LATTICE = [
    0x0000, 0x8000,   # +/- 0
    0x0001, 0x807F,   # min subnormal +, max subnormal -
    0x0080, 0x8080,   # +/- min normal
    0x3F80, 0xBF80,   # +/- 1.0
    0x4000, 0xC000,   # +/- 2.0
    0x4080, 0xC080,   # +/- 4.0
    0x4100, 0xC100,   # +/- 8.0
    0x4180, 0xC180,   # +/- 16.0
    0x7F7F, 0xFF7F,   # +/- max finite
    0x7F80, 0xFF80,   # +/- inf
    0x7FC1, 0xFFC1,   # qNaN-like
    0x3F00, 0xBF00,   # +/- 0.5
    0x3E80, 0xBE80,   # +/- 0.25
    0x40A0, 0xC0A0,   # +/- 5.0
    0x40C0, 0xC0C0,   # +/- 6.0
    0x40E0, 0xC0E0,   # +/- 7.0
]


def tile_c_value(r: int, c: int) -> int:
    """Tile C: special-class lattice with row/col phase scrambling."""
    idx = (r * 13 + c * 7) % len(_CLASS_LATTICE)
    return _CLASS_LATTICE[idx] & 0xFFFF


def tile_d_value(r: int, c: int) -> int:
    """Tile D: alternating large/small magnitude, complementary mantissas."""
    pat = ((c * 17) ^ (r * 5)) & 0x7F
    if r & 1:
        return bf16(0, 0x88, pat)
    else:
        sign = (c >> 4) & 1
        return bf16(sign, 0x70, (~pat) & 0x7F)


TILE_BUILDERS = [
    ("A", tile_a_value),
    ("B", tile_b_value),
    ("C", tile_c_value),
    ("D", tile_d_value),
]


# ---------------------------------------------------------------------------
# Tile -> bank-split row layout (matches gen_vpu_overlap_lut_pack_reduce).
# ---------------------------------------------------------------------------

def build_tile_rows(builder) -> list[list[int]]:
    """Build a 32x32 tile and return the 64-row bank-split layout used by
    the VPU functional model: rows[0..31] = bank 0 lanes 0..15 by row,
    rows[32..63] = bank 1 lanes 16..31 by row.
    """
    matrix = [[builder(r, c) & 0xFFFF for c in range(32)] for r in range(32)]
    rows: list[list[int]] = []
    for r in range(32):
        rows.append(matrix[r][:BF16_PER_BEAT])
    for r in range(32):
        rows.append(matrix[r][BF16_PER_BEAT:])
    if len(rows) != ROWS_PER_TENSOR:
        raise AssertionError(
            f"expected {ROWS_PER_TENSOR} rows per tile, got {len(rows)}"
        )
    return rows


def split_pair(rows: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    if len(rows) != ROWS_PER_TENSOR:
        raise ValueError(f"expected {ROWS_PER_TENSOR} rows, got {len(rows)}")
    return rows[:ROWS_PER_REGISTER], rows[ROWS_PER_REGISTER:]


def emit_split_tensor_checks(
    checks: list[dict],
    base_lo_bytes: int,
    base_hi_bytes: int,
    rows: list[list[int]],
) -> None:
    lo_rows, hi_rows = split_pair(rows)
    if base_lo_bytes % 32 != 0 or base_hi_bytes % 32 != 0:
        raise ValueError("DRAM byte offsets must be 32-byte aligned")
    checks.extend(tensor_checks(base_lo_bytes // 32, lo_rows))
    checks.extend(tensor_checks(base_hi_bytes // 32, hi_rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    preloads: list[dict] = []
    checks: list[dict] = []

    tile_rows_by_idx: list[list[list[int]]] = []

    for tile_idx, (name, builder) in enumerate(TILE_BUILDERS):
        rows = build_tile_rows(builder)
        tile_rows_by_idx.append(rows)

        in_base_bytes = INPUT_BASES_BYTES[tile_idx]
        if in_base_bytes % 32 != 0:
            raise ValueError("input DRAM base must be 32-byte aligned")
        preloads.extend(tensor_preloads(in_base_bytes // 32, rows))

        for op_idx, (model_op, _asm) in enumerate(OP_SEQUENCE):
            out_rows = run_unary_rows(model_op, rows)
            pass_idx = tile_idx * len(OP_SEQUENCE) + op_idx
            out_base_bytes = (
                UNARY_OUTPUT_BASE_BYTES + pass_idx * TILE_STRIDE_BYTES
            )
            if out_base_bytes % 32 != 0:
                raise ValueError("unary output DRAM base must be 32-byte aligned")
            checks.extend(tensor_checks(out_base_bytes // 32, out_rows))

    # ------------------------------------------------------------------
    # Chain phase — reuses Tile A as the source for all chain unary ops.
    # ------------------------------------------------------------------
    a_rows = tile_rows_by_idx[0]

    log_rows  = run_unary_rows("log",  a_rows)
    sqrt_rows = run_unary_rows("sqrt", a_rows)
    exp2_rows = run_unary_rows("exp2", a_rows)
    rcp_rows  = run_unary_rows("rcp",  a_rows)
    tanh_rows = run_unary_rows("tanh", a_rows)

    mul_rows = run_binary_rows("mul", log_rows,  sqrt_rows)   # m28/m29
    add_rows = run_binary_rows("add", exp2_rows, rcp_rows)    # m30/m31
    sub_rows = run_binary_rows("sub", tanh_rows, mul_rows)    # m32/m33

    rsum_rows = run_row_reduce_tensor("rsum", add_rows)       # m34/m35

    pack_rows = run_fp8_pack_rows(
        add_rows[:ROWS_PER_REGISTER],
        add_rows[ROWS_PER_REGISTER:],
        scale_e8m0=PACK_SCALE_E8M0,
    )                                                         # m36 single mreg
    unpack_rows = run_fp8_unpack_rows(
        pack_rows, scale_e8m0=UNPACK_SCALE_E8M0
    )                                                         # m38/m39

    csum_rows = run_col_reduce_tensor(
        "csum", unpack_rows, ROWS_PER_TENSOR
    )                                                         # m40/m41

    # Chain output checks. Pair-mreg outputs are written to consecutive 1 KiB
    # halves matching VSTORE vd, x8, 0 / VSTORE vd+1, x8, 8 layout.
    def emit_pair_checks(base_bytes: int, rows: list[list[int]]) -> None:
        if base_bytes % 32 != 0:
            raise ValueError("chain DRAM base must be 32-byte aligned")
        checks.extend(tensor_checks(base_bytes // 32, rows))

    emit_pair_checks(CHAIN_MUL_BASE,  mul_rows)
    emit_pair_checks(CHAIN_ADD_BASE,  add_rows)
    emit_pair_checks(CHAIN_SUB_BASE,  sub_rows)
    emit_pair_checks(CHAIN_RSUM_BASE, rsum_rows)

    # FP8 pack output is a single 1 KiB mreg (32 rows of 32 FP8 lanes).
    if CHAIN_PACK_BASE % 32 != 0:
        raise ValueError("pack DRAM base must be 32-byte aligned")
    checks.extend(fp8_checks(CHAIN_PACK_BASE // 32, pack_rows))

    emit_pair_checks(CHAIN_UNPACK_BASE, unpack_rows)
    emit_pair_checks(CHAIN_CSUM_BASE,   csum_rows)

    sys.stderr.write(
        "[gen_vpu_lut_bitflip_megasweep] "
        f"tiles={len(TILE_BUILDERS)} ops_per_tile={len(OP_SEQUENCE)} "
        f"preloads={len(preloads)} checks={len(checks)} "
        f"pack_scale=0x{PACK_SCALE_E8M0:02x} "
        f"unpack_scale=0x{UNPACK_SCALE_E8M0:02x}\n"
    )

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

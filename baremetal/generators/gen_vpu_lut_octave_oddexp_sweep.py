#!/usr/bin/env python3
"""gen_vpu_lut_octave_oddexp_sweep.py — golden generator for
vpu_lut_octave_oddexp_sweep.S.

The matching assembly applies five LUT-backed VPU unary ops (VSQRT,
VLOG2, VRECIP.BF16, VEXP, VTANH) to four 32x32 BF16 tiles that are
constructed at the bit level instead of the float level.

Each tile is designed to drive specific weak signals in the VPU LUT
wrappers:

  Tile 0 (oddExp dense, |x| around 1):
    Walks BF16 biased exponents [122..132], all positive, every
    mantissa.  This makes Sqrt's `oddExp = !aVec[i][7]` flag flip on
    every cycle and dense-walks the 7-bit LUT address bus
    `lutInputMant`.

  Tile 1 (oddExp dense, mixed sign):
    Same exponent ladder as tile 0, but adds the sign bit per lane and
    interleaves through every column. Each lane now sees alternating
    signs across rows, which toggles `neg` (used as the input sign in
    Rcp/Log/Sqrt path) on a per-cycle basis.

  Tile 2 (full octave ladder, both signs):
    Walks ~50 distinct biased exponents from 1 (smallest normal) to 254
    (largest finite), with mixed signs and a moving mantissa. This is
    where the 8-bit `lutInputExp` bus toggles over its full range and
    the Log helper alternates between `isNeg = exp < bias` and
    `isNeg = exp >= bias` paths.

  Tile 3 (special-case + saturating lattice):
    Densely revisits the helper special cases — signed zero, smallest
    and largest signed subnormals, +/-inf, qNaN/sNaN payloads, x ==
    1.0 (Log's `is_zero_lut` branch), and exponents that drive
    expUnsigned >= 255 in Sqrt/Rcp.  Repeating these throughout the
    tile makes every `if` branch in `lutFixedToBf16{Sqrt,Rcp,Log}`
    fire many times.

Outputs are computed by the same VPU functional model used by every
other gen_vpu_*.py script, so the oracle is bit-exact against the RTL.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vpu_gen_utils import (  # noqa: E402
    BF16_PER_BEAT,
    ROWS_PER_TENSOR,
    pack_u16_le,
    run_unary_rows,
)


TIMEOUT = 600000
TILE_BYTES = 2048
TILE_BEATS = TILE_BYTES // 32   # 64 beats per tile

INPUT_TILE_BASES_BYTES = [0x0000, 0x0800, 0x1000, 0x1800]
OUTPUT_BASE_BYTES = 0x4000
OUTPUT_STRIDE_BYTES = 0x0800     # 2 KiB per output pass

# Op order MUST match the assembly: VSQRT, VLOG2, VRECIP.BF16, VEXP, VTANH
OP_SEQUENCE = [
    ("sqrt", "VSQRT"),
    ("log",  "VLOG2"),
    ("rcp",  "VRECIP.BF16"),
    ("exp",  "VEXP"),
    ("tanh", "VTANH"),
]


def bf16(sign: int, exp: int, mant: int) -> int:
    return (((sign & 1) << 15) | ((exp & 0xFF) << 7) | (mant & 0x7F)) & 0xFFFF


# ----------------------------------------------------------------------
# Tile builders -- each returns 1024 BF16 bit patterns in row-major
# 32x32 layout (we pad/truncate to 1024 if necessary).
# ----------------------------------------------------------------------

def _pad_to_1024(values: list[int]) -> list[int]:
    out = [v & 0xFFFF for v in values]
    if not out:
        raise ValueError("tile must be non-empty")
    i = 0
    while len(out) < 1024:
        out.append(out[(i * 41 + 17) % len(values)])
        i += 1
    return out[:1024]


def tile0_oddexp_dense_positive() -> list[int]:
    """Walk biased exponents in [122..132] dense, positive sign only."""
    values: list[int] = []
    for exp in range(122, 133):           # 11 exponents
        for mant in range(128):            # all mantissas
            values.append(bf16(0, exp, mant))
    # 11 * 128 = 1408 -> truncate to 1024 in pad_to_1024.
    return _pad_to_1024(values)


def tile1_oddexp_mixed_sign() -> list[int]:
    """Same exponent walk as tile 0, signs alternate per (row, col)."""
    values: list[int] = []
    for row in range(32):
        for col in range(32):
            # 32 lanes per row -> stride exponents through 11 values
            exp = 122 + ((col + (row >> 1)) % 11)
            mant = (row * 11 + col * 5 + 13) & 0x7F
            sign = ((col + row) & 1) ^ ((row >> 2) & 1)
            values.append(bf16(sign, exp, mant))
    return values


def tile2_full_octave_ladder() -> list[int]:
    """Walk biased exponents 1..254 with mixed signs and rotating mant."""
    values: list[int] = []
    # 254 exponents, sample one mantissa per exponent first (254 values),
    # then a second pass with a different mantissa stride (more variety).
    for pass_idx in range(4):
        for exp in range(1, 255):
            mant_seed = (exp * (3 + pass_idx) + pass_idx * 19 + 7)
            mant = mant_seed & 0x7F
            sign = ((exp + pass_idx) >> 1) & 1
            values.append(bf16(sign, exp, mant))
    return _pad_to_1024(values)


def tile3_special_lattice() -> list[int]:
    """Dense special cases that drive every helper conditional branch.

    The values below cover, per-helper:
      - sqrt: input zero/subnormal/inf/NaN -> signed zero, exp_unsigned
              >= 255 saturation -> +/-inf branch
      - rcp:  zero/subnormal/exp_unsigned>=255 -> signed inf branch,
              inf/NaN -> signed zero branch
      - log:  is_zero_lut (when LUT raw is 0, e.g. input == 1.0) ->
              signed zero, zero/subnormal/inf -> signed inf
      - exp:  underflow/overflow saturation
      - tanh: |x| > 4 saturation, |x| ~ 0
    """
    base = [
        # Signed zero
        0x0000, 0x8000,
        # Smallest/largest subnormals
        0x0001, 0x8001, 0x007F, 0x807F,
        # Smallest normals
        0x0080, 0x8080, 0x00FF, 0x80FF,
        # 1.0 / just-below 1 / just-above 1 (drives Log's is_zero_lut at exact 1.0)
        0x3F80, 0xBF80,            # +/-1.0
        0x3F7F, 0xBF7F,            # just below
        0x3F81, 0xBF81,            # just above
        # 2, 4, 8, 0.5, 0.25
        0x4000, 0xC000, 0x4080, 0xC080, 0x4100, 0xC100,
        0x3F00, 0xBF00, 0x3E80, 0xBE80,
        # Largest finite / +-inf / NaN payloads
        0x7F7F, 0xFF7F,
        0x7F80, 0xFF80,
        0x7FC1, 0xFFC1,
        0x7F81, 0xFF81,
        0x7FFF, 0xFFFF,
    ]
    # Sweep the borderline where Sqrt's `bfExpSigned` saturates.
    # bfExpSigned = bias + ((unbiasedExp + exp - bias) >> 1); for very
    # large exp, exp_unsigned >= 255 triggers the saturation branch.
    for exp in [240, 248, 250, 252, 253, 254]:
        for mant in [0x00, 0x10, 0x20, 0x40, 0x60, 0x7F]:
            base.append(bf16(0, exp, mant))
            base.append(bf16(1, exp, mant))

    # Sweep the borderline where Rcp's expUnsigned >= 255 happens for
    # very small inputs (large 1/x).
    for exp in [1, 2, 4, 6, 8, 10, 14]:
        for mant in [0x00, 0x10, 0x20, 0x40, 0x60, 0x7F]:
            base.append(bf16(0, exp, mant))
            base.append(bf16(1, exp, mant))

    # Tanh saturation lattice (|x| sweeps across the 4.0 cliff).
    for exp in [128, 129, 130, 131, 132, 133, 134, 135, 140, 150]:
        for mant in [0x00, 0x20, 0x40, 0x60, 0x7F]:
            base.append(bf16(0, exp, mant))
            base.append(bf16(1, exp, mant))

    # Exp's overflow ladder (around exp(89) which overflows BF16).
    for exp in [128, 132, 133, 134, 135, 136, 137, 138]:
        for mant in [0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x7F]:
            base.append(bf16(0, exp, mant))
            base.append(bf16(1, exp, mant))

    return _pad_to_1024(base)


TILE_BUILDERS = [
    tile0_oddexp_dense_positive,
    tile1_oddexp_mixed_sign,
    tile2_full_octave_ladder,
    tile3_special_lattice,
]

assert len(TILE_BUILDERS) == len(INPUT_TILE_BASES_BYTES)


# ----------------------------------------------------------------------
# Layout helpers shared with vpu_gen_utils-style generators.
# ----------------------------------------------------------------------

def values_to_rows(values: list[int]) -> list[list[int]]:
    if len(values) != 1024:
        raise ValueError(f"tile must have 1024 BF16 values, got {len(values)}")
    mat = [values[r * 32 : (r + 1) * 32] for r in range(32)]
    rows: list[list[int]] = []
    # Lo half of every matrix row first (32 rows of 16 lanes), then hi half.
    for r in range(32):
        rows.append(mat[r][:BF16_PER_BEAT])
    for r in range(32):
        rows.append(mat[r][BF16_PER_BEAT:])
    if len(rows) != ROWS_PER_TENSOR:
        raise AssertionError(f"expected {ROWS_PER_TENSOR} rows, got {len(rows)}")
    return rows


def rows_to_entries(byte_base: int, rows: list[list[int]], key: str) -> list[dict]:
    if byte_base % 32 != 0:
        raise ValueError(f"byte_base {byte_base} must be 32-byte aligned")
    base_beat = byte_base // 32
    if len(rows) != ROWS_PER_TENSOR:
        raise ValueError(f"expected {ROWS_PER_TENSOR} rows, got {len(rows)}")
    return [
        {"word_offset": base_beat + i, key: pack_u16_le(row)}
        for i, row in enumerate(rows)
    ]


def main() -> None:
    preloads: list[dict] = []
    checks: list[dict] = []

    for tile_idx, builder in enumerate(TILE_BUILDERS):
        rows = values_to_rows(builder())
        preloads.extend(rows_to_entries(INPUT_TILE_BASES_BYTES[tile_idx], rows, "data"))
        for op_idx, (model_op, _asm) in enumerate(OP_SEQUENCE):
            out_rows = run_unary_rows(model_op, rows)
            pass_idx = tile_idx * len(OP_SEQUENCE) + op_idx
            out_base = OUTPUT_BASE_BYTES + pass_idx * OUTPUT_STRIDE_BYTES
            checks.extend(rows_to_entries(out_base, out_rows, "expected"))

    sys.stderr.write(
        f"[gen_vpu_lut_octave_oddexp_sweep] tiles={len(TILE_BUILDERS)} "
        f"ops={len(OP_SEQUENCE)} preloads={len(preloads)} checks={len(checks)}\n"
    )

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

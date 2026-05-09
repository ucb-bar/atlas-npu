#!/usr/bin/env python3
"""gen_vpu_exp_round_edge_blast.py - golden generator for
vpu_exp_round_edge_blast.S.

Drives the 16 `core.vpu.core.exp.roundToRecFn_*` HardFloat
roundAnyRawFNToRecFN instances (cond 49.3%) over their data-dependent
branches by feeding VEXP and VEXP2 four tiles whose outputs land in
distinct rounder regions:

  Tile 0  overflow boundary (exp output near +inf)
  Tile 1  underflow boundary (exp output near +0 / subnormal)
  Tile 2  mid-range mantissa walk (drives LUT interpolation r_lower
          and rounder sticky over the full range)
  Tile 3  tiny inputs (exp output ~= 1.0; barely-inexact rounder path)

Each tile fires VEXP and VEXP2, both routed through the bit-exact
software model `Exp` in `software_models.vpu.lane_boxes.exp`, which
defers to FPEX for `exp` and synthesises a matching `exp2` from the
same RawFloat -> Q(m,n) -> LUT -> HardFloat round-trip.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import float_to_bf16_bits  # noqa: E402
from vpu_gen_utils import (  # noqa: E402
    BF16_PER_BEAT,
    ROWS_PER_TENSOR,
    pack_u16_le,
    run_unary_rows,
)


TIMEOUT = 700000
TILE_BYTES = 2048
TILE_BEATS = TILE_BYTES // 32

INPUT_TILE_BASES_BYTES = [0x0000, 0x0800, 0x1000, 0x1800]
OUTPUT_BASE_BYTES = 0x4000
OUTPUT_STRIDE_BYTES = 0x0800

OP_SEQUENCE = [
    ("exp",  "VEXP"),
    ("exp2", "VEXP2"),
]


def fbits(x: float) -> int:
    return float_to_bf16_bits(x)


def _pad_to_1024(values: list[int]) -> list[int]:
    out = [v & 0xFFFF for v in values]
    if not out:
        raise ValueError("tile must be non-empty")
    i = 0
    while len(out) < 1024:
        out.append(out[(i * 31 + 7) % len(values)])
        i += 1
    return out[:1024]


def tile0_overflow_boundary() -> list[int]:
    """Inputs whose exp() / exp2() results sit on the overflow cliff.

    For natural exp:   exp(x) -> +inf when x > ln(BF16_max) ~= 88.7
    For exp2:          2^x   -> +inf when x > log2(BF16_max) ~= 128

    Walking x densely across both cliffs forces the rounder to flip
    between "overflow detected" and "still finite" branches lane-by-lane.
    Both signs included so the negative-sign overflow path also fires
    (negative VEXP -> 0, negative VEXP2 -> 0; the rounder still cares
    about sign on the final compose).
    """
    values: list[int] = []
    # Overflow ladder for natural exp.
    for k in range(0, 256):
        x = 87.5 + (k / 64.0)        # 87.5..91.5 in 1/64 steps -> 256 vals
        values.append(fbits(x))
    # Overflow ladder for exp2.
    for k in range(0, 256):
        x = 126.0 + (k / 32.0)       # 126.0..134.0
        values.append(fbits(x))
    # Sign mirror on a subset.
    for k in range(0, 256):
        x = -(87.5 + (k / 64.0))
        values.append(fbits(x))
    return _pad_to_1024(values)


def tile1_underflow_boundary() -> list[int]:
    """Inputs whose exp() / exp2() results approach BF16 underflow.

    For natural exp:   exp(x) underflows at x < -94 (roughly).
    For exp2:          2^x   underflows at x < -149.

    Walking x densely past both thresholds drives the rounder's
    "tininess" detection and the subnormal-flush branch.
    """
    values: list[int] = []
    for k in range(0, 192):
        x = -91.0 - (k / 16.0)       # -91 .. ~-103
        values.append(fbits(x))
    for k in range(0, 192):
        x = -120.0 - (k / 4.0)       # -120 .. -168
        values.append(fbits(x))
    # Mirror with positives that produce normal results, to make sure
    # the lane masking and 'not underflow' branch also fires alongside.
    for k in range(0, 192):
        x = 0.5 + (k / 64.0)         # 0.5..3.5 -> exp(x) finite
        values.append(fbits(x))
    return _pad_to_1024(values)


def tile2_mantissa_bit_walk() -> list[int]:
    """Walk every 7-bit BF16 mantissa value across a few exponents,
    both signs.

    Makes the LUT input mantissa cover 0..127 densely while the
    exponent visits {-3, -2, -1, 0, 1, 2, 3}; combined with both
    signs this drives the LUT addr `r >> R_LOW_BITS` over its full
    7-bit range AND the LUT interpolation `r_lower` over its full
    R_LOW_BITS range. The downstream rounder therefore sees a rich
    distribution of inexact / sticky bit combinations.
    """
    def bf16(sign: int, exp: int, mant: int) -> int:
        return (((sign & 1) << 15) | ((exp & 0xFF) << 7) | (mant & 0x7F)) & 0xFFFF
    values: list[int] = []
    for biased_exp in range(124, 131):           # real exp -3..3
        for mant in range(128):
            for sign in (0, 1):
                values.append(bf16(sign, biased_exp, mant))
    return _pad_to_1024(values)


def tile3_tiny_inputs() -> list[int]:
    """Inputs |x| <= 1, ranging down to BF16 subnormals so that exp(x)
    is ~1.0 with tiny inexact bits."""
    values: list[int] = []
    for k in range(1, 31):
        x = 2.0 ** -k
        values.append(fbits(x))
        values.append(fbits(-x))
    # Add a pseudo-uniform sweep of tiny x.
    for k in range(0, 256):
        x = (k - 128) / 1024.0       # -0.125 .. ~+0.125
        values.append(fbits(x))
    # Plus exact 0, +/-1, +/-0.5, +/-2.
    for x in [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
        values.append(fbits(x))
    return _pad_to_1024(values)


TILE_BUILDERS = [
    tile0_overflow_boundary,
    tile1_underflow_boundary,
    tile2_mantissa_bit_walk,
    tile3_tiny_inputs,
]
assert len(TILE_BUILDERS) == len(INPUT_TILE_BASES_BYTES)


def values_to_rows(values: list[int]) -> list[list[int]]:
    if len(values) != 1024:
        raise ValueError(f"tile must have 1024 BF16 values, got {len(values)}")
    mat = [values[r * 32 : (r + 1) * 32] for r in range(32)]
    rows: list[list[int]] = []
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
        f"[gen_vpu_exp_round_edge_blast] tiles={len(TILE_BUILDERS)} "
        f"ops={len(OP_SEQUENCE)} preloads={len(preloads)} checks={len(checks)}\n"
    )

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

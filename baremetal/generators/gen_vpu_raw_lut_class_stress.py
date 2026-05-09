#!/usr/bin/env python3
"""
Golden generator for vpu_raw_lut_class_stress.S.

This test is intentionally bit-pattern driven. The weak VPU leaves are LUT
and class-heavy blocks, so the input tensors walk BF16 exponent/mantissa/sign
fields directly instead of relying on float-domain random values to happen to
quantize into interesting addresses.
"""

from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vpu_gen_utils import (  # noqa: E402
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    float_to_bf16,
    pack_u16_le,
    run_unary_rows,
)


TIMEOUT = 900000
TILE_BYTES = 2048
TILE_STRIDE_BYTES = 0x0800
OUTPUT_BASE_BYTES = 0x3000

OP_SEQUENCE = [
    ("sqrt", "VSQRT"),
    ("log", "VLOG2"),
    ("rcp", "VRECIP.BF16"),
    ("exp", "VEXP"),
    ("exp2", "VEXP2"),
    ("tanh", "VTANH"),
    ("square", "VSQUARE.BF16"),
    ("cube", "VCUBE.BF16"),
]


def bf16(sign: int, exp: int, mant: int) -> int:
    return ((sign & 1) << 15) | ((exp & 0xFF) << 7) | (mant & 0x7F)


def fbits(value: float) -> int:
    return float_to_bf16(value)


def pad_to_tile(values: list[int]) -> list[int]:
    out = [v & 0xFFFF for v in values]
    if not out:
        raise ValueError("tile must have at least one value")
    i = 0
    while len(out) < 32 * 32:
        # Revisit the same classes in a different lane/row phase so lane-local
        # state and row counters do not see only a short repeating prefix.
        out.append(out[(i * 37 + 11) % len(values)])
        i += 1
    return out[: 32 * 32]


def values_to_rows(values: list[int]) -> list[list[int]]:
    vals = pad_to_tile(values)
    mat = [vals[i : i + 32] for i in range(0, len(vals), 32)]
    rows: list[list[int]] = []
    for r in range(32):
        rows.append(mat[r][:BF16_PER_BEAT])
    for r in range(32):
        rows.append(mat[r][BF16_PER_BEAT:])
    if len(rows) != ROWS_PER_TENSOR:
        raise AssertionError(f"expected {ROWS_PER_TENSOR} rows, got {len(rows)}")
    return rows


def full_mantissa_walk() -> list[int]:
    values: list[int] = []
    exps = [1, 32, 96, 126, 127, 128, 160, 254]
    for exp in exps:
        for mant in range(128):
            sign = (mant >> 6) ^ (exp >> 7)
            values.append(bf16(sign, exp, mant))
    return values


def exponent_ladder() -> list[int]:
    values: list[int] = []
    for i in range(1024):
        exp = 1 + (i % 254)
        mant = ((i * 37) ^ (i >> 2) ^ (i << 3)) & 0x7F
        sign = (i >> 5) & 1
        values.append(bf16(sign, exp, mant))
    return values


def special_class_lattice() -> list[int]:
    base = [
        0x0000, 0x8000,       # +/- zero
        0x0001, 0x8001,       # smallest signed subnormal
        0x007F, 0x807F,       # largest signed subnormal
        0x0080, 0x8080,       # min normal
        0x00FF, 0x80FF,
        0x3F7F, 0xBF7F,       # just below +/-1
        0x3F80, 0xBF80,       # +/-1
        0x3F81, 0xBF81,       # just above +/-1
        0x4000, 0xC000,       # +/-2
        0x4080, 0xC080,       # +/-4
        0x4100, 0xC100,       # +/-8
        0x7F7F, 0xFF7F,       # max finite
        0x7F80, 0xFF80,       # +/-inf
        0x7FC1, 0xFFC1,       # qNaN-like payloads
        0x7F81, 0xFF81,       # sNaN-like payloads
    ]

    # Square/cube branch edges: underflow, normal, overflow, signed cube.
    for exp in [1, 2, 32, 63, 64, 65, 84, 85, 86, 126, 127, 128, 169, 170, 171, 212, 213, 214, 253, 254]:
        for mant in [0x00, 0x01, 0x3F, 0x40, 0x7E, 0x7F]:
            base.append(bf16(0, exp, mant))
            base.append(bf16(1, exp, mant))
    return pad_to_tile(base)


def tanh_address_walk() -> list[int]:
    values: list[int] = []
    # Tanh maps |x| in [0,4) into 32 LUT bins, one bin per 0.125.
    for addr in range(32):
        lo = addr / 8.0
        for frac in [0.0, 1.0 / 32.0, 1.0 / 16.0, 3.0 / 32.0]:
            x = lo + frac
            values.append(fbits(x))
            values.append(fbits(-x))
    # Saturating and tiny values exercise the non-LUT branches.
    for x in [0.0, -0.0, 2.0**-20, -(2.0**-20), 3.875, -3.875, 4.0, -4.0, 8.0, -8.0, float("inf"), -float("inf")]:
        values.append(fbits(x))
    values.extend([0x7FC1, 0xFFC1])
    return pad_to_tile(values)


def exp_squarecube_thresholds() -> list[int]:
    values: list[int] = []

    # Exp/exp2 fixed-point and overflow/underflow thresholds.
    for x in [
        -128.0, -100.0, -90.0, -88.75, -64.0, -32.0, -16.0, -8.0, -4.0,
        -2.0, -1.5, -1.0, -0.6931471805599453, -0.5, -0.25, -0.125,
        -0.0, 0.0, 0.125, 0.25, 0.5, 0.6931471805599453, 1.0, 1.5, 2.0,
        4.0, 8.0, 16.0, 32.0, 64.0, 80.0, 88.0, 88.75, 89.0, 100.0,
        128.0, float("inf"), -float("inf"),
    ]:
        values.append(fbits(x))

    # Dense fractional addresses around small positive/negative magnitudes.
    for k in range(-64, 65):
        values.append(fbits(k / 16.0))

    # Square/cube integer-normalization carry edges.
    for exp in [63, 64, 65, 84, 85, 86, 126, 127, 128, 169, 170, 171, 212, 213, 214]:
        for mant in [0x00, 0x01, 0x02, 0x3E, 0x3F, 0x40, 0x41, 0x7D, 0x7E, 0x7F]:
            values.append(bf16(0, exp, mant))
            values.append(bf16(1, exp, mant))

    values.extend([0x0001, 0x8001, 0x007F, 0x807F, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC1, 0xFFC1])
    return pad_to_tile(values)


TILE_BUILDERS = [
    full_mantissa_walk,
    exponent_ladder,
    special_class_lattice,
    tanh_address_walk,
    exp_squarecube_thresholds,
]


def rows_to_entries(base_byte_off: int, rows: list[list[int]], key: str) -> list[dict]:
    if base_byte_off % 32 != 0:
        raise ValueError("DRAM byte offset must be 32-byte aligned")
    base_beat = base_byte_off // 32
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
        preloads.extend(rows_to_entries(tile_idx * TILE_BYTES, rows, "data"))

        for op_idx, (model_op, _asm) in enumerate(OP_SEQUENCE):
            out_rows = run_unary_rows(model_op, rows)
            pass_idx = tile_idx * len(OP_SEQUENCE) + op_idx
            out_base = OUTPUT_BASE_BYTES + pass_idx * TILE_STRIDE_BYTES
            checks.extend(rows_to_entries(out_base, out_rows, "expected"))

    sys.stderr.write(
        f"[gen_vpu_raw_lut_class_stress] tiles={len(TILE_BUILDERS)} "
        f"ops={len(OP_SEQUENCE)} preloads={len(preloads)} checks={len(checks)}\n"
    )
    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

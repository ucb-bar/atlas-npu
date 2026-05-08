#!/usr/bin/env python3
"""gen_vpu_sqcb_branch_blast.py - golden generator for vpu_sqcb_branch_blast.S.

Targets `core.vpu.core.sqcb` branch 23.30% / cond 42.86% by feeding the
SquareCubeVec lane box four 32x32 BF16 tiles, each carefully built to
*force* a different family of conditional arms in
`VectorParam.squareCube` (`VectorParam.scala:6-96`).

Tiles
-----
  Tile 0 (DRAM 0x0000): mid-range biased exponents [120..134] paired
                        with every mantissa, both signs. Drives the
                        normal arms and toggles the full mantissa
                        bit-walk in `squareMant`/`cubeMant`.

  Tile 1 (DRAM 0x0800): tiny magnitudes (biased exp 1..63), both
                        signs, dense mantissa walk.
                        For square: real_exp <= -64 -> squareExp <= -128
                        -> adjustedSquareExp <= 0 - extraExp + 127 < 0
                        -> isSquareResultZero arm fires.
                        For cube:   real_exp <= -64 -> cubeExp <= -192
                        -> adjustedCubeExp <= 0 - extraExp + 127 < 0
                        -> isCubeResultZero arm fires.

  Tile 2 (DRAM 0x1000): huge magnitudes (biased exp 192..254), both
                        signs, dense mantissa walk.
                        For square: real_exp >= 65 -> squareExp >= 130
                        -> adjustedSquareExp >= 257 -> isSquareResultInf.
                        For cube:   real_exp >= 65 -> cubeExp >= 195
                        -> adjustedCubeExp >= 322 -> isCubeResultInf.

  Tile 3 (DRAM 0x1800): special-class lattice. Densely revisits +/-zero,
                        smallest/largest signed subnormals, +/-inf, qNaN
                        and sNaN payloads, exact +/-1.0, and the borderline
                        exponents where adjustedSquareExp/adjustedCubeExp
                        flip from 0->1 and 254->255.

Both VSQUARE.BF16 and VCUBE.BF16 fire on every tile (8 passes).

Goldens come from the shared VPU functional model via run_unary_rows;
the helper is bit-exact against the RTL.
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


TIMEOUT = 700000
TILE_BYTES = 2048
TILE_BEATS = TILE_BYTES // 32

INPUT_TILE_BASES_BYTES = [0x0000, 0x0800, 0x1000, 0x1800]
OUTPUT_BASE_BYTES = 0x4000
OUTPUT_STRIDE_BYTES = 0x0800

OP_SEQUENCE = [
    ("square", "VSQUARE.BF16"),
    ("cube",   "VCUBE.BF16"),
]


def bf16(sign: int, exp: int, mant: int) -> int:
    return (((sign & 1) << 15) | ((exp & 0xFF) << 7) | (mant & 0x7F)) & 0xFFFF


def _pad_to_1024(values: list[int]) -> list[int]:
    out = [v & 0xFFFF for v in values]
    if not out:
        raise ValueError("tile must be non-empty")
    i = 0
    while len(out) < 1024:
        out.append(out[(i * 41 + 17) % len(values)])
        i += 1
    return out[:1024]


def tile0_normal_dense_walk() -> list[int]:
    """Mid-range biased exp [120..134] x every mantissa x both signs.

    Hits squareMant bits 14..15 and cubeMant bits 21..23 dense, drives
    the normal `.otherwise` arm of both result chains repeatedly while
    walking the for-loop mantissa bit indices.
    """
    values: list[int] = []
    for exp in range(120, 135):           # 15 exponents around the bias
        for mant in range(128):
            sign = (mant >> 6) & 1
            values.append(bf16(sign, exp, mant))
    return _pad_to_1024(values)


def tile1_tiny_magnitudes() -> list[int]:
    """Small biased exponents drive isSquareResultZero AND
    isCubeResultZero.

    For square: real_exp <= -64 (biased exp <= 63) -> adjustedSquareExp <= 0.
    For cube:   real_exp <= -43 (biased exp <= 84) -> adjustedCubeExp <= 0.
    Choosing exp in [1, 50] guarantees BOTH paths fire on every lane.
    """
    values: list[int] = []
    for exp in range(1, 64):              # 63 distinct exponents
        for mant in [0x00, 0x10, 0x20, 0x30, 0x40, 0x55, 0x60, 0x70, 0x7F,
                     0x01, 0x02, 0x04, 0x08, 0x40, 0x7E, 0x55]:
            for sign in (0, 1):
                values.append(bf16(sign, exp, mant))
    return _pad_to_1024(values)


def tile2_huge_magnitudes() -> list[int]:
    """Large biased exponents drive isSquareResultInf AND isCubeResultInf.

    For square: real_exp >= 64 (biased exp >= 191) -> adjustedSquareExp >= 255.
    For cube:   real_exp >= 43 (biased exp >= 170) -> adjustedCubeExp >= 255.
    Choosing exp in [192, 254] guarantees both paths fire on every lane.
    """
    values: list[int] = []
    for exp in range(192, 255):           # 63 distinct exponents (avoid 255 = inf)
        for mant in [0x00, 0x10, 0x20, 0x30, 0x40, 0x55, 0x60, 0x70, 0x7F,
                     0x01, 0x02, 0x04, 0x08, 0x40, 0x7E, 0x55]:
            for sign in (0, 1):
                values.append(bf16(sign, exp, mant))
    return _pad_to_1024(values)


def tile3_special_lattice() -> list[int]:
    """Dense special-class lattice + threshold-borderline mantissas.

    Hits isInputZero, isInputSubnormal, isInputNaN, isInputInf arms.
    Also includes exponents right at the boundary where the result
    flips from normal -> result-zero or normal -> result-inf for either
    square or cube, with diverse mantissas that drive `extraExp` flag
    flips inside `cubeExtraExp` (3-arm chain) and `squareExtraExp`
    (2-arm chain).
    """
    base: list[int] = []

    # Signed zeros -> isInputZero
    base += [0x0000, 0x8000]

    # Signed subnormals (every fra value once with each sign)
    for fra in range(1, 128):
        base.append(bf16(0, 0, fra))
        base.append(bf16(1, 0, fra))

    # Signed infinity -> isInputInf
    base += [0x7F80, 0xFF80]

    # NaN payloads (qNaN and sNaN-like, several payloads, both signs)
    for fra in [0x01, 0x02, 0x10, 0x40, 0x41, 0x55, 0x7E, 0x7F]:
        base.append(bf16(0, 0xFF, fra))
        base.append(bf16(1, 0xFF, fra))

    # Square-result borderline (biased exp around 63/64, both edges)
    for exp in [62, 63, 64, 65, 66]:
        for mant in [0x00, 0x40, 0x55, 0x7E, 0x7F]:
            for sign in (0, 1):
                base.append(bf16(sign, exp, mant))

    # Cube-result borderline (biased exp around 84/85, both edges)
    for exp in [82, 83, 84, 85, 86, 87, 88]:
        for mant in [0x00, 0x40, 0x55, 0x7E, 0x7F]:
            for sign in (0, 1):
                base.append(bf16(sign, exp, mant))

    # Square-overflow borderline (biased exp around 191)
    for exp in [189, 190, 191, 192, 193]:
        for mant in [0x00, 0x40, 0x55, 0x7E, 0x7F]:
            for sign in (0, 1):
                base.append(bf16(sign, exp, mant))

    # Cube-overflow borderline (biased exp around 170)
    for exp in [167, 168, 169, 170, 171, 172, 173]:
        for mant in [0x00, 0x40, 0x55, 0x7E, 0x7F]:
            for sign in (0, 1):
                base.append(bf16(sign, exp, mant))

    # Mantissa-bit-walk drivers (every (mant, fra) that flips a unique
    # bit position in mant^2 and mant^3): feed every mantissa once at a
    # safe normal exponent so the normal arm fires while the for-loop
    # bit walks happen.
    for fra in range(128):
        base.append(bf16(0, 127, fra))
        base.append(bf16(1, 127, fra))

    # extraExp threshold drivers:
    #   square: bit15 of mant^2 flips between mant=181 and mant=182
    for fra in [52, 53, 54, 55, 56, 57]:
        base.append(bf16(0, 127, fra))
    #   cube:   bit22 flips around mant=162, bit23 flips around mant=204
    for fra in [33, 34, 35, 36, 37, 38, 75, 76, 77, 78, 79, 80]:
        base.append(bf16(0, 127, fra))

    # Exact +/-1.0 (cube preserves sign through normal path)
    base += [0x3F80, 0xBF80]

    return _pad_to_1024(base)


TILE_BUILDERS = [
    tile0_normal_dense_walk,
    tile1_tiny_magnitudes,
    tile2_huge_magnitudes,
    tile3_special_lattice,
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
        f"[gen_vpu_sqcb_branch_blast] tiles={len(TILE_BUILDERS)} "
        f"ops={len(OP_SEQUENCE)} preloads={len(preloads)} checks={len(checks)}\n"
    )

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

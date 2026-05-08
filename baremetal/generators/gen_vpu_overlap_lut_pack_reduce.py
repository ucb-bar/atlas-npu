#!/usr/bin/env python3
"""Golden generator for vpu_overlap_lut_pack_reduce.S.

The assembly deliberately overlaps independent VPU single-input lane boxes,
then pushes their outputs through binary arithmetic, reductions, and FP8
format conversion. This generator mirrors that architectural dataflow exactly.
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
    float_to_bf16,
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


TIMEOUT = 800000

A_BASE = 0x0000 // 32
B_BASE = 0x0800 // 32
PACK_BASE = 0x1000 // 32
UNPACK_LO_BASE = 0x1400 // 32
UNPACK_HI_BASE = 0x1800 // 32
ROW_LO_BASE = 0x1C00 // 32
ROW_HI_BASE = 0x2000 // 32
COL_LO_BASE = 0x2400 // 32
COL_HI_BASE = 0x2800 // 32

PACK_SCALE_E8M0 = 0x7B
UNPACK_SCALE_E8M0 = 0x86


def bf16(sign: int, exp: int, mant: int) -> int:
    return ((sign & 1) << 15) | ((exp & 0xFF) << 7) | (mant & 0x7F)


def rows_from_values(values: list[int]) -> list[list[int]]:
    """Convert 1024 BF16 values into bank-split rows matching two mregs."""
    if len(values) != 32 * 32:
        raise ValueError(f"expected 1024 values, got {len(values)}")
    matrix = [values[i : i + 32] for i in range(0, len(values), 32)]
    rows = []
    for r in range(32):
        rows.append(matrix[r][:BF16_PER_BEAT])
    for r in range(32):
        rows.append(matrix[r][BF16_PER_BEAT:])
    return rows


def positive_log_stimulus() -> list[list[int]]:
    """Positive BF16s that walk log/exp/reciprocal LUT address classes."""
    values: list[int] = []
    exps = [1, 2, 8, 32, 64, 96, 112, 120, 126, 127, 128, 129, 136, 160, 192, 224, 253, 254]
    mants = [0x00, 0x01, 0x02, 0x07, 0x0F, 0x10, 0x1F, 0x20,
             0x3E, 0x3F, 0x40, 0x41, 0x5F, 0x60, 0x7E, 0x7F]
    for row in range(32):
        for col in range(32):
            idx = row * 32 + col
            exp = exps[(idx + 3 * row + col) % len(exps)]
            mant = mants[(idx * 5 + row * 7 + col * 11) % len(mants)]
            values.append(bf16(0, exp, mant))

    # Make the first rows hit special log/reciprocal classes without dominating.
    specials = [
        0x0000, 0x0001, 0x007F, 0x0080,
        0x3E80, 0x3F00, 0x3F7F, 0x3F80,
        0x3F81, 0x4000, 0x4040, 0x4080,
        0x7F7F, 0x7F80, 0x7FC1, 0x7F81,
    ]
    values[: len(specials)] = specials
    return rows_from_values(values)


def sqrt_tanh_stimulus() -> list[list[int]]:
    """Non-negative values chosen to vary sqrt, tanh, and sin lanes."""
    floats = [
        0.0, 2.0**-20, 2.0**-14, 2.0**-10, 2.0**-6,
        1.0 / 16.0, 1.0 / 8.0, 3.0 / 16.0, 0.25, 0.375,
        0.5, 0.75, 0.875, 1.0, 1.125, 1.5,
        1.875, 2.0, 3.0, 4.0, 6.0, 8.0,
        12.0, 16.0, 24.0, 32.0, 48.0, 64.0,
        96.0, 128.0, 192.0, 256.0,
    ]
    values: list[int] = []
    for row in range(32):
        for col in range(32):
            base = floats[(row * 9 + col * 5) % len(floats)]
            # Alternate exact BF16 field walks with float-derived thresholds.
            if (row + col) % 5 == 0:
                exp = [1, 32, 63, 64, 65, 84, 85, 86, 126, 127, 128, 169, 170, 171, 212, 253][
                    (row + col) % 16
                ]
                mant = (row * 13 + col * 17) & 0x7F
                values.append(bf16(0, exp, mant))
            else:
                values.append(float_to_bf16(base))

    specials = [0x0000, 0x0001, 0x007F, 0x0080, 0x3F80, 0x4000, 0x4080, 0x7F7F]
    values[32: 32 + len(specials)] = specials
    return rows_from_values(values)


def split_pair(rows: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    if len(rows) != ROWS_PER_TENSOR:
        raise ValueError(f"expected one BF16 tensor pair, got {len(rows)} rows")
    return rows[:ROWS_PER_REGISTER], rows[ROWS_PER_REGISTER:]


def emit_split_tensor_checks(checks: list[dict], base_lo: int, base_hi: int, rows: list[list[int]]) -> None:
    lo_rows, hi_rows = split_pair(rows)
    checks.extend(tensor_checks(base_lo, lo_rows))
    checks.extend(tensor_checks(base_hi, hi_rows))


def main() -> None:
    a_rows = positive_log_stimulus()
    b_rows = sqrt_tanh_stimulus()

    # Mirrors assembly:
    #   log2(A) || sqrt(B)
    #   exp2(log2(A)) || tanh(sqrt(B))
    #   recip(exp2(...)) || sin(tanh(...))
    log_rows = run_unary_rows("log", a_rows)
    sqrt_rows = run_unary_rows("sqrt", b_rows)
    exp2_rows = run_unary_rows("exp2", log_rows)
    tanh_rows = run_unary_rows("tanh", sqrt_rows)
    recip_rows = run_unary_rows("rcp", exp2_rows)
    sin_rows = run_unary_rows("sin", tanh_rows)
    cos_rows = run_unary_rows("cos", exp2_rows)

    mul_rows = run_binary_rows("mul", recip_rows, sin_rows)
    mixed_rows = run_binary_rows("add", mul_rows, cos_rows)
    row_sum_rows = run_row_reduce_tensor("rsum", mixed_rows)
    packed_rows = run_fp8_pack_rows(
        mixed_rows[:ROWS_PER_REGISTER],
        mixed_rows[ROWS_PER_REGISTER:],
        scale_e8m0=PACK_SCALE_E8M0,
    )
    unpacked_rows = run_fp8_unpack_rows(packed_rows, scale_e8m0=UNPACK_SCALE_E8M0)
    col_sum_rows = run_col_reduce_tensor("csum", unpacked_rows, ROWS_PER_TENSOR)

    preloads = []
    preloads.extend(tensor_preloads(A_BASE, a_rows))
    preloads.extend(tensor_preloads(B_BASE, b_rows))

    checks = []
    checks.extend(fp8_checks(PACK_BASE, packed_rows))
    emit_split_tensor_checks(checks, UNPACK_LO_BASE, UNPACK_HI_BASE, unpacked_rows)
    emit_split_tensor_checks(checks, ROW_LO_BASE, ROW_HI_BASE, row_sum_rows)
    emit_split_tensor_checks(checks, COL_LO_BASE, COL_HI_BASE, col_sum_rows)

    sys.stderr.write(
        "[gen_vpu_overlap_lut_pack_reduce] "
        f"preloads={len(preloads)} checks={len(checks)} "
        f"pack_scale=0x{PACK_SCALE_E8M0:02x} unpack_scale=0x{UNPACK_SCALE_E8M0:02x}\n"
    )
    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()

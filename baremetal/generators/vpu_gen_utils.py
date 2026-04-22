#!/usr/bin/env python3
"""Shared helpers for the baremetal `gen_vpu_*` scripts.

The VPU software functional model is the source of truth for expected
results; these helpers keep the generators thin while preserving the
existing DRAM layouts used by the assembly tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from software_models.vpu.bf16_utils import f32_to_bf16_bits_rne
from software_models.vpu.vector_engine_model import VectorEngineModel
from software_models.vpu.vector_params import VectorParams


PARAMS = VectorParams()
MODEL = VectorEngineModel(PARAMS)

BF16_PER_BEAT = PARAMS.num_lanes
FP8_PER_BEAT = 2 * PARAMS.num_lanes
ROWS_PER_REGISTER = PARAMS.rows_per_register
ROWS_PER_TENSOR = 2 * ROWS_PER_REGISTER


def float_to_bf16(x: float) -> int:
    return f32_to_bf16_bits_rne(float(x)) & 0xFFFF


def pack_u16_le(vals: Iterable[int]) -> str:
    vals = list(vals)
    if len(vals) != BF16_PER_BEAT:
        raise ValueError(
            f"expected {BF16_PER_BEAT} BF16 lanes per beat, got {len(vals)}"
        )
    word = 0
    for i, v in enumerate(vals):
        word |= (int(v) & 0xFFFF) << (16 * i)
    return f"0x{word:064x}"


def pack_u8_le(vals: Iterable[int]) -> str:
    vals = list(vals)
    if len(vals) != FP8_PER_BEAT:
        raise ValueError(
            f"expected {FP8_PER_BEAT} FP8 lanes per beat, got {len(vals)}"
        )
    word = 0
    for i, v in enumerate(vals):
        word |= (int(v) & 0xFF) << (8 * i)
    return f"0x{word:064x}"


def const_bf16_row(bf16_val: int) -> list[int]:
    return [bf16_val & 0xFFFF] * BF16_PER_BEAT


def constant_bf16_rows(num_rows: int, bf16_val: int) -> list[list[int]]:
    row = const_bf16_row(bf16_val)
    return [list(row) for _ in range(num_rows)]


def repeat_bf16_row(row: Sequence[int], num_rows: int) -> list[list[int]]:
    if len(row) != BF16_PER_BEAT:
        raise ValueError(f"row must have {BF16_PER_BEAT} BF16 lanes")
    row_copy = [int(v) & 0xFFFF for v in row]
    return [list(row_copy) for _ in range(num_rows)]


def tensor_preloads(base_word_offset: int, rows: Sequence[Sequence[int]]) -> list[dict]:
    return [
        {"word_offset": base_word_offset + row_idx, "data": pack_u16_le(row)}
        for row_idx, row in enumerate(rows)
    ]


def tensor_checks(base_word_offset: int, rows: Sequence[Sequence[int]]) -> list[dict]:
    return [
        {"word_offset": base_word_offset + row_idx, "expected": pack_u16_le(row)}
        for row_idx, row in enumerate(rows)
    ]


def fp8_checks(base_word_offset: int, rows: Sequence[Sequence[int]]) -> list[dict]:
    checks = []
    for row_idx, row in enumerate(rows):
        if len(row) == BF16_PER_BEAT:
            packed = pack_u16_le(row)
        else:
            packed = pack_u8_le(row)
        checks.append({"word_offset": base_word_offset + row_idx, "expected": packed})
    return checks


def run_unary_rows(op: str, rows: Sequence[Sequence[int]]) -> list[list[int]]:
    return [list(MODEL.execute(op, a_vec=list(row))) for row in rows]


def run_row_reduce_tensor(op: str, rows: Sequence[Sequence[int]]) -> list[list[int]]:
    """Row-reduce `rows` treated as a concatenation of bank-split tile pairs.

    Each tile pair is `ROWS_PER_TENSOR` rows: the first `ROWS_PER_REGISTER`
    are bank 0 and the next `ROWS_PER_REGISTER` are bank 1. The reduction is
    *local* to each tile pair — no cross-tile semantics — so arbitrary
    multiples of `ROWS_PER_TENSOR` are supported by reducing each chunk
    independently and concatenating the results.

    For a single 64-row input this reproduces the original single-tile
    contract exactly (back-compat for stock smolvla_reduction_sum).
    """
    if len(rows) == 0 or len(rows) % ROWS_PER_TENSOR != 0:
        raise ValueError(
            f"{op}: expected a positive multiple of {ROWS_PER_TENSOR} rows "
            f"(one full bank-split tile pair per chunk), got {len(rows)}"
        )
    out_all: list[list[int]] = []
    for chunk_start in range(0, len(rows), ROWS_PER_TENSOR):
        chunk = rows[chunk_start:chunk_start + ROWS_PER_TENSOR]
        chunk_out: list[list[int]] = []
        for row_idx in range(ROWS_PER_REGISTER):
            reduced = list(
                MODEL.execute(
                    op,
                    a_vec=list(chunk[row_idx]),
                    b_vec=list(chunk[row_idx + ROWS_PER_REGISTER]),
                )
            )
            chunk_out.append(list(reduced))
        # VREDSUM.ROW.BF16 broadcasts the scalar back to both banks → H0 == H1.
        out_all.extend(chunk_out)
        out_all.extend([list(r) for r in chunk_out])
    return out_all


def run_binary_rows(
    op: str,
    a_rows: Sequence[Sequence[int]],
    b_rows: Sequence[Sequence[int]],
) -> list[list[int]]:
    if len(a_rows) != len(b_rows):
        raise ValueError(
            f"binary op {op} needs matching row counts, got {len(a_rows)} and "
            f"{len(b_rows)}"
        )
    return [
        list(MODEL.execute(op, a_vec=list(a_row), b_vec=list(b_row)))
        for a_row, b_row in zip(a_rows, b_rows)
    ]


def run_col_reduce_tensor(
    op: str,
    rows: Sequence[Sequence[int]],
    out_rows: int | None = None,
) -> list[list[int]]:
    reduced_row = list(MODEL.stream_col_reduce(op, [list(row) for row in rows]))
    return repeat_bf16_row(reduced_row, len(rows) if out_rows is None else out_rows)


def run_fp8_pack_rows(
    low_rows: Sequence[Sequence[int]],
    high_rows: Sequence[Sequence[int]],
    scale_e8m0: int,
) -> list[list[int]]:
    """Mirror the current VPU `fp8pack` register-level contract.

    The hardware streams the full first BF16 mreg (rows 0..31) and then the
    full second BF16 mreg (rows 0..31) through the 2→1 phased pack lane box.
    That means packed output row `i` is formed from source stream rows
    `2*i` and `2*i+1`, not from `low_rows[i]` paired with `high_rows[i]`.
    """
    if len(low_rows) != len(high_rows):
        raise ValueError(
            f"fp8pack needs matching row counts, got {len(low_rows)} and "
            f"{len(high_rows)}"
        )
    stream_rows = [list(row) for row in low_rows] + [list(row) for row in high_rows]
    if len(stream_rows) % 2 != 0:
        raise ValueError(
            f"fp8pack needs an even total row count, got {len(stream_rows)}"
        )
    return [
        list(
            MODEL.execute(
                "fp8pack",
                a_vec=stream_rows[row_idx],
                b_vec=stream_rows[row_idx + 1],
                scale_e8m0=scale_e8m0,
            )
        )
        for row_idx in range(0, len(stream_rows), 2)
    ]


def run_fp8_unpack_rows(
    packed_rows: Sequence[Sequence[int]],
    scale_e8m0: int,
) -> list[list[int]]:
    """Mirror the current VPU `fp8unpack` register-level contract.

    The hardware expands one packed row into two BF16 rows and writes them as a
    single stream: low0, high0, low1, high1, ... . It does NOT write all low
    rows first and all high rows second.
    """
    out_rows: list[list[int]] = []
    for packed_row in packed_rows:
        unpacked = list(
            MODEL.execute("fp8unpack", a_vec=list(packed_row), scale_e8m0=scale_e8m0)
        )
        out_rows.append(unpacked[:BF16_PER_BEAT])
        out_rows.append(unpacked[BF16_PER_BEAT:])
    return out_rows


def run_vli_rows(op: str, imm: int, num_rows: int = ROWS_PER_REGISTER) -> list[list[int]]:
    return [
        list(MODEL.execute(op, imm=imm, row_idx=row_idx))
        for row_idx in range(num_rows)
    ]


def run_vli_registers(
    op: str,
    imm: int,
    dst_bank: int = 0,
) -> dict[int, list[list[int]]]:
    return {
        bank: [list(row) for row in rows]
        for bank, rows in MODEL.execute_vli_registers(op, imm=imm, dst_bank=dst_bank).items()
    }

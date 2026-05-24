#!/usr/bin/env python3
"""Shared pairwise VPU-overlap case generation.

These suites exhaustively cover every ordered pair of VPU ops that the
current FSM can accept back-to-back:

  - only single-input ops are eligible for overlap
  - same-op and shared-logic families are excluded
  - double-port ops never overlap in the current FSM implementation

The same case table drives both:
  - golden JSON generation (default stdout path used by run_asm_tests.sh)
  - assembly emission (`--emit-asm`) for the checked-in baremetal tests
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from vpu_gen_utils import (
    ROWS_PER_REGISTER,
    constant_bf16_rows,
    float_to_bf16,
    run_col_reduce_tensor,
    run_fp8_pack_rows,
    run_fp8_unpack_rows,
    run_unary_rows,
    run_vli_registers,
    tensor_checks,
    tensor_preloads,
)


ROWS_PER_BANK = ROWS_PER_REGISTER
ROWS_PER_TENSOR = 2 * ROWS_PER_BANK
SIGNATURE_ROWS = 2
BF16_ZERO = float_to_bf16(0.0)

INPUT_BANK_BYTES = 1024
SIGNATURE_COPY_BYTES = SIGNATURE_ROWS * 32
CASE_SIGNATURE_BYTES = 4 * SIGNATURE_COPY_BYTES

INPUT_COPY_SIZE_WORDS = INPUT_BANK_BYTES
SIGNATURE_COPY_SIZE_WORDS = SIGNATURE_COPY_BYTES

SRC_SAFE1_BASE = 0
SRC_SIGNED1_BASE = 64
SRC_SAFE2_BASE = 128
SRC_SIGNED2_BASE = 192
SRC_PACKED1_BASE = 256
SRC_PACKED2_BASE = 288
OUTPUT_BASE = 320

SLOT1_PAIR_BASE = 16
SLOT2_PAIR_BASE = 18
SLOT1_ODD_BANK = SLOT1_PAIR_BASE + 1
SLOT2_ODD_BANK = SLOT2_PAIR_BASE + 1

SCRATCH_SLOT1_LOW = 1024
SCRATCH_SLOT1_HIGH = 1280
SCRATCH_SLOT2_LOW = 1536
SCRATCH_SLOT2_HIGH = 1792

INIT_DELAY = 150
CASE_DELAY = 300
VSTORE_DELAY = 40

PACK_SCALE_SLOT1 = 0x80
UNPACK_SCALE_SLOT1 = 0x81
PACK_SCALE_SLOT2 = 0x82
UNPACK_SCALE_SLOT2 = 0x7F

PACK_SCALE_REG_SLOT1 = 0
UNPACK_SCALE_REG_SLOT1 = 1
PACK_SCALE_REG_SLOT2 = 2
UNPACK_SCALE_REG_SLOT2 = 3

TIMEOUT = 600000


def _bf16_row_from_floats(vals: list[float]) -> list[int]:
    return [float_to_bf16(v) for v in vals]


def _make_safe_rows(phase: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for row_idx in range(ROWS_PER_TENSOR):
        row = []
        for lane_idx in range(16):
            val = 0.5 + 0.125 * ((row_idx + lane_idx + phase) % 12)
            row.append(float_to_bf16(val))
        rows.append(row)
    return rows


def _make_signed_rows(phase: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for row_idx in range(ROWS_PER_TENSOR):
        row = []
        for lane_idx in range(16):
            step = (3 * row_idx + lane_idx + phase) % 8
            val = -1.5 + 0.5 * step
            row.append(float_to_bf16(val))
        rows.append(row)
    return rows


SAFE_ROWS_SLOT1 = _make_safe_rows(0)
SIGNED_ROWS_SLOT1 = _make_signed_rows(1)
SAFE_ROWS_SLOT2 = _make_safe_rows(5)
SIGNED_ROWS_SLOT2 = _make_signed_rows(3)

PACKED_ROWS_SLOT1 = run_fp8_pack_rows(
    SAFE_ROWS_SLOT1[:ROWS_PER_BANK],
    SAFE_ROWS_SLOT1[ROWS_PER_BANK:],
    scale_e8m0=PACK_SCALE_SLOT1,
)
PACKED_ROWS_SLOT2 = run_fp8_pack_rows(
    SAFE_ROWS_SLOT2[:ROWS_PER_BANK],
    SAFE_ROWS_SLOT2[ROWS_PER_BANK:],
    scale_e8m0=PACK_SCALE_SLOT2,
)

ZERO_BANK_ROWS = constant_bf16_rows(ROWS_PER_BANK, BF16_ZERO)

VLI_IMMS = {
    "vliAll": float_to_bf16(1.25),
    "vliRow": float_to_bf16(2.5),
    "vliCol": float_to_bf16(3.75),
    "vliOne": float_to_bf16(4.5),
}


@dataclass(frozen=True)
class OpSpec:
    name: str
    mnemonic: str
    kind: str
    shared_group: str
    source_family: str
    init_required: bool
    writes_full_pair: bool

    def _pair_base(self, slot: int) -> int:
        return SLOT1_PAIR_BASE if slot == 1 else SLOT2_PAIR_BASE

    def _odd_bank(self, slot: int) -> int:
        return SLOT1_ODD_BANK if slot == 1 else SLOT2_ODD_BANK

    def _pack_scale_reg(self, slot: int) -> int:
        return PACK_SCALE_REG_SLOT1 if slot == 1 else PACK_SCALE_REG_SLOT2

    def _unpack_scale_reg(self, slot: int) -> int:
        return UNPACK_SCALE_REG_SLOT1 if slot == 1 else UNPACK_SCALE_REG_SLOT2

    def _imm(self) -> int:
        return VLI_IMMS[self.name]

    def asm(self, slot: int) -> str:
        pair_base = self._pair_base(slot)
        if self.kind == "unary":
            src_bank = 0 if slot == 1 else 4
            return f"{self.mnemonic} {pair_base}, {src_bank}"
        if self.kind == "signed_reduce":
            src_bank = 2 if slot == 1 else 6
            return f"{self.mnemonic} {pair_base}, {src_bank}"
        if self.kind == "fp8pack":
            src_bank = 0 if slot == 1 else 4
            return f"{self.mnemonic} {pair_base}, {src_bank}, {self._pack_scale_reg(slot)}"
        if self.kind == "fp8unpack":
            src_bank = 8 if slot == 1 else 9
            return f"{self.mnemonic} {pair_base}, {src_bank}, {self._unpack_scale_reg(slot)}"
        if self.kind == "vli_pair":
            return f"{self.mnemonic} {pair_base}, 0x{self._imm():04X}"
        if self.kind == "vli_single":
            return f"{self.mnemonic} {self._odd_bank(slot)}, 0x{self._imm():04X}"
        raise ValueError(f"unknown asm kind {self.kind!r}")

    def result_rows(self, slot: int) -> tuple[list[list[int]], list[list[int]]]:
        pair_base = self._pair_base(slot)

        if self.kind == "unary":
            src_rows = SAFE_ROWS_SLOT1 if slot == 1 else SAFE_ROWS_SLOT2
            out_rows = run_unary_rows(self.name, src_rows)
            return out_rows[:ROWS_PER_BANK], out_rows[ROWS_PER_BANK:]

        if self.kind == "signed_reduce":
            src_rows = SIGNED_ROWS_SLOT1 if slot == 1 else SIGNED_ROWS_SLOT2
            out_rows = run_col_reduce_tensor(self.name, src_rows, ROWS_PER_TENSOR)
            return out_rows[:ROWS_PER_BANK], out_rows[ROWS_PER_BANK:]

        if self.kind == "fp8pack":
            src_rows = SAFE_ROWS_SLOT1 if slot == 1 else SAFE_ROWS_SLOT2
            packed_rows = run_fp8_pack_rows(
                src_rows[:ROWS_PER_BANK],
                src_rows[ROWS_PER_BANK:],
                scale_e8m0=PACK_SCALE_SLOT1 if slot == 1 else PACK_SCALE_SLOT2,
            )
            return packed_rows, ZERO_BANK_ROWS

        if self.kind == "fp8unpack":
            packed_rows = PACKED_ROWS_SLOT1 if slot == 1 else PACKED_ROWS_SLOT2
            out_rows = run_fp8_unpack_rows(
                packed_rows,
                scale_e8m0=UNPACK_SCALE_SLOT1 if slot == 1 else UNPACK_SCALE_SLOT2,
            )
            return out_rows[:ROWS_PER_BANK], out_rows[ROWS_PER_BANK:]

        if self.kind == "vli_pair":
            regs = run_vli_registers(self.name, self._imm(), dst_bank=pair_base)
            return regs[pair_base], regs[pair_base + 1]

        if self.kind == "vli_single":
            regs = run_vli_registers(self.name, self._imm(), dst_bank=self._odd_bank(slot))
            return ZERO_BANK_ROWS, regs[self._odd_bank(slot)]

        raise ValueError(f"unknown result kind {self.kind!r}")


ALL_SINGLE_OPS: list[OpSpec] = [
    OpSpec("mov", "VMOV", "unary", "mov", "safe", False, True),
    OpSpec("rcp", "VRECIP.BF16", "unary", "rcp", "safe", False, True),
    OpSpec("exp", "VEXP", "unary", "exp", "safe", False, True),
    OpSpec("exp2", "VEXP2", "unary", "exp", "safe", False, True),
    OpSpec("square", "VSQUARE.BF16", "unary", "squarecube", "safe", False, True),
    OpSpec("cube", "VCUBE.BF16", "unary", "squarecube", "safe", False, True),
    OpSpec("fp8pack", "VFP8PACK", "fp8pack", "fp8pack", "safe", True, False),
    OpSpec("fp8unpack", "VFP8UNPACK", "fp8unpack", "fp8unpack", "packed", False, True),
    OpSpec("relu", "VRELU", "unary", "relu", "safe", False, True),
    OpSpec("sin", "VSIN", "unary", "sincos", "safe", False, True),
    OpSpec("cos", "VCOS", "unary", "sincos", "safe", False, True),
    OpSpec("tanh", "VTANH", "unary", "tanh", "safe", False, True),
    OpSpec("log", "VLOG2", "unary", "log", "safe", False, True),
    OpSpec("sqrt", "VSQRT", "unary", "sqrt", "safe", False, True),
    OpSpec("csum", "VREDSUM.BF16", "signed_reduce", "csum", "signed", False, True),
    OpSpec("cmin", "VREDMIN.BF16", "signed_reduce", "cmin", "signed", False, True),
    OpSpec("cmax", "VREDMAX.BF16", "signed_reduce", "cmax", "signed", False, True),
    OpSpec("vliAll", "VLI.ALL", "vli_pair", "vli", "none", False, True),
    OpSpec("vliRow", "VLI.ROW", "vli_pair", "vli", "none", True, True),
    OpSpec("vliCol", "VLI.COL", "vli_single", "vli", "none", True, False),
    OpSpec("vliOne", "VLI.ONE", "vli_single", "vli", "none", True, False),
]

OPS_BY_NAME = {op.name: op for op in ALL_SINGLE_OPS}

POINTWISE_PACK_FIRST_OPS = [
    "mov", "rcp", "exp", "exp2", "square", "cube", "fp8pack", "fp8unpack",
    "relu", "sin", "cos", "tanh",
]

REDUCE_VLI_FIRST_OPS = [
    "log", "sqrt", "csum", "cmin", "cmax",
    "vliAll", "vliRow", "vliCol", "vliOne",
]


def _legal_overlap(op1: OpSpec, op2: OpSpec) -> bool:
    if op1.name == op2.name:
        return False
    if op1.shared_group == op2.shared_group:
        return False
    return True


@dataclass(frozen=True)
class OverlapCase:
    idx: int
    op1: OpSpec
    op2: OpSpec

    @property
    def output_base(self) -> int:
        return OUTPUT_BASE + self.idx * 8

    @property
    def output_base_bytes(self) -> int:
        return self.output_base * 32


def enumerate_cases(first_op_names: list[str]) -> list[OverlapCase]:
    first_ops = [OPS_BY_NAME[name] for name in first_op_names]
    cases: list[OverlapCase] = []
    for op1 in first_ops:
        for op2 in ALL_SINGLE_OPS:
            if _legal_overlap(op1, op2):
                cases.append(OverlapCase(len(cases), op1, op2))
    return cases


def _preloads() -> list[dict]:
    preloads = []
    preloads.extend(tensor_preloads(SRC_SAFE1_BASE, SAFE_ROWS_SLOT1))
    preloads.extend(tensor_preloads(SRC_SIGNED1_BASE, SIGNED_ROWS_SLOT1))
    preloads.extend(tensor_preloads(SRC_SAFE2_BASE, SAFE_ROWS_SLOT2))
    preloads.extend(tensor_preloads(SRC_SIGNED2_BASE, SIGNED_ROWS_SLOT2))
    preloads.extend(tensor_preloads(SRC_PACKED1_BASE, PACKED_ROWS_SLOT1))
    preloads.extend(tensor_preloads(SRC_PACKED2_BASE, PACKED_ROWS_SLOT2))
    return preloads


def _case_checks(case: OverlapCase) -> list[dict]:
    slot1_low, slot1_high = case.op1.result_rows(slot=1)
    slot2_low, slot2_high = case.op2.result_rows(slot=2)

    checks = []
    checks.extend(tensor_checks(case.output_base + 0, slot1_low[:SIGNATURE_ROWS]))
    checks.extend(tensor_checks(case.output_base + 2, slot1_high[:SIGNATURE_ROWS]))
    checks.extend(tensor_checks(case.output_base + 4, slot2_low[:SIGNATURE_ROWS]))
    checks.extend(tensor_checks(case.output_base + 6, slot2_high[:SIGNATURE_ROWS]))
    return checks


def emit_json(first_op_names: list[str]) -> str:
    cases = enumerate_cases(first_op_names)
    checks = []
    for case in cases:
        checks.extend(_case_checks(case))
    return json.dumps(
        {
            "dram_preloads": _preloads(),
            "dram_checks": checks,
            "timeout": TIMEOUT,
        },
        indent=2,
    )


FPGA_DRAM_BASE = 0x90000000  # FPGA DRAM window starts here (lower-32 byte addr)


def _emit_bf16_pair_load(lines: list[str], name: str, dram_base_bytes: int, vmem_base_words: int, dst_bank: int) -> None:
    lines.extend(
        [
            f"    # Load {name} -> m{dst_bank},m{dst_bank + 1}",
            f"    LI    x6, {vmem_base_words}",
            f"    LI    x1, 0x{FPGA_DRAM_BASE + dram_base_bytes:08X}",
            "    DMA.LOAD  x6, x1, x2, 0",
            "    DMA.WAIT  0",
            f"    LI    x6, {vmem_base_words + 256}",
            f"    LI    x1, 0x{FPGA_DRAM_BASE + dram_base_bytes + INPUT_BANK_BYTES:08X}",
            "    DMA.LOAD  x6, x1, x2, 0",
            "    DMA.WAIT  0",
            f"    LI    x6, {vmem_base_words}",
            f"    VLOAD {dst_bank}, x6, 0",
            f"    DELAY {VSTORE_DELAY}",
            f"    VLOAD {dst_bank + 1}, x6, 8",
            f"    DELAY {VSTORE_DELAY}",
            "",
        ]
    )


def _emit_single_bank_load(lines: list[str], name: str, dram_base_bytes: int, vmem_base_words: int, dst_bank: int) -> None:
    lines.extend(
        [
            f"    # Load {name} -> m{dst_bank}",
            f"    LI    x6, {vmem_base_words}",
            f"    LI    x1, 0x{FPGA_DRAM_BASE + dram_base_bytes:08X}",
            "    DMA.LOAD  x6, x1, x2, 0",
            "    DMA.WAIT  0",
            f"    LI    x6, {vmem_base_words}",
            f"    VLOAD {dst_bank}, x6, 0",
            f"    DELAY {VSTORE_DELAY}",
            "",
        ]
    )


def _emit_slot_init(lines: list[str], pair_base: int) -> None:
    lines.extend(
        [
            f"    VLI.ALL {pair_base}, 0x0000",
            f"    DELAY {INIT_DELAY}",
        ]
    )


def _emit_signature_store(lines: list[str], bank: int, scratch_words: int, dram_off_bytes: int) -> None:
    lines.extend(
        [
            f"    LI    x8, {scratch_words}",
            f"    VSTORE {bank}, x8, 0",
            f"    DELAY {VSTORE_DELAY}",
            f"    LI    x8, {scratch_words}",
            f"    LI    x3, 0x{FPGA_DRAM_BASE + dram_off_bytes:08X}",
            "    DMA.STORE x3, x8, x21, 1",
            "    DMA.WAIT  1",
        ]
    )


def emit_assembly(test_name: str, title: str, first_op_names: list[str]) -> str:
    cases = enumerate_cases(first_op_names)

    lines: list[str] = [
        f"# {test_name}.S — {title}",
        "# Exhaustive ordered overlap coverage for the current VPU single-input",
        "# overlap window: each case issues op1 and op2 in consecutive cycles",
        "# with no intervening DELAY. Double-port ops are intentionally excluded",
        "# because the current FSM never accepts them into the overlap slot.",
        f"# Cases in this file: {len(cases)}",
        f"# @TIMEOUT {TIMEOUT}",
        "# @DRAM_BASE 0x90000000",
        f"# @PYTHON_GEN gen_{test_name}.py",
        "#",
        "# DRAM inputs (offsets relative to FPGA DRAM base 0x9000_0000):",
        "#   [0x0000..0x07FF]  safe slot1 BF16 tensor pair",
        "#   [0x0800..0x0FFF]  signed slot1 BF16 tensor pair",
        "#   [0x1000..0x17FF]  safe slot2 BF16 tensor pair",
        "#   [0x1800..0x1FFF]  signed slot2 BF16 tensor pair",
        "#   [0x2000..0x23FF]  packed slot1 source bank",
        "#   [0x2400..0x27FF]  packed slot2 source bank",
        "# Outputs start at 0x2800; each case stores 4 signatures x 64 B = 256 B:",
        "#   slot1 low, slot1 high, slot2 low, slot2 high",
        "",
        "    ADDI  x28, x0, 1",
        "    LI    x5, 0x00000000",
        "    DMA.CONFIG x5, 0",
        f"    LI    x2, {INPUT_COPY_SIZE_WORDS}",
        f"    LI    x21, {SIGNATURE_COPY_SIZE_WORDS}",
        "",
        "    # Scale registers for fp8pack / fp8unpack cases",
        f"    SELI  {PACK_SCALE_REG_SLOT1}, 0x{PACK_SCALE_SLOT1:02X}",
        f"    SELI  {UNPACK_SCALE_REG_SLOT1}, 0x{UNPACK_SCALE_SLOT1:02X}",
        f"    SELI  {PACK_SCALE_REG_SLOT2}, 0x{PACK_SCALE_SLOT2:02X}",
        f"    SELI  {UNPACK_SCALE_REG_SLOT2}, 0x{UNPACK_SCALE_SLOT2:02X}",
        "",
    ]

    _emit_bf16_pair_load(lines, "safe slot1 pair", 0x0000, 0, 0)
    _emit_bf16_pair_load(lines, "signed slot1 pair", 0x0800, 512, 2)
    _emit_bf16_pair_load(lines, "safe slot2 pair", 0x1000, 1024, 4)
    _emit_bf16_pair_load(lines, "signed slot2 pair", 0x1800, 1536, 6)
    _emit_single_bank_load(lines, "packed slot1 bank", 0x2000, 2048, 8)
    _emit_single_bank_load(lines, "packed slot2 bank", 0x2400, 2304, 9)

    for case in cases:
        lines.extend(
            [
                "    # ----------------------------------------------------------------",
                f"    # Case {case.idx:03d}: {case.op1.name} || {case.op2.name}",
                "    # ----------------------------------------------------------------",
            ]
        )

        if case.op1.init_required:
            _emit_slot_init(lines, SLOT1_PAIR_BASE)
        if case.op2.init_required:
            _emit_slot_init(lines, SLOT2_PAIR_BASE)

        lines.append(f"    {case.op1.asm(slot=1)}")
        lines.append(f"    {case.op2.asm(slot=2)}")
        lines.append(f"    DELAY {CASE_DELAY}")

        _emit_signature_store(lines, SLOT1_PAIR_BASE, SCRATCH_SLOT1_LOW, case.output_base_bytes + 0x00)
        _emit_signature_store(lines, SLOT1_PAIR_BASE + 1, SCRATCH_SLOT1_HIGH, case.output_base_bytes + 0x40)
        _emit_signature_store(lines, SLOT2_PAIR_BASE, SCRATCH_SLOT2_LOW, case.output_base_bytes + 0x80)
        _emit_signature_store(lines, SLOT2_PAIR_BASE + 1, SCRATCH_SLOT2_HIGH, case.output_base_bytes + 0xC0)
        lines.append("")

    lines.extend(
        [
            "pass:",
            "    ADDI  x1, x0, 1",
            "    CSRRW x0, 0xC10, x1",
            "    ECALL",
            "",
            "fail:",
            "    CSRRW x0, 0xC10, x28",
            "    ECALL",
            "",
        ]
    )

    return "\n".join(lines)


def main_entry(test_name: str, title: str, first_op_names: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emit-asm",
        action="store_true",
        help="emit assembly instead of golden JSON",
    )
    args = parser.parse_args()

    if args.emit_asm:
        print(emit_assembly(test_name, title, first_op_names))
    else:
        print(emit_json(first_op_names))

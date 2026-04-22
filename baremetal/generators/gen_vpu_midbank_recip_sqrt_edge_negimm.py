#!/usr/bin/env python3
"""
gen_vpu_midbank_recip_sqrt_edge_negimm.py

Golden generator for vpu_midbank_recip_sqrt_edge_negimm.S.

Assembly dataflow:
  DRAM[0x0000..0x07FF] -> DMA.LOAD -> VMEM[0x37000..0x377FF]
  VLOAD m40 from base 0xDD00 with imm -8
  VLOAD m41 from base 0xDD00 with imm 0
  VRECIP.BF16 m42,m43 = recip(m40,m41)
  VSQRT       m44,m45 = sqrt(m40,m41)
  VSTORE m42/m43 -> VMEM[0x39000..0x397FF] -> DMA.STORE -> DRAM[0x0800..0x0FFF]
  VSTORE m44/m45 -> VMEM[0x3B000..0x3B7FF] -> DMA.STORE -> DRAM[0x1000..0x17FF]

The input tensor is class-directed rather than random: each logical 32-lane row
contains zeros, subnormals, normals, infinities, NaNs, and negative-domain
values, then rotates by row index so the operand classes exercise different
lanes and both physical banks across the full 32-row tile.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    tensor_preloads,
    tensor_checks,
    run_unary_rows,
)

TIMEOUT = 250000
INPUT_BASE_BEAT = 0x0000 // 32
RECIP_BASE_BEAT = 0x0800 // 32
SQRT_BASE_BEAT = 0x1000 // 32

if BF16_PER_BEAT != 16:
    raise ValueError(f"expected 16 BF16 lanes per physical row, got {BF16_PER_BEAT}")
if ROWS_PER_REGISTER != 32:
    raise ValueError(f"expected 32 rows per physical register, got {ROWS_PER_REGISTER}")
if ROWS_PER_TENSOR != 64:
    raise ValueError(f"expected 64 physical rows per BF16 tensor pair, got {ROWS_PER_TENSOR}")

# One full logical 32-lane row worth of explicit BF16 classes.
# Raw bit patterns are intentional here: the test is about classification paths,
# and the VPU golden still comes from the checked-in functional model.
EDGE_COLS = [
    0x0000,  # +0
    0x8000,  # -0
    0x0001,  # +min subnormal
    0x0020,  # +mid subnormal
    0x007F,  # +max subnormal
    0x0080,  # +min normal
    0x0081,  # +next normal
    0x3F00,  # +0.5
    0x3F80,  # +1.0
    0x3F88,  # +1.0625
    0x3FC0,  # +1.5
    0x3FF0,  # +1.875
    0x4000,  # +2.0
    0x4040,  # +3.0
    0x4120,  # +10.0
    0x7F7F,  # +max finite
    0x7F80,  # +inf
    0x7FC1,  # +qNaN
    0x7F81,  # +sNaN
    0x8001,  # -min subnormal
    0x8020,  # -mid subnormal
    0x807F,  # -max subnormal
    0x8080,  # -min normal
    0x8081,  # -next normal
    0xBF00,  # -0.5
    0xBF80,  # -1.0
    0xBFC0,  # -1.5
    0xC000,  # -2.0
    0xC120,  # -10.0
    0xFF7F,  # -max finite
    0xFF80,  # -inf
    0xFFC1,  # -qNaN
]

if len(EDGE_COLS) != 2 * BF16_PER_BEAT:
    raise ValueError("edge-class row must cover one full 32-lane logical row")


def rotate(vals, n):
    n %= len(vals)
    return vals[n:] + vals[:n]


logical_rows = []
for row_idx in range(ROWS_PER_REGISTER):
    logical_rows.append(rotate(EDGE_COLS, row_idx))

# Register-stream layout expected by tensor_preloads/tensor_checks and the VPU
# helpers: all 32 rows of the first physical register, then all 32 rows of the
# second physical register.
input_rows = [row[:BF16_PER_BEAT] for row in logical_rows]
input_rows += [row[BF16_PER_BEAT:] for row in logical_rows]

if len(input_rows) != ROWS_PER_TENSOR:
    raise ValueError("constructed wrong number of physical rows for BF16 tensor pair")

preloads = tensor_preloads(INPUT_BASE_BEAT, input_rows)

recip_rows = run_unary_rows("rcp", input_rows)
sqrt_rows = run_unary_rows("sqrt", input_rows)

checks = []
checks.extend(tensor_checks(RECIP_BASE_BEAT, recip_rows))
checks.extend(tensor_checks(SQRT_BASE_BEAT, sqrt_rows))

emit_test_data(preloads, checks, timeout=TIMEOUT)

#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data
from vpu_gen_utils import (
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    tensor_preloads,
    tensor_checks,
    fp8_checks,
    run_row_reduce_tensor,
    run_unary_rows,
    run_binary_rows,
    run_fp8_pack_rows,
    float_to_bf16,
)

TIMEOUT = 500000

# DRAM beat offsets (32 B / beat)
INPUT_LO_BEAT = 0x0000 // 32
INPUT_HI_BEAT = 0x0400 // 32
SCALE_BEAT    = 0x0800 // 32
OUT_LO_BEAT   = 0x1000 // 32
OUT_HI_BEAT   = 0x1400 // 32
PACK_BEAT     = 0x1800 // 32

# Build a deterministic positive BF16 tensor whose per-row sums are nonzero and finite.
# 32 rows total, each row has 16 BF16 lanes per physical register.
rows_lo = []
rows_hi = []
for r in range(ROWS_PER_REGISTER):
    lo_row = []
    hi_row = []
    for c in range(16):
        lo_row.append(float_to_bf16(0.5 + 0.03125 * r + 0.015625 * c))
        hi_row.append(float_to_bf16(1.0 + 0.03125 * r + 0.015625 * c))
    rows_lo.append(lo_row)
    rows_hi.append(hi_row)

input_rows = rows_lo + rows_hi

# row_sum broadcast across full tensor pair
row_sum_rows = run_row_reduce_tensor("rsum", input_rows)

# reciprocal(row_sum)
recip_rows = run_unary_rows("rcp", row_sum_rows)

# normalized = input * reciprocal(row_sum)
normalized_rows = run_binary_rows("mul", input_rows, recip_rows)

# Pack with unit scale loaded by SELD from VMEM byte 0x0800.
# E8M0 0x7F corresponds to 2^0.
scale_e8m0 = 0x7F
packed_rows = run_fp8_pack_rows(normalized_rows[:ROWS_PER_REGISTER], normalized_rows[ROWS_PER_REGISTER:], scale_e8m0)

preloads = []
preloads.extend(tensor_preloads(INPUT_LO_BEAT, rows_lo))
preloads.extend(tensor_preloads(INPUT_HI_BEAT, rows_hi))
preloads.append({
    "word_offset": SCALE_BEAT,
    "data": f"0x{scale_e8m0:02x}"
})

checks = []
checks.extend(tensor_checks(OUT_LO_BEAT, normalized_rows[:ROWS_PER_REGISTER]))
checks.extend(tensor_checks(OUT_HI_BEAT, normalized_rows[ROWS_PER_REGISTER:]))
checks.extend(fp8_checks(PACK_BEAT, packed_rows))

emit_test_data(preloads, checks, timeout=TIMEOUT)

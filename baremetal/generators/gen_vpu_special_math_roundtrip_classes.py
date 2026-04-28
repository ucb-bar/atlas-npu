#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    float_to_bf16,
    tensor_preloads,
    tensor_checks,
    run_unary_rows,
)


TIMEOUT = 700000

IN_BASE_BEAT = 0x0000 // 32
SQRT_BASE_BEAT = 0x0800 // 32
RECIP_BASE_BEAT = 0x1000 // 32
LOG2_BASE_BEAT = 0x1800 // 32
EXP2_BASE_BEAT = 0x2000 // 32


def repeat_row(row, count):
    return [list(row) for _ in range(count)]


# Keep the same class-directed stimulus as the original test, but generate
# expected results with the shared RTL-mirror VPU model rather than ideal math.
vals = [
    0.0, -0.0,
    2.0**-20, 2.0**-14, 2.0**-10, 2.0**-6,
    0.125, 0.15625, 0.1875, 0.21875,
    0.25, 0.5, 0.75,
    1.0, 1.125, 1.5, 1.875,
    2.0, 3.0, 4.0, 8.0, 16.0,
    31.0 / 16.0, 33.0 / 16.0,
    0.9921875, 1.0078125,
    63.0 / 64.0, 65.0 / 64.0,
    32.0, 64.0, 128.0, 192.0,
]

input_row_lo = [float_to_bf16(v) for v in vals[:BF16_PER_BEAT]]
input_row_hi = [float_to_bf16(v) for v in vals[BF16_PER_BEAT:]]
input_rows = repeat_row(input_row_lo, ROWS_PER_REGISTER) + repeat_row(input_row_hi, ROWS_PER_REGISTER)

sqrt_rows = run_unary_rows("sqrt", input_rows)
recip_rows = run_unary_rows("rcp", input_rows)
log2_rows = run_unary_rows("log", input_rows)
exp2_rows = run_unary_rows("exp2", log2_rows)

preloads = tensor_preloads(IN_BASE_BEAT, input_rows)

checks = []
checks.extend(tensor_checks(SQRT_BASE_BEAT, sqrt_rows))
checks.extend(tensor_checks(RECIP_BASE_BEAT, recip_rows))
checks.extend(tensor_checks(LOG2_BASE_BEAT, log2_rows))
checks.extend(tensor_checks(EXP2_BASE_BEAT, exp2_rows))

emit_test_data(preloads, checks, timeout=TIMEOUT)

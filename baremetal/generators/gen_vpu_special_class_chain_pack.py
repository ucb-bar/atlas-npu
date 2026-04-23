#!/usr/bin/env python3
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_TENSOR,
    float_to_bf16,
    tensor_preloads,
    tensor_checks,
    fp8_checks,
    run_unary_rows,
    run_binary_rows,
    run_fp8_pack_rows,
)

TIMEOUT = 300000

# DRAM byte offsets from assembly, converted to DMA-beat offsets (32 B/beat).
IN_BASE_BEAT = 0x0000 // 32
RECIP_BASE_BEAT = 0x0800 // 32
SQRT_BASE_BEAT = 0x1000 // 32
MUL_BASE_BEAT = 0x1800 // 32
PACK_BASE_BEAT = 0x2000 // 32
SCALE_BASE_BEAT = 0x2400 // 32

# Scale byte used by SELD e3. 0x7F is the unit exponent in E8M0.
SCALE_E8M0 = 0x7F

# Build 32 rows x 32 lanes of class-directed BF16 inputs.
# Each row is 16 lanes; the full tensor pair is 32 rows where rows 0..31
# stream through the pair model exactly as tensor_preloads/tensor_checks expect.
row_patterns = [
    [0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x8080, 0x3f80, 0xbf80,
     0x4000, 0xc000, 0x3fc0, 0xbfc0, 0x3fe0, 0xbfe0, 0x7f80, 0xff80],
    [0x7fc0, 0x7fa1, 0x0081, 0x0100, 0x0180, 0x3f00, 0x3f7f, 0x3f81,
     0x4001, 0x407f, 0x4080, 0x40ff, 0x4100, 0x3e80, 0x3e00, 0x3d80],
    [0x0002, 0x0003, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x007f,
     0x8081, 0x8100, 0x8180, 0x8200, 0x3c00, 0x3c80, 0x3d00, 0x3d80],
    [0x4300, 0x4380, 0x4400, 0x4480, 0x4500, 0x4580, 0x4600, 0x4680,
     0xc300, 0xc380, 0xc400, 0xc480, 0xc500, 0xc580, 0xc600, 0xc680],
    [0x3f01, 0x3f02, 0x3f04, 0x3f08, 0x3f10, 0x3f20, 0x3f40, 0x3f60,
     0x3f70, 0x3f78, 0x3f7c, 0x3f7e, 0x3f7f, 0x3f80, 0x3f81, 0x3f82],
    [0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4120,
     0x4140, 0x4160, 0x4180, 0x41a0, 0x41c0, 0x41e0, 0x4200, 0x4220],
    [0xbf00, 0xbf40, 0xbf80, 0xbfa0, 0xbfc0, 0xbfe0, 0xc000, 0xc020,
     0xc040, 0xc060, 0xc080, 0xc0a0, 0xc0c0, 0xc0e0, 0xc100, 0xc120],
    [0x7f7f, 0xff7f, 0x7f00, 0xff00, 0x7e80, 0xfe80, 0x7e00, 0xfe00,
     0x3a80, 0x3b00, 0x3b80, 0x3bc0, 0x3c40, 0x3cc0, 0x3d40, 0x3dc0],
]

input_rows = [list(row_patterns[i % len(row_patterns)]) for i in range(ROWS_PER_TENSOR)]

# VPU model goldens.
recip_rows = run_unary_rows("rcp", input_rows)
sqrt_rows = run_unary_rows("sqrt", input_rows)
mul_rows = run_binary_rows("mul", recip_rows, sqrt_rows)
packed_rows = run_fp8_pack_rows(mul_rows[:ROWS_PER_TENSOR // 2], mul_rows[ROWS_PER_TENSOR // 2:], SCALE_E8M0)

preloads = []
checks = []

# BF16 input tensor pair preload.
preloads.extend(tensor_preloads(IN_BASE_BEAT, input_rows))

# Scale table preload: one 32-byte beat, byte 0 consumed by SELD e3.
scale_bytes = [SCALE_E8M0] + [0] * (BF16_PER_BEAT * 2 - 1)
scale_word = 0
for i, b in enumerate(scale_bytes):
    scale_word |= (int(b) & 0xFF) << (8 * i)
preloads.append({"word_offset": SCALE_BASE_BEAT, "data": f"0x{scale_word:064x}"})

checks.extend(tensor_checks(RECIP_BASE_BEAT, recip_rows))
checks.extend(tensor_checks(SQRT_BASE_BEAT, sqrt_rows))
checks.extend(tensor_checks(MUL_BASE_BEAT, mul_rows))
checks.extend(fp8_checks(PACK_BASE_BEAT, packed_rows))

emit_test_data(preloads, checks, timeout=TIMEOUT)

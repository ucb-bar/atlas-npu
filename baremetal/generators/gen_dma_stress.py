#!/usr/bin/env python3
"""
gen_dma_stress.py — DMA stress test: 16 matrix copy operations.

This generator matches dma_stress.S as written:

  - 16 iterations
  - 1024 bytes copied per iteration
  - source region: beats [0..511]
  - destination region: beats [512..1023]

The assembly test harness DRAM is indexed in DMA-width beats, not 32-bit words.
For widthBytes = 32, each beat holds 8 x 32-bit words = 256 bits.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import emit_test_data

NUM = 16
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1300

DMA_WIDTH_BYTES = 32
BEATS_PER_MATRIX = 1024 // DMA_WIDTH_BYTES   # 32 beats per matrix

SRC_BASE_BEAT = 0
DST_BASE_BEAT = 512   # matches x17 = 16384 bytes = 512 beats

rng = np.random.RandomState(SEED)

preloads = []
checks = []

for i in range(NUM):
    # Generate 32 random 256-bit beats for this matrix payload
    beats = []
    for _ in range(BEATS_PER_MATRIX):
        beat = 0
        for lane in range(8):  # 8 x 32-bit words per beat
            w = int(rng.randint(0, 2**32, dtype=np.uint32))
            beat |= (w & 0xFFFFFFFF) << (32 * lane)
        beats.append(beat)

    for b, beat in enumerate(beats):
        preloads.append({
            "word_offset": SRC_BASE_BEAT + i * BEATS_PER_MATRIX + b,
            "data": f"0x{beat:064x}"
        })
        checks.append({
            "word_offset": DST_BASE_BEAT + i * BEATS_PER_MATRIX + b,
            "expected": f"0x{beat:064x}"
        })

emit_test_data(preloads, checks, timeout=300000)

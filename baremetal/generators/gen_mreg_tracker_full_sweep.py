#!/usr/bin/env python3
"""
gen_mreg_tracker_full_sweep.py — golden generator for mreg_tracker_full_sweep.S

The assembly sweeps every MREG bank 0..63 as both a VLOAD destination
and a VSTORE source. Because every VLOAD uses the SAME 1 KiB source
tile, every VSTORE result also equals that same 1 KiB tile. The DRAM
output region is therefore the input tile concatenated 64 times.

Mid-test "cross-engine" touches (VPU VMOV, XLU transpose, MXU0/MXU1
push/pop) rewrite some MREG banks, but the assembly RELOADS every
bank from the source before the VSTORE sweep, so those intermediate
rewrites do not affect the oracle. Their purpose is purely to make
MregBankTracker's readers/writers valid lines toggle for additional
engine-to-bank combinations that boost the tracker's toggle coverage.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "generators"))

from gen_utils import (
    quantize_fp8,
    matrix_to_fp8_words,
    preloads_from_words_packed,
    checks_from_words_packed,
    rand_matrix_fp8_safe,
)

TILE = 32
TILE_BYTES = 1024
NUM_BANKS = 64
SEED = 4242
TIMEOUT = 1500000

INPUT_DRAM_BYTES = 0x00000
OUTPUT_DRAM_BYTES = 0x01000       # 64 banks x 1024 B = 64 KiB output region


def beat_offset(byte_off: int) -> int:
    assert byte_off % 32 == 0
    return byte_off // 32


def main() -> None:
    # Deterministic FP8-safe tile so the oracle is reproducible.
    tile = quantize_fp8(rand_matrix_fp8_safe(TILE, TILE, seed=SEED)).astype(np.float32)
    tile_words = matrix_to_fp8_words(tile)

    preloads = []
    preloads += preloads_from_words_packed(beat_offset(INPUT_DRAM_BYTES), tile_words)

    checks = []
    for bank_idx in range(NUM_BANKS):
        dram_off = OUTPUT_DRAM_BYTES + bank_idx * TILE_BYTES
        checks += checks_from_words_packed(beat_offset(dram_off), tile_words)

    sys.stderr.write(
        f"[gen_mreg_tracker_full_sweep] banks={NUM_BANKS} "
        f"tile_bytes={TILE_BYTES} preloads={len(preloads)} checks={len(checks)}\n"
    )

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_dma_overlap.py
Generate test vectors for dma_overlap_check.S.

Tests multiple in-flight DMA transfers with early and late waits.
Each input region has a distinct byte pattern; the assembly round-trips
data through VMEM → TRF → VMEM → DRAM and the output must match.

DRAM layout (32-byte beats / "word offsets"):
  Input A:   word offsets   0..31   (1024 B, all 0xAA)
  Input B:   word offsets  32..63   (1024 B, all 0xBB)
  Input C:   word offsets  64..95   (1024 B, all 0xCC)
  Output A:  word offsets  96..127  (1024 B, expect 0xAA)
  Output B:  word offsets 128..159  (1024 B, expect 0xBB)
  Output C:  word offsets 160..191  (1024 B, expect 0xCC)
"""

import json

DMA_BEAT_BYTES = 32
WORDS_PER_TILE = 32   # 1024 bytes / 32 bytes per beat

INPUT_A_START  = 0
INPUT_B_START  = 32
INPUT_C_START  = 64
OUTPUT_A_START = 96
OUTPUT_B_START = 128
OUTPUT_C_START = 160

PATTERN_A = 0xAA
PATTERN_B = 0xBB
PATTERN_C = 0xCC

TIMEOUT = 20000


def pack_32_bytes_le(byte_val) -> str:
    """Pack 32 identical bytes into one 256-bit beat, byte 0 in LSB."""
    word = 0
    for i in range(32):
        word |= (byte_val & 0xFF) << (8 * i)
    return f"0x{word:064x}"


def make_preloads():
    preloads = []

    for pattern, start in [
        (PATTERN_A, INPUT_A_START),
        (PATTERN_B, INPUT_B_START),
        (PATTERN_C, INPUT_C_START),
    ]:
        beat = pack_32_bytes_le(pattern)
        for w in range(WORDS_PER_TILE):
            preloads.append({
                "word_offset": start + w,
                "data": beat,
            })

    return preloads


def make_checks():
    checks = []

    for pattern, start in [
        (PATTERN_A, OUTPUT_A_START),
        (PATTERN_B, OUTPUT_B_START),
        (PATTERN_C, OUTPUT_C_START),
    ]:
        beat = pack_32_bytes_le(pattern)
        for w in range(WORDS_PER_TILE):
            checks.append({
                "word_offset": start + w,
                "expected": beat,
            })

    return checks


def main():
    payload = {
        "dram_preloads": make_preloads(),
        "dram_checks": make_checks(),
        "timeout": TIMEOUT,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
    
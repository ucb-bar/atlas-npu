#!/usr/bin/env python3
"""Golden data for dma_reconfig_base.S — FPGA single-base reissue variant.

FPGA only has memory at 0x9000_0000+, so the test reissues DMA.CONFIG
with upper-32 = 0 across two distinct lower-32 windows rather than
sweeping upper-32 values.
"""

import json

BEAT_BYTES = 32
WORDS_PER_BEAT = 8
BEATS = 4

# Byte offsets *relative* to @DRAM_BASE in the .S file (currently 0x9000_0000).
# The assembler adds @DRAM_BASE to these word_offsets to compute the final
# byte address.
LOAD_REGION_A  = 0x0000   # first LOAD source
LOAD_REGION_B  = 0x4000   # second LOAD source
STORE_REGION_A = 0x2000   # first STORE dest
STORE_REGION_B = 0x6000   # second STORE dest


def beat_offset(byte_off):
    assert byte_off % BEAT_BYTES == 0
    return byte_off // BEAT_BYTES


def pack_words(words):
    assert len(words) == WORDS_PER_BEAT
    payload = 0
    for i, word in enumerate(words):
        payload |= (word & 0xFFFFFFFF) << (32 * i)
    return f"0x{payload:064x}"


def make_beat(region, beat):
    return pack_words([
        0xC0DE0000 | (region << 12) | (beat << 4) | word
        for word in range(WORDS_PER_BEAT)
    ])


def emit(entries, byte_base, region, key):
    for beat in range(BEATS):
        entries.append({
            "word_offset": beat_offset(byte_base) + beat,
            key: make_beat(region, beat),
        })


def main():
    preloads = []
    checks = []

    # First reissue: LOAD region A → VMEM → STORE region A
    emit(preloads, LOAD_REGION_A,  0, "data")
    emit(checks,   STORE_REGION_A, 0, "expected")
    # Second reissue: LOAD region B → VMEM → STORE region B
    emit(preloads, LOAD_REGION_B,  1, "data")
    emit(checks,   STORE_REGION_B, 1, "expected")

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": 100000,
    }, indent=2))


if __name__ == "__main__":
    main()

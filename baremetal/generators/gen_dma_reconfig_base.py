#!/usr/bin/env python3
"""Golden data for dma_reconfig_base.S."""

import json

BEAT_BYTES = 32
WORDS_PER_BEAT = 8
BEATS = 4


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

    emit(preloads, 0x0000, 0, "data")
    emit(preloads, 0x1000, 1, "data")
    emit(checks, 0x2000, 0, "expected")
    emit(checks, 0x3000, 1, "expected")

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": 100000,
    }, indent=2))


if __name__ == "__main__":
    main()

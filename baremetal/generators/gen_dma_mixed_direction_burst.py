#!/usr/bin/env python3
"""Golden data for dma_mixed_direction_burst.S."""

import json

BEAT_BYTES = 32
WORDS_PER_BEAT = 8
BEATS_PER_REGION = 8

STORE_INPUT_BASES = [0x0000, 0x0400, 0x0800, 0x0C00]
LOAD_INPUT_BASES = [0x1000, 0x1400, 0x1800, 0x1C00]
STORE_OUTPUT_BASES = [0x2000, 0x2400, 0x2800, 0x2C00]
LOAD_OUTPUT_BASES = [0x3000, 0x3400, 0x3800, 0x3C00]


def beat_offset(byte_off):
    assert byte_off % BEAT_BYTES == 0
    return byte_off // BEAT_BYTES


def pack_words(words):
    assert len(words) == WORDS_PER_BEAT
    payload = 0
    for i, word in enumerate(words):
        payload |= (word & 0xFFFFFFFF) << (32 * i)
    return f"0x{payload:064x}"


def make_beat(kind, index, beat):
    tag = 0xA if kind == "store_src" else 0xB
    return pack_words([
        0xD0000000 | (tag << 20) | (index << 12) | (beat << 4) | word
        for word in range(WORDS_PER_BEAT)
    ])


def emit(entries, byte_base, data, key):
    for beat, payload in enumerate(data):
        entries.append({
            "word_offset": beat_offset(byte_base) + beat,
            key: payload,
        })


def main():
    store_tiles = [
        [make_beat("store_src", i, beat) for beat in range(BEATS_PER_REGION)]
        for i in range(4)
    ]
    load_tiles = [
        [make_beat("load_src", i, beat) for beat in range(BEATS_PER_REGION)]
        for i in range(4)
    ]

    preloads = []
    for base, tile in zip(STORE_INPUT_BASES, store_tiles):
        emit(preloads, base, tile, "data")
    for base, tile in zip(LOAD_INPUT_BASES, load_tiles):
        emit(preloads, base, tile, "data")

    checks = []
    for base, tile in zip(STORE_OUTPUT_BASES, store_tiles):
        emit(checks, base, tile, "expected")
    for base, tile in zip(LOAD_OUTPUT_BASES, load_tiles):
        emit(checks, base, tile, "expected")

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": 400000,
    }, indent=2))


if __name__ == "__main__":
    main()

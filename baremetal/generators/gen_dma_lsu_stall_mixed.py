#!/usr/bin/env python3
"""Golden data for dma_lsu_stall_mixed.S."""

import json

BEAT_BYTES = 32
WORDS_PER_BEAT = 8
BEATS_PER_TILE = 32

STORE_SRC_BASES = [0x0000, 0x0400, 0x0800, 0x0C00]
LOAD_SRC_BASES = [0x1000, 0x1400, 0x1800, 0x1C00]
STORE_OUT_BASES = [0x3000, 0x3400, 0x3800, 0x3C00]
LOAD_OUT_BASES = [0x4000, 0x4400, 0x4800, 0x4C00]


def beat_offset(byte_off):
    assert byte_off % BEAT_BYTES == 0
    return byte_off // BEAT_BYTES


def pack_words(words):
    assert len(words) == WORDS_PER_BEAT
    payload = 0
    for i, word in enumerate(words):
        payload |= (word & 0xFFFFFFFF) << (32 * i)
    return f"0x{payload:064x}"


def make_tile(kind, index):
    tag = 0x51 if kind == "store_src" else 0xA7
    return [
        pack_words([
            (tag << 24) | (index << 16) | (beat << 8) | word
            for word in range(WORDS_PER_BEAT)
        ])
        for beat in range(BEATS_PER_TILE)
    ]


def emit(entries, byte_base, tile, key):
    for beat, payload in enumerate(tile):
        entries.append({
            "word_offset": beat_offset(byte_base) + beat,
            key: payload,
        })


def main():
    store_tiles = [make_tile("store_src", i) for i in range(4)]
    load_tiles = [make_tile("load_src", i) for i in range(4)]

    preloads = []
    for base, tile in zip(STORE_SRC_BASES, store_tiles):
        emit(preloads, base, tile, "data")
    for base, tile in zip(LOAD_SRC_BASES, load_tiles):
        emit(preloads, base, tile, "data")

    checks = []
    for base, tile in zip(STORE_OUT_BASES, store_tiles):
        emit(checks, base, tile, "expected")
    for base, tile in zip(LOAD_OUT_BASES, load_tiles):
        emit(checks, base, tile, "expected")

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": 800000,
    }, indent=2))


if __name__ == "__main__":
    main()

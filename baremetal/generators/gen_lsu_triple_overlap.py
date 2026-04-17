#!/usr/bin/env python3
"""
gen_lsu_triple_overlap.py

Generate golden DRAM contents for lsu_triple_overlap.S.

The test exercises:
  - scalar stores overlapping with VLOAD + VSTORE
  - scalar loads overlapping with VLOAD + VSTORE

Vector paths are checked by round-tripping full 1024-byte tiles.
Scalar overlap paths are checked by storing one 32-byte line at the front of
an output tile and by copying four loaded words into a separate VMEM output
tile.
"""

import json

DMA_BEAT_BYTES = 32
TILE_BEATS = 32
WORDS_PER_BEAT = 8
TIMEOUT = 200000

A_IN_BASE = 0x0000
B_IN_BASE = 0x0400
SCALAR_LD_SRC_BASE = 0x0800
C_IN_BASE = 0x0C00
ZERO_BASE = 0x1000

OUT_STORE_1_BASE = 0x1400
OUT_LOAD_1_BASE = 0x1800
OUT_SCALAR_STORE_BASE = 0x1C00
OUT_STORE_2_BASE = 0x2000
OUT_LOAD_2_BASE = 0x2400
OUT_SCALAR_LOAD_BASE = 0x2800

SCALAR_STORE_WORDS = [
    0x11111111,
    0x22222222,
    0x33333333,
    0x44444444,
    0x55555555,
    0x66666666,
    0x77777777,
    0x88888888,
]

SCALAR_LOAD_WORDS = [
    0x89ABCDEF,
    0x01234567,
    0xDEADBEEF,
    0xCAFEBABE,
    0x00000000,
    0x00000000,
    0x00000000,
    0x00000000,
]


def beat_offset(byte_off: int) -> int:
    assert byte_off % DMA_BEAT_BYTES == 0
    return byte_off // DMA_BEAT_BYTES


def pack_bytes_le(values: list[int]) -> str:
    assert len(values) == DMA_BEAT_BYTES
    word = 0
    for i, value in enumerate(values):
        word |= (value & 0xFF) << (8 * i)
    return f"0x{word:064x}"


def pack_words_le(words: list[int]) -> str:
    assert len(words) == WORDS_PER_BEAT
    word = 0
    for i, value in enumerate(words):
        word |= (value & 0xFFFFFFFF) << (32 * i)
    return f"0x{word:064x}"


def make_pattern_tile(seed: int) -> list[str]:
    beats = []
    for beat in range(TILE_BEATS):
        values = [((seed + beat * 17 + byte * 7) & 0xFF) for byte in range(DMA_BEAT_BYTES)]
        beats.append(pack_bytes_le(values))
    return beats


def make_zero_tile() -> list[str]:
    return [pack_words_le([0] * WORDS_PER_BEAT) for _ in range(TILE_BEATS)]


def make_scalar_load_src_tile() -> list[str]:
    beats = make_zero_tile()
    beats[0] = pack_words_le(SCALAR_LOAD_WORDS)
    return beats


def emit_tile(entries: list[dict], byte_base: int, beats: list[str], key: str) -> None:
    base = beat_offset(byte_base)
    for i, beat in enumerate(beats):
        entries.append({
            "word_offset": base + i,
            key: beat,
        })


def main() -> None:
    tile_a = make_pattern_tile(0x11)
    tile_b = make_pattern_tile(0x5A)
    tile_c = make_pattern_tile(0xA3)
    scalar_ld_src = make_scalar_load_src_tile()
    zero_tile = make_zero_tile()

    scalar_store_out = make_zero_tile()
    scalar_store_out[0] = pack_words_le(SCALAR_STORE_WORDS)

    scalar_load_out = make_zero_tile()
    scalar_load_out[0] = pack_words_le(SCALAR_LOAD_WORDS)

    preloads = []
    emit_tile(preloads, A_IN_BASE, tile_a, "data")
    emit_tile(preloads, B_IN_BASE, tile_b, "data")
    emit_tile(preloads, SCALAR_LD_SRC_BASE, scalar_ld_src, "data")
    emit_tile(preloads, C_IN_BASE, tile_c, "data")
    emit_tile(preloads, ZERO_BASE, zero_tile, "data")

    checks = []
    emit_tile(checks, OUT_STORE_1_BASE, tile_b, "expected")
    emit_tile(checks, OUT_LOAD_1_BASE, tile_a, "expected")
    emit_tile(checks, OUT_SCALAR_STORE_BASE, scalar_store_out, "expected")
    emit_tile(checks, OUT_STORE_2_BASE, tile_b, "expected")
    emit_tile(checks, OUT_LOAD_2_BASE, tile_c, "expected")
    emit_tile(checks, OUT_SCALAR_LOAD_BASE, scalar_load_out, "expected")

    payload = {
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

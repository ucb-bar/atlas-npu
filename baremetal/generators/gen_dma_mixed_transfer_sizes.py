#!/usr/bin/env python3
"""Golden data for dma_mixed_transfer_sizes.S."""

import json

BEAT_BYTES = 32
WORDS_PER_BEAT = 8

CASES = [
    # channel, size_bytes, input_byte_base, output_byte_base
    (0, 32,  0x0000, 0x2000),
    (1, 64,  0x0400, 0x2400),
    (2, 96,  0x0800, 0x2800),
    (3, 160, 0x0C00, 0x2C00),
    (4, 224, 0x1000, 0x3000),
]


def beat_offset(byte_off):
    assert byte_off % BEAT_BYTES == 0
    return byte_off // BEAT_BYTES


def pack_words(words):
    assert len(words) == WORDS_PER_BEAT
    payload = 0
    for i, word in enumerate(words):
        payload |= (word & 0xFFFFFFFF) << (32 * i)
    return f"0x{payload:064x}"


def make_beat(channel, beat):
    return pack_words([
        0x51000000 | (channel << 20) | (beat << 8) | word
        for word in range(WORDS_PER_BEAT)
    ])


def main():
    preloads = []
    checks = []

    for channel, size_bytes, input_base, output_base in CASES:
        beats = size_bytes // BEAT_BYTES
        for beat in range(beats):
            data = make_beat(channel, beat)
            preloads.append({
                "word_offset": beat_offset(input_base) + beat,
                "data": data,
            })
            checks.append({
                "word_offset": beat_offset(output_base) + beat,
                "expected": data,
            })

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": 200000,
    }, indent=2))


if __name__ == "__main__":
    main()

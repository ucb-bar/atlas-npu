#!/usr/bin/env python3
"""Golden data for dma_bank_end_edges.S."""

import json

BEAT_BYTES = 32
WORDS_PER_BEAT = 8
BEATS_PER_REGION = 4
CHANNELS = 8


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
        0xBEE00000 | (channel << 16) | (beat << 8) | word
        for word in range(WORDS_PER_BEAT)
    ])


def main():
    preloads = []
    checks = []

    for channel in range(CHANNELS):
        input_base = channel * 0x400
        output_base = 0x4000 + channel * 0x400
        for beat in range(BEATS_PER_REGION):
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
        "timeout": 300000,
    }, indent=2))


if __name__ == "__main__":
    main()

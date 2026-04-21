#!/usr/bin/env python3
"""Golden data for scalar_subword_high_offsets.S."""

import json

BEAT_BYTES = 32


def pack_bytes(values):
    assert len(values) == BEAT_BYTES
    payload = 0
    for i, value in enumerate(values):
        payload |= (value & 0xFF) << (8 * i)
    return f"0x{payload:064x}"


def main():
    zero = [0] * BEAT_BYTES
    expected = [0] * BEAT_BYTES

    expected[31] = 0x81
    expected[28] = 0x80
    expected[29] = 0x7F
    expected[26] = 0x01
    expected[27] = 0x80
    expected[20] = 0xD4
    expected[21] = 0xC3
    expected[22] = 0xB2
    expected[23] = 0xA1

    print(json.dumps({
        "dram_preloads": [{
            "word_offset": 0,
            "data": pack_bytes(zero),
        }],
        "dram_checks": [{
            "word_offset": 32,
            "expected": pack_bytes(expected),
        }],
        "timeout": 10000,
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
gen_mxu0_push_pop.py

Generate test vectors for mxu0_push_pop_check.S.

This generator matches the assembly exactly:

DRAM layout (32-byte DMA beats / "word offsets"):
  Weights:      word offsets 0..31    (1024 bytes)
  Activations:  word offsets 32..63   (1024 bytes)
  Output:       word offsets 64..95   (1024 bytes)

Assembly flow:
  DRAM -> VMEM -> VLOAD -> TRF -> MXU* -> TRF -> VSTORE -> VMEM -> DRAM

Important:
- Inputs are 32x32 E4M3 tiles, all values = 1.0 (0x38)
- MXU* computes a 32x32 matmul
- VMATPOP.BF16.MXU* 2, 0 writes BF16 results starting at m2
- VSTORE 2 writes only 1024 bytes, i.e. only the lower 16 BF16 columns
  (32 rows x 16 cols x 2 bytes = 1024 bytes)

Expected output element:
  sum_{k=0..31} 1.0 * 1.0 = 32.0
  BF16(32.0) = 0x4200
"""

import json
import struct

TILE = 32
POPPED_COLS = 16              # only lower 16 BF16 cols are stored by VSTORE 2
DMA_BEAT_BYTES = 32
BF16S_PER_BEAT = DMA_BEAT_BYTES // 2   # 16 BF16 values per 32-byte beat

WEIGHT_WORD_START = 0
ACT_WORD_START = 32
OUT_WORD_START = 64

WORDS_PER_INPUT_TILE = 32     # 32 rows x 32 bytes/row = 1024 bytes
WORDS_PER_OUTPUT_TILE = 32    # 32 rows x 16 BF16/row = 32 beats = 1024 bytes

TIMEOUT = 20000

# Normal E4M3 encoding for +1.0
WEIGHT_BYTE = 0x38
ACT_BYTE = 0x38


def float_to_bits32(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", x))[0]


def float_to_bf16_bits(x: float) -> int:
    """Round float32 to BF16 using round-to-nearest-even."""
    bits = float_to_bits32(x)
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def e4m3_to_float_hw(byte: int) -> float:
    """
    Decode E4M3 to float matching the hardware behavior described in the test:
      - exp=0      -> zero/subnormal flushed to 0
      - exp=15,m=7 -> NaN flushed to 0
      - otherwise normal decode
    """
    byte &= 0xFF
    sign = -1.0 if (byte & 0x80) else 1.0
    exp = (byte >> 3) & 0xF
    mant = byte & 0x7

    if exp == 0:
        return 0.0
    if exp == 0xF and mant == 0x7:
        return 0.0

    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def pack_32_bytes_le(byte_vals) -> str:
    """Pack 32 bytes into one 256-bit beat, byte 0 in LSB."""
    assert len(byte_vals) == 32
    word = 0
    for i, b in enumerate(byte_vals):
        word |= (int(b) & 0xFF) << (8 * i)
    return f"0x{word:064x}"


def pack_16_bf16_le(bf16_vals) -> str:
    """Pack 16 BF16 values into one 256-bit beat, lane 0 in LSB."""
    assert len(bf16_vals) == 16
    word = 0
    for i, b in enumerate(bf16_vals):
        word |= (int(b) & 0xFFFF) << (16 * i)
    return f"0x{word:064x}"


def make_preloads():
    preloads = []

    weight_word = pack_32_bytes_le([WEIGHT_BYTE] * 32)
    for w in range(WORDS_PER_INPUT_TILE):
        preloads.append({
            "word_offset": WEIGHT_WORD_START + w,
            "data": weight_word,
        })

    act_word = pack_32_bytes_le([ACT_BYTE] * 32)
    for w in range(WORDS_PER_INPUT_TILE):
        preloads.append({
            "word_offset": ACT_WORD_START + w,
            "data": act_word,
        })

    return preloads


def make_checks():
    a = e4m3_to_float_hw(ACT_BYTE)
    w = e4m3_to_float_hw(WEIGHT_BYTE)

    out_val = sum(a * w for _ in range(TILE))   # 32 products
    out_bf16 = float_to_bf16_bits(out_val)      # should be 0x4200

    # Each stored DRAM beat corresponds to one row of the lower 16 BF16 columns.
    out_word = pack_16_bf16_le([out_bf16] * BF16S_PER_BEAT)

    checks = []
    for row in range(WORDS_PER_OUTPUT_TILE):
        checks.append({
            "word_offset": OUT_WORD_START + row,
            "expected": out_word,
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
    
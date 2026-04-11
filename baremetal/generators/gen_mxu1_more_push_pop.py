#!/usr/bin/env python3
"""
gen_mxu1_more_push_pop.py
Generate test vectors for mxu1_more_push_pop.S.

DRAM layout (32-byte beats / "word offsets"):
  Weights:      word offsets   0..31   (1024 bytes, E4M3)
  Activations:  word offsets  32..63   (1024 bytes, E4M3)
  Bias lo:      word offsets  64..95   (1024 bytes, BF16 for lanes 0–15)
  Bias hi:      word offsets  96..127  (1024 bytes, BF16 for lanes 16–31)
  Output:       word offsets 128..159  (1024 bytes, E4M3)

Assembly flow:
  DRAM → VMEM → VLOAD → TRF → MXU1 → TRF → VSTORE → VMEM → DRAM

VMATPUSH.ACC.BF16.MXU1 reads two TRFs:
  m6 (mregId)   → BF16 lanes 0–15
  m7 (mregId+1) → BF16 lanes 16–31

Inputs:
  - Weights:     32×32 E4M3, all 1.0 (0x38)
  - Activations: 32×32 E4M3, all 1.0 (0x38)
  - Bias lo:     32×16 BF16, all 8.0 (0x4100)
  - Bias hi:     32×16 BF16, all 8.0 (0x4100)

Computation:
  matmul = sum_{k=0..31} 1.0 × 1.0 = 32.0
  acc    = matmul + bias = 32.0 + 8.0 = 40.0  (all 32 lanes)

VMATPOP.FP8 with SELI 0, 0x7C:
  E8M0 scale = 124 → scaleExp = 124 - 127 = -3
  quant_scale  = 2^(-3) = 0.125
  40.0 × 0.125 = 5.0 → E4M3 = 0x4A

Output is 32 rows × 32 E4M3 bytes = 1024 bytes = 32 beats.
All bytes = 0x4A.
"""

import json
import math
import struct

TILE = 32
DMA_BEAT_BYTES = 32
BF16S_PER_BEAT = DMA_BEAT_BYTES // 2  # 16

WEIGHT_WORD_START  = 0
ACT_WORD_START     = 32
BIAS_LO_WORD_START = 64
BIAS_HI_WORD_START = 96
OUT_WORD_START     = 128

WORDS_PER_TILE = 32   # 1024 bytes / 32 bytes per beat

TIMEOUT = 20000

# E4M3 encoding for +1.0:  sign=0, exp=7 (0b0111), mant=0 → 0x38
WEIGHT_BYTE = 0x38
ACT_BYTE    = 0x38

# BF16 encoding for +8.0:  float32 = 0x41000000, upper 16 bits = 0x4100
BIAS_BF16 = 0x4100

# E8M0 scale register value: 124 → 2^(124-127) = 2^(-3)
SELI_VAL  = 0x7C
SCALE_EXP = SELI_VAL - 127  # -3


def float_to_bits32(x: float) -> int:
    return struct.unpack(">I", struct.pack(">f", x))[0]


def float_to_bf16_bits(x: float) -> int:
    """Round float32 to BF16 using round-to-nearest-even."""
    bits = float_to_bits32(x)
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    return ((bits + rounding_bias) >> 16) & 0xFFFF


def e4m3_to_float(byte: int) -> float:
    """Decode E4M3 to float (hardware semantics: subnormals and NaN flush to 0)."""
    byte &= 0xFF
    sign = -1.0 if (byte & 0x80) else 1.0
    exp  = (byte >> 3) & 0xF
    mant = byte & 0x7
    if exp == 0:
        return 0.0
    if exp == 0xF and mant == 0x7:
        return 0.0
    return sign * (1.0 + mant / 8.0) * (2.0 ** (exp - 7))


def encode_e4m3_nearest(x: float) -> int:
    """Encode float to nearest E4M3 byte (round-to-nearest-even on ties)."""
    if math.isnan(x):
        return 0x7F
    if x == 0.0:
        return 0x80 if math.copysign(1.0, x) < 0 else 0x00
    candidates = []
    for b in range(256):
        v = e4m3_to_float(b)
        if not math.isnan(v):
            candidates.append((b, v))
    best_b, best_err = 0, float("inf")
    for b, v in candidates:
        err = abs(v - x)
        if err < best_err:
            best_err = err
            best_b = b
        elif err == best_err and (b & 1) == 0 and (best_b & 1) != 0:
            best_b = b
    return best_b


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

    # Weights: 32 beats of 32 × 0x38
    weight_word = pack_32_bytes_le([WEIGHT_BYTE] * 32)
    for w in range(WORDS_PER_TILE):
        preloads.append({
            "word_offset": WEIGHT_WORD_START + w,
            "data": weight_word,
        })

    # Activations: 32 beats of 32 × 0x38
    act_word = pack_32_bytes_le([ACT_BYTE] * 32)
    for w in range(WORDS_PER_TILE):
        preloads.append({
            "word_offset": ACT_WORD_START + w,
            "data": act_word,
        })

    # Bias lo (m6, lanes 0–15): 32 beats of 16 × BF16(8.0)
    bias_word = pack_16_bf16_le([BIAS_BF16] * BF16S_PER_BEAT)
    for w in range(WORDS_PER_TILE):
        preloads.append({
            "word_offset": BIAS_LO_WORD_START + w,
            "data": bias_word,
        })

    # Bias hi (m7, lanes 16–31): 32 beats of 16 × BF16(8.0)
    for w in range(WORDS_PER_TILE):
        preloads.append({
            "word_offset": BIAS_HI_WORD_START + w,
            "data": bias_word,
        })

    return preloads


def make_checks():
    a = e4m3_to_float(ACT_BYTE)     # 1.0
    w = e4m3_to_float(WEIGHT_BYTE)  # 1.0
    bias = struct.unpack(">f", struct.pack(">I", BIAS_BF16 << 16))[0]  # 8.0

    matmul_val = sum(a * w for _ in range(TILE))  # 32.0
    quant_scale = 2.0 ** SCALE_EXP                # 0.125

    # All 32 lanes now have bias = 8.0
    acc = matmul_val + bias  # 40.0
    e4m3_out = encode_e4m3_nearest(acc * quant_scale)  # 5.0 → 0x4A

    assert e4m3_to_float(e4m3_out) == 5.0, f"expected 5.0, got {e4m3_to_float(e4m3_out)}"

    # Each output beat = one row: all 32 bytes = 0x4A
    out_word = pack_32_bytes_le([e4m3_out] * 32)

    checks = []
    for row in range(WORDS_PER_TILE):
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
    
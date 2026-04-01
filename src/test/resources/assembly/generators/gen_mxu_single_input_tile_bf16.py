#!/usr/bin/env python3
"""
gen_mxu_single_input_tile_bf16.py — FP8-input single-tile matmul computing C = A @ B.

For the 32×32 case, this is one full MXU tile:
  1. DMA loads B (FP8) into bank 4
  2. XLU transposes bank 4 → bank 5
  3. VMATPUSH loads bank 5 (= B^T) into MXU weight buffer
     Weight lane j = B^T[j,:] = B[:,j]
  4. DMA loads A (FP8) into bank 0
  5. VMATMUL computes:
       output[r][j] = dot(A[r,:], B[:,j]) = (A @ B)[r][j]
  6. VMATPOP.BF16 writes lanes 0-15 to bank 2 and lanes 16-31 to bank 3

Packing convention:
  FP8:  element k → byte k (bit position k*8). Matches unpackRow.
  BF16: lane k → bit position k*16. Cat(x.reverse) puts lane 0 at LSB.
  XLU:  byte-level transpose = element-level transpose.

DRAM layout (256-bit entries, 32 bytes each):
  A (FP8):   entries 0..31
  B (FP8):   entries 32..63
  C bank2:   entries 64..95   (BF16 lanes 0-15)
  C bank3:   entries 96..127  (BF16 lanes 16-31)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (
    rand_matrix_fp8_safe,
    quantize_fp8,
    quantize_bf16,
    matrix_to_fp8_words,
    matrix_to_bf16_words,
    pack_words_into_beats,
    emit_test_data,
)

TILE = int(sys.argv[1]) if len(sys.argv) > 1 else 32
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42

NUM_LANES = 32
WORDS_PER_BEAT = 8

if TILE != 32:
    raise ValueError(f"This single-tile test expects TILE=32, got TILE={TILE}")

# Generate FP8-safe random inputs
A = rand_matrix_fp8_safe(TILE, TILE, seed=SEED)
B = rand_matrix_fp8_safe(TILE, TILE, seed=SEED + 1)

# Quantize inputs to FP8, compute reference, then quantize output to BF16
A_q = quantize_fp8(A)
B_q = quantize_fp8(B)
C = quantize_bf16(A_q @ B_q)

# --- Pack FP8 inputs ---
# For 32×32, no padding is needed: each row is exactly 32 FP8 elements = 32 bytes.
a_words = matrix_to_fp8_words(A_q)   # 32 rows × 8 words/row = 256 words
b_words = matrix_to_fp8_words(B_q)

a_beats = pack_words_into_beats(a_words, WORDS_PER_BEAT)  # 32 beats
b_beats = pack_words_into_beats(b_words, WORDS_PER_BEAT)  # 32 beats

# --- Pack BF16 output split across bank 2 and bank 3 ---
# Each output row has 32 BF16 lanes:
#   bank 2 stores lanes 0..15  -> 16 BF16 values = 32 bytes = 1 beat/row
#   bank 3 stores lanes 16..31 -> 16 BF16 values = 32 bytes = 1 beat/row
c_bank2_words = []
c_bank3_words = []

for r in range(TILE):
    row_lo = C[r, 0:16]    # lanes 0-15
    row_hi = C[r, 16:32]   # lanes 16-31

    c_bank2_words.extend(matrix_to_bf16_words(row_lo.reshape(1, -1)))
    c_bank3_words.extend(matrix_to_bf16_words(row_hi.reshape(1, -1)))

c_bank2_beats = pack_words_into_beats(c_bank2_words, WORDS_PER_BEAT)  # 32 beats
c_bank3_beats = pack_words_into_beats(c_bank3_words, WORDS_PER_BEAT)  # 32 beats

# --- DRAM entry assignments ---
A_BASE = 0
B_BASE = A_BASE + len(a_beats)               # 32
C_BANK2_BASE = B_BASE + len(b_beats)         # 64
C_BANK3_BASE = C_BANK2_BASE + len(c_bank2_beats)  # 96

preloads = []
for i, beat in enumerate(a_beats):
    preloads.append({
        "word_offset": A_BASE + i,
        "data": f"0x{beat:X}",
    })

for i, beat in enumerate(b_beats):
    preloads.append({
        "word_offset": B_BASE + i,
        "data": f"0x{beat:X}",
    })

checks = []
for i, beat in enumerate(c_bank2_beats):
    checks.append({
        "word_offset": C_BANK2_BASE + i,
        "expected": f"0x{beat:X}",
    })

for i, beat in enumerate(c_bank3_beats):
    checks.append({
        "word_offset": C_BANK3_BASE + i,
        "expected": f"0x{beat:X}",
    })

emit_test_data(preloads, checks, timeout=200000)

"""
gen_utils.py — Shared utilities for Atlas assembly test data generators.

PACKING CONVENTION (matches hardware):
  Hardware uses Cat(x.reverse) which places element 0 at the LSB.
  unpackRow reads element i from bits [(i+1)*w-1 : i*w].
  Therefore all pack functions must put element 0 at the LOWEST bits.

  FP8:  pack_fp8x4(a, b, c, d) → a at [7:0], b at [15:8], c at [23:16], d at [31:24]
  BF16: pack_bf16_pair(a, b)   → a at [15:0], b at [31:16]
"""

import ctypes
import json
import os
import struct
import subprocess
import sys
import tempfile
import textwrap
import numpy as np
from typing import List, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Global DRAM/DMA packing configuration
# ---------------------------------------------------------------------------

# Atlas test harness DRAM is indexed in DMA-width beats.
# widthBytes = 32: 1 DRAM entry = 32 bytes = 256 bits = 8 × 32-bit words
DMA_WIDTH_BYTES = 32
WORDS_PER_BEAT = DMA_WIDTH_BYTES // 4

# ---------------------------------------------------------------------------
# BF16 helpers
# ---------------------------------------------------------------------------

def float_to_bf16_bits(f: float) -> int:
    """Convert a Python float to its bfloat16 bit representation (16 bits)."""
    fp32_bits = struct.unpack('>I', struct.pack('>f', float(f)))[0]
    rounding_bias = ((fp32_bits >> 16) & 1) + 0x7FFF
    bf16_bits = (fp32_bits + rounding_bias) >> 16
    return bf16_bits & 0xFFFF

def bf16_bits_to_float(bits: int) -> float:
    """Convert bfloat16 bit pattern back to Python float."""
    fp32_bits = (bits & 0xFFFF) << 16
    return struct.unpack('>f', struct.pack('>I', fp32_bits))[0]

def pack_bf16_pair(first: float, second: float) -> int:
    """Pack two bf16 values: first at [15:0], second at [31:16].
    Matches hardware Cat(x.reverse) where lane 0 is at LSB."""
    return ((float_to_bf16_bits(second) & 0xFFFF) << 16) | (float_to_bf16_bits(first) & 0xFFFF)

def unpack_bf16_pair(word: int) -> Tuple[float, float]:
    """Unpack 32-bit word: first from [15:0], second from [31:16]."""
    first  = bf16_bits_to_float(word & 0xFFFF)
    second = bf16_bits_to_float((word >> 16) & 0xFFFF)
    return first, second

# ---------------------------------------------------------------------------
# FP8 E4M3 helpers (range ±240, no inf, NaN = 0x7F / 0xFF)
# ---------------------------------------------------------------------------

def float_to_fp8_e4m3_bits(f: float) -> int:
    """Convert float to FP8 E4M3 bit pattern (8 bits)."""
    if f != f:  # NaN
        return 0x7F
    sign = 0
    if f < 0:
        sign = 1
        f = -f
    if f == 0.0:
        return sign << 7
    if f >= 240.0:
        return (sign << 7) | 0x7E
    exp = int(np.floor(np.log2(f))) if f > 0 else -7
    exp = max(exp, -6)
    exp = min(exp, 8)
    biased_exp = exp + 7
    if biased_exp <= 0:
        mantissa = f / (2.0 ** (-6))
        mantissa_bits = int(round(mantissa * 8)) & 0x07
        biased_exp = 0
    else:
        mantissa = f / (2.0 ** exp) - 1.0
        mantissa_bits = int(round(mantissa * 8)) & 0x07
    result = (sign << 7) | ((biased_exp & 0xF) << 3) | mantissa_bits
    return result & 0xFF

def fp8_e4m3_bits_to_float(bits: int) -> float:
    """Convert FP8 E4M3 bit pattern to float."""
    sign = (bits >> 7) & 1
    exp  = (bits >> 3) & 0xF
    mant = bits & 0x7
    if exp == 0 and mant == 0:
        return -0.0 if sign else 0.0
    if exp == 0xF and mant == 0x7:
        return float('nan')
    if exp == 0:
        value = (mant / 8.0) * (2.0 ** (-6))
    else:
        value = (1.0 + mant / 8.0) * (2.0 ** (exp - 7))
    return -value if sign else value

def pack_fp8x4(a: float, b: float, c: float, d: float) -> int:
    """Pack four FP8 E4M3 values: a at [7:0], b at [15:8], c at [23:16], d at [31:24].
    Matches hardware unpackRow where element i = bits[(i+1)*8-1 : i*8]."""
    return ((float_to_fp8_e4m3_bits(d) << 24) |
            (float_to_fp8_e4m3_bits(c) << 16) |
            (float_to_fp8_e4m3_bits(b) << 8)  |
             float_to_fp8_e4m3_bits(a))

# ---------------------------------------------------------------------------
# FP8 E5M2 helpers (range ±57344, has inf/NaN)
# ---------------------------------------------------------------------------

def float_to_fp8_e5m2_bits(f: float) -> int:
    """Convert float to FP8 E5M2 bit pattern (8 bits)."""
    if f != f:
        return 0x7F
    sign = 0
    if f < 0:
        sign = 1
        f = -f
    if np.isinf(f):
        return (sign << 7) | 0x7C
    if f == 0.0:
        return sign << 7
    if f >= 57344.0:
        return (sign << 7) | 0x7B
    exp = int(np.floor(np.log2(f))) if f > 0 else -15
    exp = max(exp, -14)
    exp = min(exp, 15)
    biased_exp = exp + 15
    if biased_exp <= 0:
        mantissa = f / (2.0 ** (-14))
        mantissa_bits = int(round(mantissa * 4)) & 0x03
        biased_exp = 0
    else:
        mantissa = f / (2.0 ** exp) - 1.0
        mantissa_bits = int(round(mantissa * 4)) & 0x03
    return ((sign << 7) | ((biased_exp & 0x1F) << 2) | mantissa_bits) & 0xFF

# ---------------------------------------------------------------------------
# Matrix layout helpers
# ---------------------------------------------------------------------------

def matrix_to_bf16_words(mat: np.ndarray, row_major: bool = True) -> List[int]:
    """
    Flatten a 2D matrix into 32-bit words, each holding 2 bf16 values.
    Element at even index → bits[15:0], odd index → bits[31:16].
    """
    flat = mat.flatten() if row_major else mat.T.flatten()
    if len(flat) % 2 != 0:
        flat = np.append(flat, 0.0)
    words = []
    for i in range(0, len(flat), 2):
        words.append(pack_bf16_pair(flat[i], flat[i + 1]))
    return words

def bf16_bits_to_words(bits: np.ndarray, row_major: bool = True) -> List[int]:
    """
    Flatten a uint16 BF16-bit matrix into 32-bit words, each holding 2 BF16
    elements. The first element occupies bits[15:0], the second bits[31:16].
    """
    arr = np.asarray(bits, dtype=np.uint16)
    flat = arr.flatten() if row_major else arr.T.flatten()
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.uint16(0))
    return [
        int(flat[i]) | (int(flat[i + 1]) << 16)
        for i in range(0, len(flat), 2)
    ]

def matrix_to_fp8_words(mat: np.ndarray, row_major: bool = True) -> List[int]:
    """
    Flatten a 2D matrix into 32-bit words, each holding 4 FP8 E4M3 values.
    Element at index 4k+j → byte j of word k.
    """
    flat = mat.flatten() if row_major else mat.T.flatten()
    while len(flat) % 4 != 0:
        flat = np.append(flat, 0.0)
    words = []
    for i in range(0, len(flat), 4):
        words.append(pack_fp8x4(flat[i], flat[i + 1], flat[i + 2], flat[i + 3]))
    return words

def matrix_to_int32_words(mat: np.ndarray, row_major: bool = True) -> List[int]:
    """Flatten a 2D int32 matrix into a list of 32-bit words (1:1 mapping)."""
    flat = mat.flatten() if row_major else mat.T.flatten()
    return [int(v) & 0xFFFFFFFF for v in flat]

# ---------------------------------------------------------------------------
# DMA-beat packing helpers
# ---------------------------------------------------------------------------

def pack_words_into_beats(words: List[int], words_per_beat: int = WORDS_PER_BEAT) -> List[int]:
    """
    Pack a list of 32-bit words into DMA-width beats.
    Word 0 goes in bits[31:0] of the beat (lowest position).
    """
    if not words:
        return []
    padded = list(words)
    while len(padded) % words_per_beat != 0:
        padded.append(0)
    beats = []
    for i in range(0, len(padded), words_per_beat):
        beat = 0
        chunk = padded[i:i + words_per_beat]
        for j, w in enumerate(chunk):
            beat |= (int(w) & 0xFFFFFFFF) << (32 * j)
        beats.append(beat)
    return beats

def unpack_beats_into_words(beats: List[int], words_per_beat: int = WORDS_PER_BEAT) -> List[int]:
    """Unpack DMA-width beats back into 32-bit words."""
    words = []
    for beat in beats:
        for j in range(words_per_beat):
            words.append((int(beat) >> (32 * j)) & 0xFFFFFFFF)
    return words

# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def quantize_bf16(mat: np.ndarray) -> np.ndarray:
    """Round-trip a matrix through BF16 quantization."""
    return np.array([[bf16_bits_to_float(float_to_bf16_bits(v))
                      for v in row] for row in mat])

def quantize_fp8(mat: np.ndarray) -> np.ndarray:
    """Round-trip a matrix through FP8 E4M3 quantization."""
    return np.array([[fp8_e4m3_bits_to_float(float_to_fp8_e4m3_bits(v))
                      for v in row] for row in mat])

# ---------------------------------------------------------------------------
# Quantized matmul references (match MXU hardware behavior)
# ---------------------------------------------------------------------------

def bf16_matmul_reference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    BF16 matmul: quantize inputs → fp32 accumulate → quantize output.
    """
    a_q = quantize_bf16(a)
    b_q = quantize_bf16(b)
    c = a_q @ b_q
    return quantize_bf16(c)

def fp8_matmul_reference(a: np.ndarray, b: np.ndarray,
                         accumulate_bf16: bool = True) -> np.ndarray:
    """
    FP8 E4M3 matmul: quantize inputs → fp32 accumulate → optionally quantize.
    """
    a_q = quantize_fp8(a)
    b_q = quantize_fp8(b)
    c = a_q @ b_q
    if accumulate_bf16:
        c = quantize_bf16(c)
    return c

def mxu0_sa_bf16_bits(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Run the MXU0 systolic-array software model and return a 32x32 matrix of
    BF16 bit patterns for C = A @ B.
    """
    tile = 32
    if a.shape != (tile, tile) or b.shape != (tile, tile):
        raise ValueError(f"expected 32x32 tiles, got {a.shape} and {b.shape}")

    header_dir = os.path.join(
        os.path.dirname(__file__), "software_models", "mxu0_sa"
    )
    shim_src = textwrap.dedent(
        """\
        #include <stdint.h>
        #include <stdbool.h>
        #include <stdlib.h>
        #include <string.h>
        #include "fp_formats.h"
        #include "converters.h"
        #include "systolic_array_model.h"
        #include "systolic_array_linear.h"

        void sa_bf16_matmul_32x32(
                const uint8_t *x_e4m3,
                const uint8_t *w_e4m3,
                uint16_t *out_bits)
        {
            sa_linear_init_luts();
            SystolicArrayParams p;
            p.rows = 32;
            p.cols = 32;
            sa_linear_call(
                &p,
                x_e4m3,
                w_e4m3,
                NULL,
                32,
                32,
                32,
                0,
                OutputFmtSel_OutBF16,
                out_bits);
        }
        """
    )

    with tempfile.TemporaryDirectory(prefix="atlas_sa_ref_") as build_dir:
        shim_c = os.path.join(build_dir, "sa_ref.c")
        shim_so = os.path.join(build_dir, "sa_ref.so")
        with open(shim_c, "w") as fh:
            fh.write(shim_src)
        cmd = [
            "gcc",
            "-O3",
            "-Wall",
            "-Wno-unused-function",
            "-shared",
            "-fPIC",
            f"-I{header_dir}",
            "-o",
            shim_so,
            shim_c,
            "-lm",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                "failed to compile MXU0 SA reference shim\n"
                f"command: {' '.join(cmd)}\n"
                f"stderr:\n{result.stderr}"
            )

        lib = ctypes.CDLL(shim_so)
        lib.sa_bf16_matmul_32x32.restype = None
        lib.sa_bf16_matmul_32x32.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint16),
        ]

        x = np.ascontiguousarray(
            [[float_to_fp8_e4m3_bits(float(v)) for v in row] for row in a],
            dtype=np.uint8,
        )
        # sa_linear_call follows F.linear: y = x @ w^T, so pass B^T.
        w = np.ascontiguousarray(
            [[float_to_fp8_e4m3_bits(float(v)) for v in row] for row in b.T],
            dtype=np.uint8,
        )
        out = np.empty((tile, tile), dtype=np.uint16)

        lib.sa_bf16_matmul_32x32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        )
        return out

# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def emit_test_data(preloads: List[Dict], checks: List[Dict],
                   timeout: Optional[int] = None):
    """Print test data as JSON to stdout for the Scala test runner."""
    output = {
        "dram_preloads": preloads,
        "dram_checks": checks,
    }
    if timeout is not None:
        output["timeout"] = timeout
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write('\n')

def make_preload(word_offset: int, data: int) -> Dict:
    """Legacy 32-bit preload entry."""
    return {"word_offset": word_offset, "data": f"0x{data & 0xFFFFFFFF:08X}"}

def make_check(word_offset: int, expected: int) -> Dict:
    """Legacy 32-bit check entry."""
    return {"word_offset": word_offset, "expected": f"0x{expected & 0xFFFFFFFF:08X}"}

def make_preload_any(word_offset: int, data: int) -> Dict:
    """Width-agnostic preload entry for DMA-width beats."""
    return {"word_offset": word_offset, "data": f"0x{int(data):X}"}

def make_check_any(word_offset: int, expected: int) -> Dict:
    """Width-agnostic check entry for DMA-width beats."""
    return {"word_offset": word_offset, "expected": f"0x{int(expected):X}"}

def preloads_from_words(base_offset: int, words: List[int]) -> List[Dict]:
    """Create legacy 32-bit preload entries."""
    return [make_preload(base_offset + i, w) for i, w in enumerate(words)]

def checks_from_words(base_offset: int, words: List[int]) -> List[Dict]:
    """Create legacy 32-bit check entries."""
    return [make_check(base_offset + i, w) for i, w in enumerate(words)]

def preloads_from_words_packed(base_offset: int, words: List[int],
                               words_per_beat: int = WORDS_PER_BEAT) -> List[Dict]:
    """Pack 32-bit words into DMA-width beats, then emit preload entries."""
    beats = pack_words_into_beats(words, words_per_beat)
    return [make_preload_any(base_offset + i, beat) for i, beat in enumerate(beats)]

def checks_from_words_packed(base_offset: int, words: List[int],
                             words_per_beat: int = WORDS_PER_BEAT) -> List[Dict]:
    """Pack 32-bit words into DMA-width beats, then emit check entries."""
    beats = pack_words_into_beats(words, words_per_beat)
    return [make_check_any(base_offset + i, beat) for i, beat in enumerate(beats)]

# ---------------------------------------------------------------------------
# Reproducible random matrices
# ---------------------------------------------------------------------------

def rand_matrix(rows: int, cols: int, low: float = -1.0, high: float = 1.0,
                seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.uniform(low, high, (rows, cols))

def rand_matrix_fp8_safe(rows: int, cols: int, seed: int = 42) -> np.ndarray:
    return rand_matrix(rows, cols, low=-2.0, high=2.0, seed=seed)

def identity_matrix(n: int) -> np.ndarray:
    return np.eye(n)

def zero_matrix(rows: int, cols: int) -> np.ndarray:
    return np.zeros((rows, cols))
    

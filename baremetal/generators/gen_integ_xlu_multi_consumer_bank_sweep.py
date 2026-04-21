#!/usr/bin/env python3
import ctypes
import os
import subprocess
import sys
import tempfile
import textwrap
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from gen_utils import (
    emit_test_data,
    preloads_from_words_packed,
    checks_from_words_packed,
    matrix_to_fp8_words,
    quantize_fp8,
    float_to_fp8_e4m3_bits,
    rand_matrix_fp8_safe,
)

TILE = 32


def matrix_to_e4m3_bytes(mat):
    flat = mat.reshape(-1)
    data = [float_to_fp8_e4m3_bits(float(v)) for v in flat]
    return np.ascontiguousarray(data, dtype=np.uint8).reshape(mat.shape)


def bf16_bits_to_words(bits):
    flat = bits.reshape(-1)
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.uint16(0))
    return [
        int(flat[i]) | (int(flat[i + 1]) << 16)
        for i in range(0, len(flat), 2)
    ]


def mxu0_sa_bf16_bits(a, b):
    """Run the MXU0 systolic-array reference for C = A @ B."""
    if a.shape != (TILE, TILE) or b.shape != (TILE, TILE):
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

        x = matrix_to_e4m3_bytes(a)
        # sa_linear_call follows F.linear: y = x @ w^T, so pass B^T.
        w = matrix_to_e4m3_bytes(b.T.copy())
        out = np.empty((TILE, TILE), dtype=np.uint16)

        lib.sa_bf16_matmul_32x32(
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            w.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        )
        return out


A0 = quantize_fp8(rand_matrix_fp8_safe(32, 32, seed=101)).astype(np.float32)
B0 = quantize_fp8(rand_matrix_fp8_safe(32, 32, seed=102)).astype(np.float32)
A1 = quantize_fp8(rand_matrix_fp8_safe(32, 32, seed=103)).astype(np.float32)
B1 = quantize_fp8(rand_matrix_fp8_safe(32, 32, seed=104)).astype(np.float32)

# Assembly transposes B before pushing as MXU weight, so result is A0 @ B0.
# Use the SA model because MXU0 rounds through the PE chain, not as one fp32 dot.
C0_bits = mxu0_sa_bf16_bits(A0, B0)
B1_t = quantize_fp8(B1.T).astype(np.float32)

preloads = []
checks = []

preloads += preloads_from_words_packed(0x0000 // 32, matrix_to_fp8_words(A0))
preloads += preloads_from_words_packed(0x0400 // 32, matrix_to_fp8_words(B0))
preloads += preloads_from_words_packed(0x0800 // 32, matrix_to_fp8_words(A1))
preloads += preloads_from_words_packed(0x0C00 // 32, matrix_to_fp8_words(B1))

checks += checks_from_words_packed(
    0x2000 // 32, bf16_bits_to_words(C0_bits[:, :16])
)
checks += checks_from_words_packed(
    0x2400 // 32, bf16_bits_to_words(C0_bits[:, 16:])
)
checks += checks_from_words_packed(0x2800 // 32, matrix_to_fp8_words(B1_t))

emit_test_data(preloads, checks, timeout=500000)

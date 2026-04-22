"""
systolic_array_rtl_linear_c.py
------------------
C-accelerated systolic-array linear function.
"""

from __future__ import annotations

import atexit
import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import textwrap
from typing import Optional, Tuple

import numpy as np
import torch

from software_models.mxu1_ipt.ipt_c.fp_formats import OutputFmtSel
from software_models.mxu1_ipt.ipt_c.ipt_rtl_linear_c import (
    float_to_e4m3_bytes_c,
    _tensor_cache_key,
    _tensor_to_f32_numpy,
)
from software_models.mxu1_ipt.ipt_rtl_linear import decode_model_output_bits

# ---------------------------------------------------------------------------
# C shim source
# ---------------------------------------------------------------------------

_SHIM_C = textwrap.dedent(
    """\
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "fp_formats.h"
#include "converters.h"
#include "systolic_array_model.h"
#include "systolic_array_linear.h"

void sa_shim_init(void) {
    atlas_fp_init_lut();
    atlas_acc_init_lut();
}

void sa_shim_linear_call(
        int rows, int cols,
        const uint8_t  *x_e4m3,
        const uint8_t  *w_e4m3,
        const uint8_t  *b_e4m3,
        int batch, int in_features, int out_features,
        int scale_exp, int out_fmt_sel,
        uint16_t *out_bits)
{
    SystolicArrayParams p;
    p.rows = rows;
    p.cols = cols;

    sa_linear_call(
        &p,
        x_e4m3, w_e4m3, b_e4m3,
        batch, in_features, out_features,
        scale_exp,
        (OutputFmtSel)out_fmt_sel,
        out_bits);
}
"""
)

# ---------------------------------------------------------------------------
# Module - level singleton
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None
_build_dir: Optional[str] = None
_lib_hash: Optional[str] = None


def _shim_hash() -> str:
    return hashlib.md5(_SHIM_C.encode()).hexdigest()


def _cleanup() -> None:
    global _lib, _build_dir, _lib_hash
    _lib = _lib_hash = None
    if _build_dir and os.path.isdir(_build_dir):
        shutil.rmtree(_build_dir, ignore_errors=True)
    _build_dir = None


atexit.register(_cleanup)


def _get_lib() -> ctypes.CDLL:
    global _lib, _build_dir, _lib_hash

    current = _shim_hash()
    if _lib is not None and _lib_hash == current:
        return _lib
    if _lib is not None:
        _cleanup()

    header_dir = os.environ.get(
        "SA_HEADER_DIR",
        os.path.dirname(os.path.abspath(__file__)),
    )

    _build_dir = tempfile.mkdtemp(prefix="sa_linear_c_")
    shim_c = os.path.join(_build_dir, "sa_shim.c")
    shim_so = os.path.join(_build_dir, "libsa_linear.so")

    with open(shim_c, "w") as fh:
        fh.write(_SHIM_C)

    cmd = [
        "gcc",
        "-O3",
        "-march=native",
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
        _cleanup()
        raise RuntimeError(
            "sa_rtl_linear_c: failed to compile C shim.\n"
            f"  SA_HEADER_DIR = {header_dir!r}\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  stderr:\n{result.stderr}"
        )

    lib = ctypes.CDLL(shim_so)

    lib.sa_shim_init.restype = None
    lib.sa_shim_init.argtypes = []

    lib.sa_shim_linear_call.restype = None
    lib.sa_shim_linear_call.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint16),
    ]

    lib.sa_shim_init()
    _lib = lib
    _lib_hash = current
    return _lib


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class SARTLLinearFunction:
    def __init__(
        self,
        rows: int = 32,
        cols: int = 32,
        out_fmt_sel: OutputFmtSel = OutputFmtSel.OutBF16,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.out_fmt_sel = out_fmt_sel

        self._w_cache_key: Optional[tuple] = None
        self._b_cache_key: Optional[tuple] = None
        self._w_np: Optional[np.ndarray] = None
        self._b_np: Optional[np.ndarray] = None
        self._w_ref: Optional[torch.Tensor] = None
        self._b_ref: Optional[torch.Tensor] = None

        _get_lib()

    def _prepare_weights(
        self,
        w_q: torch.Tensor,
        b_q: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        w_key = _tensor_cache_key(w_q)
        b_key = _tensor_cache_key(b_q) if b_q is not None else None

        if w_key != self._w_cache_key or b_key != self._b_cache_key:
            self._w_np = float_to_e4m3_bytes_c(_tensor_to_f32_numpy(w_q))
            self._b_np = (
                float_to_e4m3_bytes_c(_tensor_to_f32_numpy(b_q))
                if b_q is not None
                else None
            )
            self._w_cache_key = w_key
            self._b_cache_key = b_key
            # Keep the cached tensors alive. PyTorch may otherwise recycle a
            # freed temporary's data_ptr for a different tensor, causing a
            # false cache hit and stale quantized weights.
            self._w_ref = w_q
            self._b_ref = b_q

        return self._w_np, self._b_np

    def _call_c(
        self,
        x_np: np.ndarray,
        w_np: np.ndarray,
        b_np: Optional[np.ndarray],
        batch: int,
        in_features: int,
        out_features: int,
        scale_exp: int,
    ) -> torch.Tensor:
        lib = _get_lib()

        # CRITICAL: Ensure arrays are contiguous in memory before casting to pointers
        x_np = np.ascontiguousarray(x_np, dtype=np.uint8)
        w_np = np.ascontiguousarray(w_np, dtype=np.uint8)

        x_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        w_ptr = w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        if b_np is not None:
            b_np = np.ascontiguousarray(b_np, dtype=np.uint8)
            b_ptr = b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        else:
            b_ptr = ctypes.cast(None, ctypes.POINTER(ctypes.c_uint8))

        out_np = np.empty(batch * out_features, dtype=np.uint16)
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

        lib.sa_shim_linear_call(
            ctypes.c_int(self.rows),
            ctypes.c_int(self.cols),
            x_ptr,
            w_ptr,
            b_ptr,
            ctypes.c_int(batch),
            ctypes.c_int(in_features),
            ctypes.c_int(out_features),
            ctypes.c_int(scale_exp),
            ctypes.c_int(int(self.out_fmt_sel.value)),
            out_ptr,
        )

        # Return just the raw bits. Decoding happens in __call__
        return torch.from_numpy(out_np).to(torch.int32).reshape(batch, out_features)

    def __call__(
        self,
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        b_q: Optional[torch.Tensor] = None,
        scale_exp: int = 0,
        return_bits: bool = False,
    ) -> torch.Tensor:

        original_shape = x_q.shape[:-1]
        in_features = x_q.shape[-1]
        out_features = w_q.shape[0]

        x2 = x_q.reshape(-1, in_features)
        batch = x2.shape[0]

        # Bypass float conversion if data is already passed as raw E4M3 bytes
        if x_q.dtype == torch.uint8:
            x_np = x2.contiguous().numpy()
            w_np = w_q.contiguous().numpy()
            b_np = b_q.contiguous().numpy() if b_q is not None else None
        else:
            x_np = float_to_e4m3_bytes_c(_tensor_to_f32_numpy(x2.float()))
            w_np, b_np = self._prepare_weights(w_q, b_q)

        # y_bits is [batch, out_features]
        y_bits = self._call_c(
            x_np, w_np, b_np, batch, in_features, out_features, scale_exp
        )

        if return_bits:
            return y_bits.reshape(*original_shape, out_features)

        y = decode_model_output_bits(y_bits, self.out_fmt_sel)
        return y.reshape(*original_shape, out_features)

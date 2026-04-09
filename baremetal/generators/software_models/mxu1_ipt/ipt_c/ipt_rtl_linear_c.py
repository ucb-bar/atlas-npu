"""
ipt_rtl_linear_c.py
-------------------
C-accelerated drop-in replacement for IPTLinearRTLFunction.

On first import the module compiles ipt_linear.h (and its dependencies)
into a shared library via gcc and caches the ctypes handle as a
module-level singleton.  Subsequent imports and all QuantLinearC instances
share the same .so.

Performance notes
-----------------
Three sources of overhead that would otherwise nullify the C speedup are
addressed here:

1. float_to_e4m3_bytes -- the original implementation is a Python scalar
   loop over every element.  This module exposes shim_float_to_e4m3 from
   the compiled C shim, which runs the same logic as a single tight C loop
   with no Python overhead.  float_to_e4m3_bytes_torch (a torch-vectorized
   fallback) and float_to_e4m3_bytes_numba (a Numba parallel JIT fallback)
   are also exported for benchmarking.

2. Weight re-quantization -- weights never change between forward calls, but
   without caching they would be re-quantized on every call.
   CIPTLinearRTLFunction caches the quantized weight and bias numpy arrays
   keyed on the weight tensor's data_ptr + version counter, so the cost is
   paid only when weights actually change.

3. numpy round-trip -- activations are quantized in C directly from a
   contiguous float32 numpy view, avoiding an intermediate uint8 tensor.

Required environment variable
------------------------------
    IPT_HEADER_DIR   Directory containing all C headers:
                         fp_formats.h
                         converters.h
                         params_and_requests.h
                         inner_product_trees_model.h
                         ipt_linear.h
                     Defaults to the directory of this file (ipt_mxu_model/).

The compiled .so is written to a per-process temp directory and cleaned up
on interpreter exit.
"""

from __future__ import annotations

import atexit
import ctypes
import math
import os
import shutil
import subprocess
import tempfile
import textwrap
from typing import Optional, Tuple

import numpy as np
import torch

from .fp_formats import OutputFmtSel

# ---------------------------------------------------------------------------
# Vectorized torch fallback
# ---------------------------------------------------------------------------


def float_to_e4m3_bytes_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Vectorized float32 -> E4M3 byte quantization using only torch ops.
    Matches _float_to_e4m3_byte_scalar exactly, returns uint8 tensor.
    """
    x = x.float()
    sign = (x < 0).to(torch.int32)
    absx = x.abs()

    is_nan = torch.isnan(absx)
    is_inf = torch.isinf(absx)
    is_zero = (absx == 0.0) | is_nan
    flush = (absx < 2.0**-6) & ~is_zero & ~is_inf
    saturate = is_inf | (absx >= 496.0)

    safe = absx.clamp(min=2.0**-6, max=495.999)
    exp_f = torch.floor(torch.log2(safe)).to(torch.int32).clamp(-6, 8)
    pow2 = torch.pow(2.0, exp_f.float())
    frac = torch.round((safe / pow2 - 1.0) * 8.0).to(torch.int32)

    carry = (frac >= 8).to(torch.int32)
    frac = frac * (1 - carry)
    exp_f = (exp_f + carry).clamp(-6, 9)
    saturate = saturate | (exp_f > 8)
    exp_f = exp_f.clamp(-6, 8)

    exp_field = (exp_f + 7) & 0xF
    result = (sign << 7) | (exp_field << 3) | frac.clamp(0, 7)

    result = result.masked_fill(flush | is_zero, 0)
    result = result.masked_fill(saturate & (sign == 0), 0x7E)
    result = result.masked_fill(saturate & (sign == 1), 0xFE)
    result = result.masked_fill(is_nan, 0)

    return result.to(torch.uint8)


# ---------------------------------------------------------------------------
# Numba JIT implementation
# ---------------------------------------------------------------------------

try:
    import numba

    @numba.njit(cache=True)
    def _e4m3_scalar_nb(v: float) -> int:
        """Scalar E4M3 encoder with banker's rounding, compiled by Numba."""
        if v != v:  # NaN
            return 0
        sign = 1 if v < 0.0 else 0
        a = -v if sign else v
        if a == 0.0 or a < 0.015625:  # zero or below 2^-6
            return 0
        if a >= 496.0 or math.isinf(a):
            return 0xFE if sign else 0x7E

        exp = int(math.floor(math.log2(a)))
        if exp < -6:
            exp = -6
        if exp > 8:
            exp = 8

        pow2 = 2.0**exp
        frac_f = (a / pow2 - 1.0) * 8.0

        # Banker's rounding (round-half-to-even) to match Python's round()
        fl = math.floor(frac_f)
        diff = frac_f - fl
        if diff > 0.5:
            frac = int(fl) + 1
        elif diff < 0.5:
            frac = int(fl)
        else:
            frac = int(fl) if int(fl) % 2 == 0 else int(fl) + 1

        if frac >= 8:
            frac = 0
            exp += 1
        if exp > 8:
            return 0xFE if sign else 0x7E
        if exp < -6:
            return 0

        return (sign << 7) | (((exp + 7) & 0xF) << 3) | (frac & 0x7)

    @numba.njit(cache=True)
    def _float_to_e4m3_numba_kernel_serial(flat: np.ndarray) -> np.ndarray:
        out = np.empty(flat.size, dtype=np.uint8)
        for i in range(flat.size):
            out[i] = _e4m3_scalar_nb(flat[i])
        return out

    @numba.njit(parallel=True, cache=True)
    def _float_to_e4m3_numba_kernel_parallel(flat: np.ndarray) -> np.ndarray:
        out = np.empty(flat.size, dtype=np.uint8)
        for i in numba.prange(flat.size):
            out[i] = _e4m3_scalar_nb(flat[i])
        return out

    def float_to_e4m3_bytes_numba_serial(x: np.ndarray) -> np.ndarray:
        # Serial JIT: what @njit alone gives, without multi-core parallelism.
        flat = np.ascontiguousarray(x.ravel(), dtype=np.float32)
        return _float_to_e4m3_numba_kernel_serial(flat).reshape(x.shape)

    def float_to_e4m3_bytes_numba(x: np.ndarray) -> np.ndarray:
        # Parallel JIT: all available CPU cores via numba.prange.
        # cache=True persists compiled bitcode to __pycache__.
        flat = np.ascontiguousarray(x.ravel(), dtype=np.float32)
        return _float_to_e4m3_numba_kernel_parallel(flat).reshape(x.shape)

    NUMBA_AVAILABLE = True

except ImportError:
    NUMBA_AVAILABLE = False

    def float_to_e4m3_bytes_numba_serial(x: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        raise ImportError("numba is not installed. Run: pip install numba")

    def float_to_e4m3_bytes_numba(x: np.ndarray) -> np.ndarray:  # type: ignore[misc]
        raise ImportError("numba is not installed. Run: pip install numba")


# ---------------------------------------------------------------------------
# C shim source
# ---------------------------------------------------------------------------

_SHIM_C = textwrap.dedent(
    """\
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "fp_formats.h"
#include "converters.h"
#include "params_and_requests.h"
#include "inner_product_trees_model.h"
#include "ipt_linear.h"

/* 256-entry E4M3 -> float32 decode LUT. Built once by shim_init(). */
static float _E4M3_FLOAT_LUT[256];

static void _build_e4m3_float_lut(void)
{
    for (int i = 0; i < 256; i++) {
        int sign      = (i >> 7) & 1;
        int exp_field = (i >> 3) & 0xF;
        int frac      = i & 0x7;
        float val;
        if (exp_field == 0) {
            val = (frac == 0)
                ? 0.0f
                : (float)frac / 8.0f * ldexpf(1.0f, 1 - 7);
        } else if (exp_field == 0xF) {
            val = 0.0f;
        } else {
            val = (1.0f + (float)frac / 8.0f) * ldexpf(1.0f, exp_field - 7);
        }
        _E4M3_FLOAT_LUT[i] = sign ? -val : val;
    }
}

void shim_init(void) {
    ipt_linear_init_luts();
    _build_e4m3_float_lut();
}

/*
 * shim_float_to_e4m3: float32[n] -> uint8[n].
 * Matches _float_to_e4m3_byte_scalar exactly (double log2, banker's rounding).
 */
void shim_float_to_e4m3(const float *src, uint8_t *dst, int n)
{
    for (int i = 0; i < n; i++) {
        float v = src[i];

        if (v != v) { dst[i] = 0; continue; }

        int   sign = (v < 0.0f) ? 1 : 0;
        float a    = sign ? -v : v;

        if (isinf(a) || a >= 496.0f) {
            dst[i] = sign ? (uint8_t)0xFE : (uint8_t)0x7E;
            continue;
        }

        if (a < 0.015625f || a == 0.0f) {
            dst[i] = 0;
            continue;
        }

        int   exp  = (int)floor(log2((double)a));
        if (exp < -6) exp = -6;
        if (exp >  8) exp =  8;

        double pow2   = ldexp(1.0, exp);
        double frac_f = ((double)a / pow2 - 1.0) * 8.0;

        int frac;
        {
            double fl   = floor(frac_f);
            double diff = frac_f - fl;
            if (diff > 0.5)
                frac = (int)fl + 1;
            else if (diff < 0.5)
                frac = (int)fl;
            else
                frac = ((int)fl % 2 == 0) ? (int)fl : (int)fl + 1;
        }

        if (frac >= 8) { frac = 0; exp += 1; }

        if (exp >  8) { dst[i] = sign ? (uint8_t)0xFE : (uint8_t)0x7E; continue; }
        if (exp < -6) { dst[i] = 0; continue; }

        int exp_field = (exp + 7) & 0xF;
        dst[i] = (uint8_t)((sign << 7) | (exp_field << 3) | (frac & 0x7));
    }
}

/*
 * shim_e4m3_to_float: uint8[n] -> float32[n].
 * Pure LUT lookup, branchless, O(n).
 */
void shim_e4m3_to_float(const uint8_t *src, float *dst, int n)
{
    for (int i = 0; i < n; i++)
        dst[i] = _E4M3_FLOAT_LUT[src[i]];
}

void shim_ipt_linear_call(
        int num_lanes, int vec_len, int pipeline_depth,
        const uint8_t  *x_e4m3,
        const uint8_t  *w_e4m3,
        const uint8_t  *b_e4m3,
        int batch, int in_features, int out_features,
        int scale_exp, int out_fmt_sel,
        uint16_t *out_bits)
{
    InnerProductTreeParams base;
    base.numLanes      = num_lanes;
    base.vecLen        = vec_len;
    base.accumIntWidth = 0;
    base.pipelineCuts  = 0x00;

    InnerProductTreeParams p = IPT_withPipelineDepth(pipeline_depth, &base);

    ipt_linear_call(
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
# Module-level singleton state
# ---------------------------------------------------------------------------

_lib: Optional[ctypes.CDLL] = None
_build_dir: Optional[str] = None
_lib_shim_hash: Optional[str] = None


def _shim_hash() -> str:
    import hashlib

    return hashlib.md5(_SHIM_C.encode()).hexdigest()


def _cleanup() -> None:
    global _lib, _build_dir, _lib_shim_hash
    _lib = None
    _lib_shim_hash = None
    if _build_dir and os.path.isdir(_build_dir):
        shutil.rmtree(_build_dir, ignore_errors=True)
    _build_dir = None


atexit.register(_cleanup)


def _get_lib() -> ctypes.CDLL:
    """Return the cached ctypes handle, compiling on first call.
    Recompiles automatically if _SHIM_C has changed since the last build.
    """
    global _lib, _build_dir, _lib_shim_hash

    current_hash = _shim_hash()
    if _lib is not None and _lib_shim_hash == current_hash:
        return _lib

    if _lib is not None:
        _cleanup()

    header_dir = os.environ.get(
        "IPT_HEADER_DIR",
        os.path.dirname(os.path.abspath(__file__)),
    )

    _build_dir = tempfile.mkdtemp(prefix="ipt_linear_c_")
    shim_c = os.path.join(_build_dir, "ipt_linear_shim.c")
    shim_so = os.path.join(_build_dir, "libipt_linear.so")

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
            "ipt_rtl_linear_c: failed to compile C shim.\n"
            f"  IPT_HEADER_DIR = {header_dir!r}\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  stderr:\n{result.stderr}"
        )

    lib = ctypes.CDLL(shim_so)

    lib.shim_init.restype = None
    lib.shim_init.argtypes = []

    lib.shim_float_to_e4m3.restype = None
    lib.shim_float_to_e4m3.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int,
    ]

    lib.shim_e4m3_to_float.restype = None
    lib.shim_e4m3_to_float.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # src: E4M3 bytes
        ctypes.POINTER(ctypes.c_float),  # dst: float32
        ctypes.c_int,  # n
    ]

    lib.shim_ipt_linear_call.restype = None
    lib.shim_ipt_linear_call.argtypes = [
        ctypes.c_int,
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

    lib.shim_init()
    _lib = lib
    _lib_shim_hash = current_hash
    return _lib


# ---------------------------------------------------------------------------
# C-backed float_to_e4m3_bytes
# ---------------------------------------------------------------------------


def float_to_e4m3_bytes_c(x: np.ndarray) -> np.ndarray:
    """
    Quantize a contiguous float32 numpy array to E4M3 uint8 in C.
    x must be float32, C-contiguous. Returns a uint8 array of the same shape.
    """
    lib = _get_lib()
    flat = x.ravel()
    n = flat.size
    dst = np.empty(n, dtype=np.uint8)

    src_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    lib.shim_float_to_e4m3(src_ptr, dst_ptr, ctypes.c_int(n))

    return dst.reshape(x.shape)


# ---------------------------------------------------------------------------
# C-backed e4m3_bytes_to_float
# ---------------------------------------------------------------------------


def e4m3_bytes_to_float_c(x: np.ndarray) -> np.ndarray:
    """
    Decode E4M3 uint8 bytes to float32 using a C LUT lookup.
    Equivalent to ipt_rtl_linear.e4m3_bytes_to_float but with no Python loop.
    x must be uint8. Returns a float32 array of the same shape.
    """
    lib = _get_lib()
    flat = np.ascontiguousarray(x.ravel(), dtype=np.uint8)
    dst = np.empty(flat.size, dtype=np.float32)

    src_ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    dst_ptr = dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    lib.shim_e4m3_to_float(src_ptr, dst_ptr, ctypes.c_int(flat.size))

    return dst.reshape(x.shape)


# ---------------------------------------------------------------------------
# Weight cache helpers
# ---------------------------------------------------------------------------


def _tensor_cache_key(t: torch.Tensor) -> tuple:
    return (
        t.data_ptr(),
        tuple(t.shape),
        tuple(t.stride()),
        str(t.dtype),
        getattr(t, "_version", None),
    )


def _tensor_to_f32_numpy(t: torch.Tensor) -> np.ndarray:
    """Return a contiguous float32 C-order numpy array, no copy if possible."""
    return t.detach().float().cpu().contiguous().numpy()


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class CIPTLinearRTLFunction:
    """
    C-accelerated drop-in replacement for IPTLinearRTLFunction.

    The public interface is identical:

        fn = CIPTLinearRTLFunction(vec_len=32, num_lanes=16,
                                   pipeline_depth=1,
                                   out_fmt_sel=OutputFmtSel.OutBF16)
        y = fn(x_q, w_q, b_q, scale_exp=0)   # float32 tensor

    The first instantiation triggers a one-time gcc compilation.
    All subsequent instances reuse the cached .so.
    """

    def __init__(
        self,
        vec_len: int = 32,
        num_lanes: int = 16,
        pipeline_depth: int = 1,
        out_fmt_sel: OutputFmtSel = OutputFmtSel.OutBF16,
    ) -> None:
        self.vec_len = vec_len
        self.num_lanes = num_lanes
        self.pipeline_depth = pipeline_depth
        self.out_fmt_sel = out_fmt_sel

        self._w_cache_key: Optional[tuple] = None
        self._b_cache_key: Optional[tuple] = None
        self._w_np: Optional[np.ndarray] = None
        self._b_np: Optional[np.ndarray] = None

        _get_lib()

    def _prepare_weights(
        self,
        w_q: torch.Tensor,
        b_q: Optional[torch.Tensor],
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        w_key = _tensor_cache_key(w_q)
        b_key = _tensor_cache_key(b_q) if b_q is not None else None

        if w_key != self._w_cache_key or b_key != self._b_cache_key:
            # Weights are large and only quantized once (cached thereafter).
            # Numba parallel beats single-threaded C at large shapes
            # (55x vs 42x over scalar at 4096x1024). Falls back to C if
            # Numba is not installed.
            _quant = (
                float_to_e4m3_bytes_numba if NUMBA_AVAILABLE else float_to_e4m3_bytes_c
            )
            self._w_np = _quant(_tensor_to_f32_numpy(w_q))
            self._b_np = _quant(_tensor_to_f32_numpy(b_q)) if b_q is not None else None
            self._w_cache_key = w_key
            self._b_cache_key = b_key

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

        x_ptr = x_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        w_ptr = w_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        b_ptr = (
            b_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            if b_np is not None
            else ctypes.cast(None, ctypes.POINTER(ctypes.c_uint8))
        )

        out_np = np.empty(batch * out_features, dtype=np.uint16)
        out_ptr = out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))

        lib.shim_ipt_linear_call(
            ctypes.c_int(self.num_lanes),
            ctypes.c_int(self.vec_len),
            ctypes.c_int(self.pipeline_depth),
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

        if self.out_fmt_sel == OutputFmtSel.OutBF16:
            # BF16 output: reinterpret uint16 bits as float32 via a torch view.
            # This is a zero-copy cast -- no C decode needed.
            u32 = torch.from_numpy(out_np.astype(np.int32)).reshape(batch, out_features)
            u32 = (u32 & 0xFFFF) << 16
            return u32.view(torch.float32)
        else:
            # E4M3 output: low byte of each uint16 is an E4M3 byte.
            # Decode via the C LUT -- faster than the torch gather path.
            e4m3_np = (out_np & 0xFF).astype(np.uint8)
            return torch.from_numpy(
                e4m3_bytes_to_float_c(e4m3_np).reshape(batch, out_features)
            )

    def __call__(
        self,
        x_q: torch.Tensor,
        w_q: torch.Tensor,
        b_q: Optional[torch.Tensor] = None,
        scale_exp: int = 0,
    ) -> torch.Tensor:
        original_shape = x_q.shape[:-1]
        in_features = x_q.shape[-1]
        out_features = w_q.shape[0]

        x2 = x_q.reshape(-1, in_features).float()
        batch = x2.shape[0]

        x_np = float_to_e4m3_bytes_c(_tensor_to_f32_numpy(x2))
        w_np, b_np = self._prepare_weights(w_q, b_q)

        y = self._call_c(x_np, w_np, b_np, batch, in_features, out_features, scale_exp)
        return y.reshape(*original_shape, out_features)

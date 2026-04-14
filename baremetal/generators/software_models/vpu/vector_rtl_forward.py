"""vector_rtl_forward.py — torch adapter for the vector engine funct model.

`VectorRTLFunctions` is a thin wrapper around `VectorEngineModel` that
takes real `torch.Tensor` inputs, quantizes them to BF16, splits them
into `num_lanes`-wide chunks, dispatches each chunk through the model,
and returns the dequantized result as a torch tensor of the original
dtype/device.

Design notes:

- The model has no persistent weights — unlike the IPT / SA linear
  adapters, nothing needs to be cached across calls. Every method is
  a pure function of its inputs.
- All inputs are first cast to `torch.float32`, then to BF16 bits via
  `torch_float_to_bf16_bits` (RNE, matching
  `funct_models_ipt.python_ipt_base.fp_formats.f32_to_bf16_bits_rne`).
- Pointwise ops (add/sub/mul/pairwise_max/pairwise_min/relu/rcp/sqrt/
  sin/cos/log/tanh/exp/exp2/square/cube/mov) flatten the input, chunk
  it into `num_lanes`-wide rows (zero-padding the last row if the
  flat length is not a multiple of `num_lanes`), dispatch each chunk,
  then reshape the output back to the input shape.
- Row reductions (`rsum`, `rmax`, `rmin`) require the last dim to be
  a multiple of `num_lanes`. The adapter reshapes `(..., K)` →
  `(..., K // num_lanes, num_lanes)`, runs the reduction on each
  length-`num_lanes` slice, and returns a tensor of shape
  `(..., K // num_lanes)` holding each slice's scalar reduction.
- Col reductions (`csum`, `cmax`, `cmin`) take an input of shape
  `(rows, num_lanes)` and return a length-`num_lanes` vector of
  per-column reductions, via `VectorEngineModel.stream_col_reduce`.
  Unlike the single-call pointwise path, csum keeps a widened
  FP32 accumulator across rows; the final narrowing to BF16 is a raw
  upper-16-bit slice matching `ColAddVec.scala`.
- `fp8_pack` takes an input of shape `(rows_pair, num_lanes)` where
  `rows_pair` must equal 2, plus a scalar `scale_e8m0`; returns a
  length-`num_lanes` `torch.int32` holding the packed 16-bit slots.
- `fp8_unpack` takes an `int`/`uint` tensor of shape `(num_lanes,)`
  holding packed 16-bit slots plus a scalar `scale_e8m0`; returns a
  `(2, num_lanes)` BF16-quantized float tensor.

The adapter follows the style of
`funct_models_ipt.python_ipt_base.ipt_rtl_linear` — BF16 round-trip via
`torch_float_to_bf16_bits` / `torch_bf16_bits_to_float`, inputs
materialized on CPU for the dispatch loop, outputs returned on the
caller's device. This is fine for functional correctness; performance
optimization (batched dispatch, a vectorized path) is deliberately out
of scope.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import torch

from .vector_engine_model import VectorEngineModel
from .vector_params import VectorParams


# ----------------------------------------------------------------
#  BF16 round-trip helpers (copied from ipt_rtl_linear for style
#  consistency — the two adapters don't share a helper module yet).
# ----------------------------------------------------------------

def torch_float_to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """torch.float → BF16 bit pattern (uint16 stored in int32). RNE."""
    x_f32 = x.float().contiguous()
    u32 = x_f32.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = (u32 >> 16) & 0xFFFF
    lsb = upper & 1
    rounded = u32 + (0x7FFF + lsb)
    return ((rounded >> 16) & 0xFFFF).to(torch.int32)


def torch_bf16_bits_to_float(bits: torch.Tensor) -> torch.Tensor:
    """BF16 bit pattern (int32 with uint16 value) → torch.float32."""
    u32 = (bits.to(torch.int32) & 0xFFFF) << 16
    return u32.view(torch.float32)


# Pointwise ops dispatched through `VectorEngineModel.execute`.
_BINARY_OPS = ("add", "sub", "mul", "pairmax", "pairmin")
_UNARY_OPS = (
    "rcp", "sqrt", "sin", "cos", "log", "tanh", "exp", "exp2",
    "square", "cube", "relu", "mov",
)
_ROW_REDUCE_OPS = ("rsum", "rmax", "rmin")
_COL_REDUCE_OPS = ("csum", "cmax", "cmin")


class VectorRTLFunctions:
    """Torch-facing wrapper around `VectorEngineModel.execute` and
    `stream_col_reduce`. Single instance per `num_lanes`; safe to reuse
    across calls."""

    def __init__(self, num_lanes: int = 16):
        self.num_lanes = num_lanes
        self._params = dataclasses.replace(VectorParams(), num_lanes=num_lanes)
        self.model = VectorEngineModel(self._params)

    # ------------------------------------------------------------
    #  Binary pointwise
    # ------------------------------------------------------------

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._pointwise_binary("add", a, b)

    def sub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._pointwise_binary("sub", a, b)

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._pointwise_binary("mul", a, b)

    def pairwise_max(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._pointwise_binary("pairmax", a, b)

    def pairwise_min(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self._pointwise_binary("pairmin", a, b)

    # ------------------------------------------------------------
    #  Unary pointwise
    # ------------------------------------------------------------

    def relu(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("relu", a)

    def rcp(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("rcp", a)

    def sqrt(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("sqrt", a)

    def sin(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("sin", a)

    def cos(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("cos", a)

    def log2(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("log", a)

    def tanh(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("tanh", a)

    def exp(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("exp", a)

    def exp2(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("exp2", a)

    def square(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("square", a)

    def cube(self, a: torch.Tensor) -> torch.Tensor:
        return self._pointwise_unary("cube", a)

    # ------------------------------------------------------------
    #  Row reductions
    # ------------------------------------------------------------

    def rsum(self, a: torch.Tensor) -> torch.Tensor:
        return self._row_reduce("rsum", a)

    def rmax(self, a: torch.Tensor) -> torch.Tensor:
        return self._row_reduce("rmax", a)

    def rmin(self, a: torch.Tensor) -> torch.Tensor:
        return self._row_reduce("rmin", a)

    # ------------------------------------------------------------
    #  Col reductions (cross-row streaming)
    # ------------------------------------------------------------

    def csum(self, a: torch.Tensor) -> torch.Tensor:
        """Stream N rows of `num_lanes` BF16 values through the FP32
        widened accumulator. Input: `(N, num_lanes)`. Output:
        `(num_lanes,)` per-column BF16 sums."""
        return self._col_reduce("csum", a)

    def cmax(self, a: torch.Tensor) -> torch.Tensor:
        return self._col_reduce("cmax", a)

    def cmin(self, a: torch.Tensor) -> torch.Tensor:
        return self._col_reduce("cmin", a)

    # ------------------------------------------------------------
    #  FP8 pack / unpack
    # ------------------------------------------------------------

    def fp8_pack(self, a: torch.Tensor, scale_e8m0: int) -> torch.Tensor:
        """Two BF16 rows → 16 UInt16 packed slots. Input shape must be
        `(2, num_lanes)`; output is an int32 tensor of shape
        `(num_lanes,)` holding the 16-bit packed slots."""
        if a.shape != (2, self.num_lanes):
            raise ValueError(
                f"fp8_pack expects shape (2, {self.num_lanes}), got {tuple(a.shape)}"
            )
        bits_row_lo = torch_float_to_bf16_bits(a[0]).tolist()
        bits_row_hi = torch_float_to_bf16_bits(a[1]).tolist()
        packed = self.model.execute(
            "fp8pack",
            a_vec=bits_row_lo,
            b_vec=bits_row_hi,
            scale_e8m0=scale_e8m0,
        )
        return torch.tensor(packed, dtype=torch.int32, device=a.device)

    def fp8_unpack(self, a: torch.Tensor, scale_e8m0: int) -> torch.Tensor:
        """16 UInt16 packed slots → two BF16 rows. Input shape must be
        `(num_lanes,)` int-like; output is a float32 tensor of shape
        `(2, num_lanes)` with each value BF16-quantized."""
        if a.shape != (self.num_lanes,):
            raise ValueError(
                f"fp8_unpack expects shape ({self.num_lanes},), got {tuple(a.shape)}"
            )
        packed = a.to(torch.int32).tolist()
        bits_out = self.model.execute(
            "fp8unpack", a_vec=packed, scale_e8m0=scale_e8m0
        )
        low = torch_bf16_bits_to_float(
            torch.tensor(bits_out[: self.num_lanes], dtype=torch.int32, device=a.device)
        )
        high = torch_bf16_bits_to_float(
            torch.tensor(bits_out[self.num_lanes :], dtype=torch.int32, device=a.device)
        )
        return torch.stack([low, high], dim=0)

    # ------------------------------------------------------------
    #  Internals
    # ------------------------------------------------------------

    def _pointwise_binary(
        self, op: str, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        if a.shape != b.shape:
            raise ValueError(
                f"{op}: shape mismatch a={tuple(a.shape)} b={tuple(b.shape)}"
            )
        a_bits = self._flatten_to_bits(a)
        b_bits = self._flatten_to_bits(b)
        out_bits = self._dispatch_chunks_binary(op, a_bits, b_bits)
        return self._reshape_out(out_bits, a.shape, a.device, a.dtype)

    def _pointwise_unary(self, op: str, a: torch.Tensor) -> torch.Tensor:
        a_bits = self._flatten_to_bits(a)
        out_bits = self._dispatch_chunks_unary(op, a_bits)
        return self._reshape_out(out_bits, a.shape, a.device, a.dtype)

    def _row_reduce(self, op: str, a: torch.Tensor) -> torch.Tensor:
        """`(..., K)` → `(..., K // num_lanes)` where the last dim must
        be a multiple of `num_lanes`. Each length-`num_lanes` slice is
        reduced via the lane_box; the scalar result (always lane 0)
        becomes one element of the output."""
        if a.shape[-1] % self.num_lanes != 0:
            raise ValueError(
                f"{op}: last dim {a.shape[-1]} must be a multiple of "
                f"{self.num_lanes}"
            )
        leading = a.shape[:-1]
        last = a.shape[-1]
        num_slices = last // self.num_lanes
        bits = self._flatten_to_bits(a)
        out_scalars: list[int] = []
        for i in range(0, len(bits), self.num_lanes):
            chunk = bits[i : i + self.num_lanes]
            result = self.model.execute(op, a_vec=chunk)
            # Every reduction lane_box broadcasts the scalar across all
            # `num_lanes` output slots. Take the first one.
            out_scalars.append(result[0] & 0xFFFF)
        out_shape = tuple(leading) + (num_slices,)
        out_tensor = torch.tensor(out_scalars, dtype=torch.int32, device="cpu")
        floats = torch_bf16_bits_to_float(out_tensor).reshape(out_shape)
        return floats.to(a.device).to(a.dtype)

    def _col_reduce(self, op: str, a: torch.Tensor) -> torch.Tensor:
        """`(N, num_lanes)` → `(num_lanes,)` via streaming."""
        if a.dim() != 2 or a.shape[1] != self.num_lanes:
            raise ValueError(
                f"{op}: expected shape (N, {self.num_lanes}), got {tuple(a.shape)}"
            )
        rows_bits: list[list[int]] = []
        for i in range(a.shape[0]):
            row = torch_float_to_bf16_bits(a[i]).tolist()
            rows_bits.append(row)
        result = self.model.stream_col_reduce(op, rows_bits)
        out_tensor = torch.tensor(result, dtype=torch.int32, device="cpu")
        floats = torch_bf16_bits_to_float(out_tensor)
        return floats.to(a.device).to(a.dtype)

    # ------------------------------------------------------------

    def _flatten_to_bits(self, a: torch.Tensor) -> list[int]:
        return torch_float_to_bf16_bits(a.reshape(-1).cpu()).tolist()

    def _dispatch_chunks_binary(
        self, op: str, a_bits: list[int], b_bits: list[int]
    ) -> list[int]:
        n = self.num_lanes
        out: list[int] = []
        total = len(a_bits)
        for i in range(0, total, n):
            chunk_a = a_bits[i : i + n]
            chunk_b = b_bits[i : i + n]
            pad = n - len(chunk_a)
            if pad:
                chunk_a = chunk_a + [0] * pad
                chunk_b = chunk_b + [0] * pad
            result = self.model.execute(op, a_vec=chunk_a, b_vec=chunk_b)
            out.extend(result[:n])
        return out[:total]

    def _dispatch_chunks_unary(
        self, op: str, a_bits: list[int]
    ) -> list[int]:
        n = self.num_lanes
        out: list[int] = []
        total = len(a_bits)
        for i in range(0, total, n):
            chunk = a_bits[i : i + n]
            pad = n - len(chunk)
            if pad:
                chunk = chunk + [0] * pad
            result = self.model.execute(op, a_vec=chunk)
            out.extend(result[:n])
        return out[:total]

    def _reshape_out(
        self,
        out_bits: Iterable[int],
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        bits_tensor = torch.tensor(list(out_bits), dtype=torch.int32, device="cpu")
        floats = torch_bf16_bits_to_float(bits_tensor)
        floats = floats.reshape(shape)
        return floats.to(device).to(dtype)

#!/usr/bin/env python3
"""Shared MXU/FP8 helpers for the smolvla_fused_* generators.

Wraps the per-element `bf16_scale_to_e4m3` from the IPT converter into
a tile-level loop. Mirrors atlas's hardware BF16->FP8 quantization path
(`VMATPUSH.ACC.BF16.MXU0` + `SELI` + `VMATPOP.FP8.MXU0`).

Underflow: BF16 values whose scaled unbiased exponent is below -6
flush to +0 / -0. The converter does NOT emit E4M3 subnormals — see
`bf16_scale_to_e4m3` lines 166-168 in
`software_models/mxu1_ipt/converters.py`. Callers that need subnormal
representation must handle it upstream.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from software_models.mxu1_ipt.converters import bf16_scale_to_e4m3


def bf16_tile_to_e4m3_bytes(
    tile: torch.Tensor, scale_exp: int = 0
) -> torch.Tensor:
    """Quantize a BF16 tile to FP8 E4M3 uint8 bytes via the IPT converter.

    `scale_exp` is the E8M0 block-scale exponent applied during the
    quant (matches `SELI <reg>, 0x7F + scale_exp` in hardware). Pass
    `0` for unit scale.
    """
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bits = tile.contiguous().view(torch.int16).numpy().astype(np.uint16)
    flat = bits.ravel()
    out = np.empty(flat.size, dtype=np.uint8)
    for i in range(flat.size):
        out[i] = bf16_scale_to_e4m3(int(flat[i]), int(scale_exp)) & 0xFF
    return torch.from_numpy(out.reshape(bits.shape))

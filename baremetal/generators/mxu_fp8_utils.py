#!/usr/bin/env python3
"""Shared MXU/FP8 helpers for the smolvla_fused_* generators.

`bf16_tile_to_e4m3_bytes` wraps the per-element `bf16_scale_to_e4m3` from
`software_models/mxu1_ipt/converters.py` into a tile-level loop: BF16 values
whose scaled unbiased exponent is below -6 flush to +0, and it emits no E4M3
subnormals.

`bf16_tile_to_e4m3_bytes_mxu1` mirrors atlas's hardware BF16->FP8 quantization
path (`VMATPUSH.ACC.BF16` + `SELI` + `VMATPOP.FP8`), which rounds once with E4M3
subnormals (E4M3_RN_SATFINITE, npu_model configs/cvt.py).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from npu_model.configs import cvt
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


def bf16_tile_to_e4m3_bytes_mxu1(tile: torch.Tensor, scale_exp: int = 0) -> torch.Tensor:
    """E4M3 bytes of a BF16 tile after `VMATPUSH.ACC.BF16.MXU1` + `SELI` +
    `VMATPOP.FP8.MXU1` (e = 127 + scale_exp): E4M3_RN_SATFINITE of tile * 2^scale_exp."""
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bits = tile.contiguous().view(torch.int16).numpy().astype(np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(cvt.e4m3_rn_satfinite(bits, 127 + scale_exp))

#!/usr/bin/env python3
"""Shared MXU/FP8 helpers for the smolvla_fused_* generators.

`bf16_tile_to_e4m3_bytes_mxu1` mirrors atlas's hardware BF16->FP8 quantization
path (`VMATPUSH.ACC.BF16` + `SELI` + `VMATPOP.FP8`), which rounds once with E4M3
subnormals (E4M3_RN_SATFINITE, npu_model configs/cvt.py).
"""

import numpy as np
import torch

from npu_model.configs import cvt


def bf16_tile_to_e4m3_bytes_mxu1(tile: torch.Tensor, scale_exp: int = 0) -> torch.Tensor:
    """E4M3 bytes of a BF16 tile after `VMATPUSH.ACC.BF16.MXU1` + `SELI` +
    `VMATPOP.FP8.MXU1` (e = 127 + scale_exp): E4M3_RN_SATFINITE of tile * 2^scale_exp."""
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bits = tile.contiguous().view(torch.int16).numpy().astype(np.uint16).astype(np.uint32) << 16
    return torch.from_numpy(cvt.e4m3_rn_satfinite(bits, 127 + scale_exp))

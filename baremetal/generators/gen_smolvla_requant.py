#!/usr/bin/env python3
"""Generate test vectors for `smolvla_requant.S`.

Port of npu-model/npu_model/configs/programs/smolvla_requant.py.
BF16 → FP8 unit-scale cast on a 32x32 tile via atlas VFP8PACK. The
scale register is loaded with E8M0 0x7F (= bias 127 → 2^0 = 1.0), so
the pack subtracts zero from the BF16 exponent and the result is a
plain BF16→E4M3 truncation per lane.

The packed output fits in a single mreg (32 rows × 32 FP8 bytes =
1024 B), half the size of the BF16 input tile.
"""

import json

import torch

from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    ROWS_PER_TENSOR,
    fp8_checks,
    run_fp8_pack_rows,
    tensor_preloads,
)


BEATS_PER_TENSOR = ROWS_PER_TENSOR
TIMEOUT = 30000

X_BASE = 0
OUT_BASE = BEATS_PER_TENSOR
SCALE_E8M0_UNIT = 0x7F  # 127 biased = 2^0


def bf16_tile_to_bank_rows(tile: torch.Tensor) -> tuple[list[list[int]], list[list[int]]]:
    if tile.shape != (ROWS_PER_REGISTER, 2 * BF16_PER_BEAT):
        raise ValueError(
            f"expected ({ROWS_PER_REGISTER}, {2 * BF16_PER_BEAT}) bf16 tile, "
            f"got {tuple(tile.shape)}"
        )
    if tile.dtype != torch.bfloat16:
        raise ValueError(f"expected bfloat16 tile, got {tile.dtype}")
    bank0 = tile[:, :BF16_PER_BEAT].contiguous().view(torch.int16)
    bank1 = tile[:, BF16_PER_BEAT:].contiguous().view(torch.int16)
    low_rows = [[int(v) & 0xFFFF for v in bank0[i].tolist()] for i in range(ROWS_PER_REGISTER)]
    high_rows = [[int(v) & 0xFFFF for v in bank1[i].tolist()] for i in range(ROWS_PER_REGISTER)]
    return low_rows, high_rows


def main():
    torch.manual_seed(48)
    x_tile = (torch.randn(32, 32, dtype=torch.bfloat16) * 0.5).to(torch.bfloat16)
    low_rows, high_rows = bf16_tile_to_bank_rows(x_tile)

    # Preloads: bank 0 then bank 1 (64 beats total, matching DMA loads below).
    preload_rows = low_rows + high_rows

    packed_rows = run_fp8_pack_rows(low_rows, high_rows, SCALE_E8M0_UNIT)

    preloads = tensor_preloads(X_BASE, preload_rows)
    checks = fp8_checks(OUT_BASE, packed_rows)

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

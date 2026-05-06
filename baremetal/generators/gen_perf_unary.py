#!/usr/bin/env python3
"""Generate test vectors for `perf_vec_max_utilization_32x32.S`.

Three staggered transcendental chains run across 8 fully-paired slots,
achieving 100% FU-slot utilization in the timed region.

Chain A (m0→m2), 8 ops in timed region:
    square → tanh → exp → sqrt → log2 → recip → square → tanh

Chain B (m4→m6), primed 1 step in setup, 5 ops in timed region:
    [setup: tanh]  tanh → exp → sqrt → log2 → recip

Chain C (m10→m8), primed 2 steps in setup, 3 ops in timed region:
    [setup: square → tanh]  square → tanh → exp

Slot pairing schedule (timed region, 520 cycles):
    Slot 1: VSQUARE(A,m0→m2)  ∥ VTANH(B,m4→m6)   SquareCube ∥ TanhRec ✓
    Slot 2: VTANH(A,m2)       ∥ VEXP(B,m6)         TanhRec   ∥ Exp     ✓
    Slot 3: VEXP(A,m2)        ∥ VSQRT(B,m6)        Exp       ∥ Sqrt    ✓
    Slot 4: VSQRT(A,m2)       ∥ VLOG2(B,m6)        Sqrt      ∥ Log     ✓
    Slot 5: VLOG2(A,m2)       ∥ VRECIP(B,m6)       Log       ∥ Rcp     ✓
    Slot 6: VRECIP(A,m2)      ∥ VSQUARE(C,m10→m8)  Rcp       ∥ SquareCube ✓
    Slot 7: VSQUARE(A,m2)     ∥ VTANH(C,m8)        SquareCube∥ TanhRec ✓
    Slot 8: VTANH(A,m2)       ∥ VEXP(C,m8)         TanhRec   ∥ Exp     ✓

    element-ops:       16 ops × 1024 = 16384
    actual throughput: 16384 / 520   = 31.51 elem/cy
    peak throughput:   2 × 1024 / 65 = 31.51 elem/cy
    utilization:       100.0%

DRAM layout:
    0x0000  A_raw (2048 B, BF16 32×32)  input for chain A
    0x0800  B_raw (2048 B, BF16 32×32)  input for chains B and C
    0x1000  Y_a   (2048 B, BF16 32×32)  chain A timed output, checked
    0x1800  Y_b   (2048 B, BF16 32×32)  chain B timed output, checked
    0x2000  Y_c   (2048 B, BF16 32×32)  chain C timed output, checked
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import emit_test_data
from vpu_gen_utils import (
    BF16_PER_BEAT,
    ROWS_PER_REGISTER,
    pack_u16_le,
    run_unary_rows,
)

ROWS    = 32
COLS    = 32
TIMEOUT = 25000

A_BASE  = 0x0000 // 32   # beat   0
B_BASE  = 0x0800 // 32   # beat  64
YA_BASE = 0x1000 // 32   # beat 128
YB_BASE = 0x1800 // 32   # beat 192
YC_BASE = 0x2000 // 32   # beat 256


def col_block_beats(tile: torch.Tensor) -> list[list[int]]:
    rows, cols = tile.shape
    beats: list[list[int]] = []
    for col_start in range(0, cols, BF16_PER_BEAT):
        block = (
            tile[:, col_start : col_start + BF16_PER_BEAT]
            .contiguous().view(torch.int16).numpy()
        )
        for i in range(rows):
            beats.append([int(v) & 0xFFFF for v in block[i].tolist()])
    return beats


def tile_to_bank_rows(tile: torch.Tensor) -> tuple[list[list[int]], list[list[int]]]:
    assert tile.shape == (ROWS, COLS)
    low  = tile[:, :BF16_PER_BEAT].contiguous().view(torch.int16)
    high = tile[:, BF16_PER_BEAT:].contiguous().view(torch.int16)
    low_rows  = [[int(v) & 0xFFFF for v in low[i].tolist()]  for i in range(ROWS)]
    high_rows = [[int(v) & 0xFFFF for v in high[i].tolist()] for i in range(ROWS)]
    return low_rows, high_rows


def bank_rows_to_tile(low_rows, high_rows) -> torch.Tensor:
    low_np  = np.array(low_rows,  dtype=np.uint16).view(np.int16).copy()
    high_np = np.array(high_rows, dtype=np.uint16).view(np.int16).copy()
    return torch.cat([
        torch.from_numpy(low_np).view(torch.bfloat16),
        torch.from_numpy(high_np).view(torch.bfloat16),
    ], dim=1)


def apply_ops(rows: list[list[int]], ops: list[str]) -> list[list[int]]:
    """Apply a sequence of named unary ops to bank rows."""
    y = rows
    for op in ops:
        y = run_unary_rows(op, y)
    return y


def main() -> None:
    # ── Seeded inputs ──────────────────────────────────────────────────────
    # Scale 0.3: keeps squared values in [0, ~0.09×N], tanh outputs
    # near-linear, exp outputs near 1, sqrt/log2/recip all well-conditioned.
    torch.manual_seed(42)
    A_raw = (torch.randn(ROWS, COLS) * 0.3).to(torch.bfloat16)
    B_raw = (torch.randn(ROWS, COLS) * 0.3).to(torch.bfloat16)

    a_low, a_high = tile_to_bank_rows(A_raw)
    b_low, b_high = tile_to_bank_rows(B_raw)
    a_rows = a_low + a_high
    b_rows = b_low + b_high

    # ── Chain A golden ─────────────────────────────────────────────────────
    # Full 8-op timed sequence (no setup priming needed for A):
    #   square → tanh → exp → sqrt → log2 → recip → square → tanh
    ya_rows = apply_ops(a_rows, [
        "square", "tanh", "exp", "sqrt", "log", "rcp", "square", "tanh"
    ])

    # ── Chain B golden ─────────────────────────────────────────────────────
    # Setup primes B with 1 op (tanh), then timed region adds 5 more:
    #   [setup: tanh]  tanh → exp → sqrt → log2 → recip
    # Full chain from raw input = tanh + tanh + exp + sqrt + log2 + recip
    yb_rows = apply_ops(b_rows, [
        "tanh", "exp", "sqrt", "log", "rcp"  # timed region slots 1-5
    ])

    # ── Chain C golden ─────────────────────────────────────────────────────
    # Uses B_raw as input (same as B). Setup primes C with 2 ops:
    #   [setup: square → tanh]  square → tanh → exp
    # Full chain from raw input = square + tanh + square + tanh + exp
    yc_rows = apply_ops(b_rows, [
        "square", "tanh", "exp"          # timed region slots 6-8
    ])

    # ── Reconstruct tiles ──────────────────────────────────────────────────
    Ya_tile = bank_rows_to_tile(ya_rows[:ROWS], ya_rows[ROWS:])
    Yb_tile = bank_rows_to_tile(yb_rows[:ROWS], yb_rows[ROWS:])
    Yc_tile = bank_rows_to_tile(yc_rows[:ROWS], yc_rows[ROWS:])

    # ── DRAM preloads ──────────────────────────────────────────────────────
    preloads: list[dict] = []
    for tile_raw, base in [(A_raw, A_BASE), (B_raw, B_BASE)]:
        for i, row in enumerate(col_block_beats(tile_raw)):
            preloads.append({"word_offset": base + i, "data": pack_u16_le(row)})

    # ── Golden checks ──────────────────────────────────────────────────────
    checks: list[dict] = []
    for tile_out, base in [(Ya_tile, YA_BASE), (Yb_tile, YB_BASE), (Yc_tile, YC_BASE)]:
        for i, row in enumerate(col_block_beats(tile_out)):
            checks.append({"word_offset": base + i, "expected": pack_u16_le(row)})

    emit_test_data(preloads, checks, timeout=TIMEOUT)


if __name__ == "__main__":
    main()
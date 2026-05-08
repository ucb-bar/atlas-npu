#!/usr/bin/env python3
"""gen_mreg_cross_engine_full_rotor.py - golden generator for
mreg_cross_engine_full_rotor.S.

The matching assembly walks VTRPOSE.XLU through every bank index 0..63
and then runs a VPU VMOV pair rotor across every even pair, finally
draining bank 63 back to DRAM at byte offset 0x1000.

To avoid having to model the precise XLU element-width semantics, the
input tile is constructed to be SYMMETRIC at the byte level:
    byte[r][c] == byte[c][r]   for r, c in 0..31
Any number of byte-level transposes of a symmetric matrix is a no-op,
so every bank's bytes after the chain equal the input tile, and so
does the VSTORE drain.

The generator therefore emits:
    * preload at DRAM 0x0000..0x03FF: the symmetric tile
    * check   at DRAM 0x1000..0x13FF: the same symmetric tile
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from gen_utils import (  # noqa: E402
    make_check_any,
    make_preload_any,
)


TIMEOUT = 800000
TILE_BYTES = 1024
INPUT_DRAM_BASE = 0x0000
OUTPUT_DRAM_BASE = 0x1000


def symmetric_byte(r: int, c: int) -> int:
    """Return a deterministic byte that depends only on the unordered
    pair {r, c}, so byte[r][c] == byte[c][r].

    The combination uses several primes so adjacent cells (and tiles
    rotated through XLU/VPU) drive many bit positions, helping toggle
    coverage on the MREG bank shells.
    """
    a, b = (r, c) if r <= c else (c, r)
    val = (
        ((a * b * 7) & 0xFF)
        ^ (((a + b) * 13) & 0xFF)
        ^ ((a * 0x0B) & 0xFF)
        ^ ((b * 0x0B) & 0xFF)
        ^ 0xA5
    )
    return val & 0xFF


def build_symmetric_tile() -> list[int]:
    payload: list[int] = []
    for r in range(32):
        for c in range(32):
            payload.append(symmetric_byte(r, c))
    if len(payload) != TILE_BYTES:
        raise AssertionError(f"expected {TILE_BYTES} bytes, got {len(payload)}")
    return payload


def bytes_to_beats(payload: list[int]) -> list[int]:
    if len(payload) % 32 != 0:
        raise ValueError(
            f"payload length {len(payload)} is not a multiple of 32 bytes"
        )
    beats: list[int] = []
    for beat_start in range(0, len(payload), 32):
        beat = 0
        for j in range(32):
            beat |= (payload[beat_start + j] & 0xFF) << (8 * j)
        beats.append(beat)
    return beats


def beats_to_preloads(byte_base: int, beats: list[int]) -> list[dict]:
    if byte_base % 32 != 0:
        raise ValueError(f"byte_base {byte_base} not 32-byte aligned")
    base_beat = byte_base // 32
    return [make_preload_any(base_beat + i, beat) for i, beat in enumerate(beats)]


def beats_to_checks(byte_base: int, beats: list[int]) -> list[dict]:
    if byte_base % 32 != 0:
        raise ValueError(f"byte_base {byte_base} not 32-byte aligned")
    base_beat = byte_base // 32
    return [make_check_any(base_beat + i, beat) for i, beat in enumerate(beats)]


def main() -> None:
    payload = build_symmetric_tile()
    # Symmetric assertion (cheap correctness check).
    for r in range(32):
        for c in range(32):
            if payload[r * 32 + c] != payload[c * 32 + r]:
                raise AssertionError(f"tile not symmetric at ({r},{c})")
    beats = bytes_to_beats(payload)
    preloads = beats_to_preloads(INPUT_DRAM_BASE, beats)
    checks = beats_to_checks(OUTPUT_DRAM_BASE, beats)

    sys.stderr.write(
        f"[gen_mreg_cross_engine_full_rotor] preloads={len(preloads)} "
        f"checks={len(checks)}\n"
    )

    print(json.dumps({
        "dram_preloads": preloads,
        "dram_checks": checks,
        "timeout": TIMEOUT,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Shared data and golden helpers for the NVFP4 MXU tests (gen_nvfp4_*.py).

The arithmetic is npu_model/configs/nvfp4.py (from ucb-bar/nvfp4@0c611e1:host/nvfp4_model.py, the
model of fp4_fedp) and the BF16 conversion npu_model/configs/cvt.py, loaded by path so the
generators need no npu_model environment. Layouts
follow docs/nvfp4-mxu.md:
  tile data   32 rows x 32 bytes, 64 E2M1 codes per row (low nibble first)
  tile scales 32 rows x 4 UE4M3 bytes (128 bytes)
  FP32 tile   4 m-registers, register q row r = columns 8q .. 8q + 7
"""
import importlib.util
import json
import random
import struct
from pathlib import Path

import numpy as np


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[2] / "npu-model" / "npu_model" / "configs" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nvfp4, cvt = _load("nvfp4"), _load("cvt")

BEAT = 32
ROWS = 32


def rand_tile(rng: random.Random, zero_rows=()) -> list[bytes]:
    """32 rows of 64 random E2M1 codes (every code, signed zeros included)."""
    rows = []
    for r in range(ROWS):
        if r in zero_rows:
            rows.append(bytes(BEAT))
        else:
            rows.append(bytes(rng.randrange(256) for _ in range(BEAT)))
    return rows


def rand_scales(rng: random.Random, specials: bool = False) -> list[bytes]:
    """32 rows of 4 UE4M3 scales, mostly 2^-3 .. 2^3; with specials also zero, subnormal,
    the largest finite scale, bit 7 set (ignored) and NaN (0x7F)."""
    def one():
        if specials and rng.random() < 0.15:
            return rng.choice([0x00, 0x01, 0x07, 0x7E, 0xB8, 0x7F, 0x08])
        return (rng.randrange(4, 11) << 3) | rng.randrange(8)
    return [bytes(one() for _ in range(4)) for _ in range(ROWS)]


def f32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def rand_fp32_tile(rng: random.Random, specials: bool = False) -> list[list[int]]:
    def one():
        if specials and rng.random() < 0.1:
            return rng.choice([0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0x8000_0000, 0x0000_0001,
                               0x8040_0000, 0x7F7F_FFFF])
        return f32(rng.uniform(-64.0, 64.0))
    return [[one() for _ in range(32)] for _ in range(ROWS)]


def matmul(a, sa, w, sw, c=None):
    c = c if c is not None else [[0] * 32 for _ in range(ROWS)]
    return nvfp4.matmul(a, sa, w, sw, c)


def fp32_regs(tile) -> list[bytes]:
    """FP32 tile -> 4 m-registers (1 KiB each) as stored by vmatpop.fp32.acc.nv."""
    return [b"".join(struct.pack("<8I", *row[8 * q:8 * q + 8]) for row in tile) for q in range(4)]


def bf16_regs(tile) -> list[bytes]:
    """FP32 tile -> {m[d], m[d+1]} as stored by vmatpop.bf16.acc.nv (BF16_RNE)."""
    bits = cvt.bf16_rne(np.array(tile, dtype=np.uint32))
    return [bits[:, 16 * h:16 * h + 16].astype("<u2").tobytes() for h in range(2)]


class Image:
    """DRAM image at the test's @DRAM_BASE, in 32-byte beats."""

    def __init__(self):
        self.preloads, self.checks = [], []

    @staticmethod
    def _beats(data: bytes):
        assert len(data) % BEAT == 0
        for i in range(0, len(data), BEAT):
            yield i // BEAT, "0x%064x" % int.from_bytes(data[i:i + BEAT], "little")

    def preload(self, beat: int, data: bytes) -> None:
        self.preloads += [{"word_offset": beat + i, "data": h} for i, h in self._beats(data)]

    def check(self, beat: int, data: bytes) -> None:
        self.checks += [{"word_offset": beat + i, "expected": h} for i, h in self._beats(data)]

    def dump(self) -> None:
        print(json.dumps({"dram_preloads": self.preloads, "dram_checks": self.checks}, indent=1))

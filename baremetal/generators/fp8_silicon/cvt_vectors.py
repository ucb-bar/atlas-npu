#!/usr/bin/env python3
"""Inputs of the conversion vectors of fp8_silicon (README.md), and the reader of *.cvt.gz files.

    python3 -I cvt_vectors.py gen DIR    # DIR/pop.bin, the input records of `fp8_cvt pop`

pop.bin records are {uint32 x, uint32 e}, little-endian: x is FP32 bits, e is 0..255.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

SPECIAL = np.array([0, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0000, 0x7F80_0001,
                    0xFFFF_FFFF, 0x0000_0001, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF, 0x0080_0000,
                    0x8080_0000, 0x7F7F_FFFF, 0xFF7F_FFFF], np.uint32)


def e4m3_boundaries():
    """Every positive finite E4M3 value, every midpoint between two neighbours, and 464."""
    code = np.arange(127)
    field, m = code >> 3, code & 7
    value = np.where(field > 0, 8 + m, m) * 2.0 ** (np.maximum(field, 1) - 10)
    return np.concatenate([value[1:], (value[:-1] + value[1:]) / 2, [464.0]])


def pop_inputs():
    """(x, e): at every e, each FP32 x for which x * G(e) is an E4M3 boundary, its FP32
    neighbours, both signs and the SPECIAL values; then random x, half of them with x * G(e) in
    2^-12 .. 2^10."""
    e = np.arange(256)
    x = e4m3_boundaries()[:, None] * 2.0 ** (127 - e)
    with np.errstate(over="ignore"):
        x32 = x.astype(np.float32)
    exact = x32 == x
    bits = x32.view(np.uint32)[exact].astype(np.int64)[:, None] + [-1, 0, 1]
    bits = np.concatenate([bits, bits | 0x8000_0000], 1)
    e_b = np.broadcast_to(e, x.shape)[exact][:, None].repeat(6, 1)
    rng = np.random.default_rng(1)
    n = 50_000
    e_r = rng.integers(0, 256, 2 * n)
    field = np.clip(rng.integers(-12, 10, n) + 254 - e_r[:n], 0, 254)
    sign = rng.integers(0, 2, n).astype(np.uint32) << 31
    near = sign | field.astype(np.uint32) << 23 | rng.integers(0, 1 << 23, n).astype(np.uint32)
    x = np.concatenate([bits.ravel(), np.tile(SPECIAL, 256), near,
                        rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)])
    e = np.concatenate([e_b.ravel(), e.repeat(len(SPECIAL)), e_r])
    return x.astype(np.uint32), e.astype(np.uint8)


def read_cvt(path):
    """(header, {column: array}) of a cvt-vectors-v1 file."""
    raw = gzip.decompress(Path(path).read_bytes())
    nl = raw.index(b"\n")
    hdr = json.loads(raw[:nl])
    cols, offset = {}, nl + 1
    for name, dtype in hdr["columns"]:
        cols[name] = np.frombuffer(raw, dtype, hdr["count"], offset)
        offset += cols[name].nbytes
    return hdr, cols


def main(argv):
    if argv[:1] == ["gen"] and len(argv) == 2:
        out = Path(argv[1])
        out.mkdir(parents=True, exist_ok=True)
        x, e = pop_inputs()
        (out / "pop.bin").write_bytes(np.stack([x, e.astype(np.uint32)], 1).astype("<u4").tobytes())
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

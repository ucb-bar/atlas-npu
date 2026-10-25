#!/usr/bin/env python3
"""Inputs of the conversion vectors of fp8_silicon (README.md), and the packer and the reader of
*.cvt.gz files.

    python3 -I cvt_vectors.py gen DIR        # DIR/pop.bin, the input records of `fp8_cvt pop`
    python3 -I cvt_vectors.py pack GPU OUT   # OUT/*.cvt.gz from the fp8_cvt outputs in GPU

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
SOURCE = ("NVIDIA GeForce RTX 5090 (sm_120, driver 595.99.02), CUDA 13.2 V13.2.78, fp8_cvt.cu built with "
          "-gencode arch=compute_120a,code=sm_120a, run on 2026-10-24")
FMUL = "x * G(e) in FP32 by FMUL (for e = 255: FMUL by 2^127, then FMUL by 2)"
TO_E4M3 = "cvt.rn.satfinite.e4m3x2.f32 (SASS F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C)"
UNPACK = "cvt.rn.f16x2.e4m3x2 (SASS F2FP.F16.E4M3.UNPACK_B)"
RUNS = ("run k: every FP32 x from x[k] up to x[k + 1] - 1 (for the last run, up to 0xFFFFFFFF) "
        "converts to rtx5090[k]; the runs cover all 2^32 FP32 bit patterns")


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


def write_cvt(path, name, columns, description, inputs, source):
    """Write a cvt-vectors-v1 file; columns are (name, numpy type, values)."""
    hdr = dict(format="cvt-vectors-v1", name=name, count=len(columns[0][2]),
               columns=[[n, np.dtype(d).str] for n, d, _ in columns],
               description=description, inputs=inputs, source=f"{SOURCE}; {source}")
    body = b"".join(np.asarray(v).astype(np.dtype(d).newbyteorder("<")).tobytes() for _, d, v in columns)
    Path(path).write_bytes(gzip.compress(json.dumps(hdr).encode() + b"\n" + body, 9, mtime=0))


def pack(gpu, out):
    """The *.cvt.gz files from the outputs of the fp8_cvt modes in gpu (README.md, Commands)."""
    out, gpu = Path(out), Path(gpu)
    write_cvt(out / "vpack.cvt.gz", "vpack", [("rtx5090", "u1", np.fromfile(gpu / "vpack.out", np.uint8))],
              "vpack.bf16.fp8: E4M3 of x * G(e) for every BF16 x at every e",
              "vector i: e = i >> 16, x = BF16 bits i & 0xFFFF",
              f"fp8_cvt vpack: {FMUL}, then {TO_E4M3}")
    write_cvt(out / "vunpack.cvt.gz", "vunpack", [("rtx5090", "u2", np.fromfile(gpu / "vunpack.out", "<u2"))],
              "vunpack.fp8.bf16: BF16 of x * G(e) for every E4M3 x at every e",
              "vector i: e = i >> 8, x = E4M3 code i & 0xFF",
              f"fp8_cvt vunpack: {UNPACK}, cvt.f32.f16 (HADD2.F32), {FMUL}, then cvt.rn.bf16.f32 "
              "(F2FP.BF16.F32.PACK_AB)")
    write_cvt(out / "push.cvt.gz", "push", [("rtx5090", "u4", np.fromfile(gpu / "push.out", "<u4"))],
              "vmatpush.acc.fp8.mxu1: FP32 of every E4M3 x", "vector i: x = E4M3 code i",
              f"fp8_cvt push: {UNPACK}, then cvt.f32.f16 (HADD2.F32)")
    x, e = pop_inputs()
    write_cvt(out / "pop.cvt.gz", "pop",
              [("x", "u4", x), ("e", "u1", e), ("rtx5090", "u1", np.fromfile(gpu / "pop.out", np.uint8))],
              "vmatpop.fp8.acc.mxu1: E4M3 of x * G(e) for FP32 x on and next to the E4M3 boundaries at every e, "
              "special and extreme x at every e, and random x (cvt_vectors.py pop_inputs())",
              "columns x (FP32 bits) and e", f"fp8_cvt pop: {FMUL}, then {TO_E4M3}")
    r = np.fromfile(gpu / "e4m3.out", "<u4").reshape(-1, 2)
    write_cvt(out / "e4m3_runs.cvt.gz", "e4m3_runs", [("x", "u4", r[:, 0]), ("rtx5090", "u1", r[:, 1])],
              "E4M3 of every FP32 x, as runs", RUNS, f"fp8_cvt e4m3: {TO_E4M3}")
    r = np.fromfile(gpu / "bf16.out", "<u4").reshape(-1, 2)
    assert (r == np.fromfile(gpu / "bf16x2.out", "<u4").reshape(-1, 2)).all(), "bf16 and bf16x2 differ"
    write_cvt(out / "bf16_runs.cvt.gz", "bf16_runs", [("x", "u4", r[:, 0]), ("rtx5090", "u2", r[:, 1])],
              "BF16 of every FP32 x, as runs", RUNS,
              "fp8_cvt bf16: cvt.rn.bf16.f32 (SASS F2F.BF16.F32); fp8_cvt bf16x2, cvt.rn.bf16x2.f32 "
              "(SASS F2FP.BF16.F32.PACK_AB), gave the same runs")


def main(argv):
    if argv[:1] == ["gen"] and len(argv) == 2:
        out = Path(argv[1])
        out.mkdir(parents=True, exist_ok=True)
        x, e = pop_inputs()
        (out / "pop.bin").write_bytes(np.stack([x, e.astype(np.uint32)], 1).astype("<u4").tobytes())
        return 0
    if argv[:1] == ["pack"] and len(argv) == 3:
        Path(argv[2]).mkdir(parents=True, exist_ok=True)
        pack(argv[1], argv[2])
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

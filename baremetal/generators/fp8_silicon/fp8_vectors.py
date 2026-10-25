#!/usr/bin/env python3
"""Generate the E4M3 dot-product vector sets of fp8_silicon (README.md), and extract records.

    python3 -I fp8_vectors.py gen DIR                 # DIR/<set>.bin, the full sets
    python3 -I fp8_vectors.py records FILE.vec.gz OUT # the records of a stored set, for fp8_mma

Records are 68 bytes: a[32] b[32] c[4] (byte k is E4M3 element k; c is FP32, little-endian).
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np


def codes(rng, field):
    """Random-sign, random-mantissa E4M3 codes with exponent field `field`; NaN becomes 448."""
    sign = rng.integers(0, 2, field.shape) << 7
    code = sign | field.astype(np.int64) << 3 | rng.integers(0, 8, field.shape)
    return np.where(code & 0x7F == 0x7F, code ^ 1, code).astype(np.uint8)


def f32_bits(x):
    return np.asarray(x, np.float32).view(np.uint32)


def fp32_field(rng, lo, hi, n):
    """Random FP32 values with exponent field in [lo, hi]."""
    sign = rng.integers(0, 2, n).astype(np.uint32) << 31
    return sign | rng.integers(lo, hi + 1, n).astype(np.uint32) << 23 | rng.integers(0, 1 << 23, n).astype(np.uint32)


def e4m3_value(code):
    field, m = (code >> 3) & 15, code & 7
    mag = np.where(field > 0, (8 + m) * 2.0 ** (field.astype(np.int64) - 10), m * 2.0 ** -9)
    return np.where(code >> 7 == 1, -mag, mag)


def exact_dot(a, b):
    """a . b in float64; exact, as E4M3 products are multiples of 2^-18 and |a . b| < 2^23."""
    return (e4m3_value(a) * e4m3_value(b)).sum(-1)


def directed():
    """(names, a, b, c): vectors that probe one rule each of npu_model/configs/fp8_mma.py.
    Each case lists its nonzero elements as {k: (a_k, b_k)}; a float is an E4M3 value and an int
    is a raw E4M3 code. C is a float value or, as an int, raw FP32 bits."""
    cases = []
    add = lambda name, pairs, c: cases.append((name, pairs, c))
    cancel = {0: (1.0, 1.0), 1: (1.0, -1.0)}
    for j in (23, 24, 25, 26, 27):
        add(f"F: 1 - 1 + 2^-{j}", cancel, 2.0 ** -j)
        add(f"F: 1 - 1 - 2^-{j}", cancel, -2.0 ** -j)
    add("F: 1 - 1 + 3*2^-26", cancel, 3 * 2.0 ** -26)
    add("F: 1 - 1 - 3*2^-26", cancel, -3 * 2.0 ** -26)
    for j in (24, 25, 26):  # 1.5 * 1.5 is 2.25 x 2^0, not 1.125 x 2^1
        add(f"U: 2.25 - 2.25 + 2^-{j}", {0: (1.5, 1.5), 1: (1.5, -1.5)}, 2.0 ** -j)
    add("T: 2^20 - 1.125^2", {0: (1.125, -1.125)}, 2.0 ** 20)
    add("T: 2^20 + 1.125^2", {0: (1.125, 1.125)}, 2.0 ** 20)
    add("T: -2^20 + 1.125^2", {0: (1.125, 1.125)}, -2.0 ** 20)
    add("T: FLT_MAX - 1", {0: (1.0, -1.0)}, 0x7F7F_FFFF)
    add("T: FLT_MAX + 32*448^2", {k: (448.0, 448.0) for k in range(32)}, 0x7F7F_FFFF)
    add("T: -FLT_MAX - 32*448^2", {k: (448.0, -448.0) for k in range(32)}, 0xFF7F_FFFF)
    add("C: 16 - (1 + 2^-23)", {0: (4.0, 4.0)}, -(1 + 2.0 ** -23))
    add("C: 16 + (1 + 2^-23)", {0: (4.0, 4.0)}, 1 + 2.0 ** -23)
    add("C: -16 + (1 + 2^-23)", {0: (4.0, -4.0)}, 1 + 2.0 ** -23)
    add("R: 1 + 0.75 ulp", {0: (1.0, 1.0)}, 2.0 ** -24 + 2.0 ** -25)
    add("R: -1 - 0.75 ulp", {0: (1.0, -1.0)}, -(2.0 ** -24 + 2.0 ** -25))
    add("R: 1 + 0.5 ulp", {0: (1.0, 1.0)}, 2.0 ** -24)
    add("R: 1 + 1.5 ulp", {0: (1.0, 1.0)}, 2.0 ** -23 + 2.0 ** -24)
    for j in range(1, 32):
        add(f"G: a0b0 = 1, a{j}b{j} = -1, C = 2^-24", {0: (1.0, 1.0), j: (1.0, -1.0)}, 2.0 ** -24)
    for j in range(2, 32):
        big = {0: (448.0, 448.0), 1: (448.0, -448.0)}
        add(f"G: 448^2 - 448^2 + a{j}b{j} = 2^-12", big | {j: (2.0 ** -6, 2.0 ** -6)}, 0)
    tail = 2.0 ** -6 + 2.0 ** -26
    add("Z: 0 * 448 + 1 * -2^-6 + C", {0: (0x00, 448.0), 1: (1.0, -2.0 ** -6)}, tail)
    add("Z: 448 * 0 + 1 * -2^-6 + C", {0: (448.0, 0x00), 1: (1.0, -2.0 ** -6)}, tail)
    add("Z: -0 * 448 + 1 * -2^-6 + C", {0: (0x80, 448.0), 1: (1.0, -2.0 ** -6)}, tail)
    add("Z: 0 * 448, C = 2^-100", {0: (0x00, 448.0)}, 2.0 ** -100)
    add("Z: all zero, C = 2^-100", {}, 2.0 ** -100)
    add("Z: all zero, C = 0x00000001", {}, 0x0000_0001)
    add("Z: all zero, C = 0x80000003", {}, 0x8000_0003)
    add("Z: 32 x 448 * 0, C = 0x00000005", {k: (448.0, 0x00) for k in range(32)}, 0x0000_0005)
    add("S: 2^-9 * 1 + C", {0: (0x01, 1.0)}, -(2.0 ** -9 - 2.0 ** -33))
    add("S: 1 * 2^-9 + C", {0: (1.0, 0x01)}, -(2.0 ** -9 - 2.0 ** -33))
    add("S: 2^-9 * 2^-9 + C", {0: (0x01, 0x01)}, -(2.0 ** -18 - 2.0 ** -42))
    add("S: 0x07 * 0x0F", {0: (0x07, 0x0F)}, 0)
    add("S: 0x07 * 448", {0: (0x07, 448.0)}, 0)
    add("S: 2^-9 * 2^-9 + 1", {0: (0x01, 0x01)}, 1.0)
    add("N: a0 = 0x7F", {0: (0x7F, 1.0)}, 1.0)
    add("N: a0 = 0xFF", {0: (0xFF, 1.0)}, 1.0)
    add("N: 0x7F * 0", {0: (0x7F, 0x00)}, 0)
    add("N: 0 * 0xFF", {0: (0x00, 0xFF)}, 0)
    add("N: 0x7F * 1, C = -Inf", {0: (0x7F, 1.0)}, 0xFF80_0000)
    add("N: C = +Inf", {0: (448.0, -448.0)}, 0x7F80_0000)
    add("N: C = -Inf", {0: (448.0, 448.0)}, 0xFF80_0000)
    add("N: all zero, C = +Inf", {}, 0x7F80_0000)
    for c in (0x7FC0_0000, 0xFFC0_0001, 0x7F80_0001, 0x7FFF_FFFF):
        add(f"N: C = {c:08x}", {0: (1.0, 1.0)}, c)
    add("0: all zero, C = -0", {}, 0x8000_0000)
    add("0: 32 x -0 * 0, C = -0", {k: (0x80, 0x00) for k in range(32)}, 0x8000_0000)
    add("0: 32 x -0 * -0, C = -0", {k: (0x80, 0x80) for k in range(32)}, 0x8000_0000)
    add("0: 32 x -0 * 0, C = +0", {k: (0x80, 0x00) for k in range(32)}, 0)
    add("0: 32 x 0 * -1, C = -0", {k: (0x00, 0xB8) for k in range(32)}, 0x8000_0000)
    add("0: 32 x -0 * 448, C = -0", {k: (0x80, 0x7E) for k in range(32)}, 0x8000_0000)
    add("0: 32 x -0 * 2^-9, C = -0", {k: (0x80, 0x01) for k in range(32)}, 0x8000_0000)
    add("0: 1 - 1, C = -0", cancel, 0x8000_0000)
    add("0: 1 - 1, C = +0", cancel, 0)
    add("0: 1, C = -1", {0: (1.0, 1.0)}, -1.0)
    add("0: -1, C = 1", {0: (1.0, -1.0)}, 1.0)
    add("0: -1 - 1 + 2, C = -0", {0: (1.0, -1.0), 1: (1.0, -1.0), 2: (2.0, 1.0)}, 0x8000_0000)
    add("0: 1 - 1, C = -2^-30", cancel, -2.0 ** -30)
    add("0: -1 + 1, C = -2^-100", {0: (1.0, -1.0), 1: (1.0, 1.0)}, -2.0 ** -100)
    table = {float(e4m3_value(np.uint8(x))): x for x in range(256) if x & 0x7F != 0x7F and x != 0x80}
    code = lambda x: x if isinstance(x, int) else table[x]
    a, b = np.zeros((2, len(cases), 32), np.uint8)
    for i, (_, pairs, _) in enumerate(cases):
        for k, (x, y) in pairs.items():
            a[i, k], b[i, k] = code(x), code(y)
    c = np.array([x if isinstance(x, int) else f32_bits(x) for _, _, x in cases], np.uint32)
    return [name for name, _, _ in cases], a, b, c


def gen_bits(rng, n):
    a, b = rng.integers(0, 256, (2, n, 32), dtype=np.uint8)
    return a, b, rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)


def gen_normal(rng, n):
    field = lambda: np.clip(np.rint(rng.normal(7, 2, (n, 32))), 0, 15)
    a, b = codes(rng, field()), codes(rng, field())
    return a, b, f32_bits(rng.normal(0, 8, n))


def gen_wide(rng, n):
    a, b = codes(rng, rng.integers(0, 16, (n, 32))), codes(rng, rng.integers(0, 16, (n, 32)))
    c = fp32_field(rng, 87, 160, n)
    return a, b, np.where(rng.random(n) < 0.1, 0, c).astype(np.uint32)


def gen_cancel(rng, n):
    """Elements 16..31 cancel 0..15 exactly; up to 3 of them are then replaced, and C cancels
    most of the rest (C = -a.b rounded to FP32, then moved by up to 4 ULP) or is small."""
    field = lambda: np.clip(np.rint(rng.normal(7, 3, (n, 16))), 0, 15)
    a, b = codes(rng, field()), codes(rng, field())
    perm = rng.permuted(np.tile(np.arange(16), (n, 1)), axis=1)
    a = np.concatenate([a, np.take_along_axis(a, perm, 1)], 1)
    b = np.concatenate([b, np.take_along_axis(b, perm, 1) ^ 0x80], 1)
    for _ in range(3):
        k, hit = rng.integers(0, 32, n), rng.random(n) < 0.5
        a[hit, k[hit]] = codes(rng, rng.integers(0, 16, hit.sum()))
    c = f32_bits(-exact_dot(a, b)).astype(np.int64) + rng.integers(-4, 5, n)
    c = np.where(c & 0x7FFF_FFFF >= 0x7F80_0000, 0, c).astype(np.uint32)
    small = fp32_field(rng, 60, 120, n)
    pick = rng.random(n)
    return a, b, np.where(pick < 0.6, c, np.where(pick < 0.8, small, 0)).astype(np.uint32)


def gen_cspecial(rng, n):
    """Special and extreme C; a quarter of the vectors have only zero products."""
    a, b, _ = gen_normal(rng, n)
    zero = rng.random(n) < 0.25
    a[zero] = rng.integers(0, 2, (zero.sum(), 32)) << 7
    menu = np.array([0, 0x8000_0000, 1, 0x8000_0001, 0x007F_FFFF, 0x807F_FFFF, 0x0080_0000, 0x8080_0000,
                     0x7F7F_FFFF, 0xFF7F_FFFF, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0xFFC0_0000,
                     0x7F80_0001, 0x7FFF_FFFF], np.uint32)
    kind = rng.integers(0, 4, n)
    c = np.select([kind == 0, kind == 1, kind == 2],
                  [menu[rng.integers(0, len(menu), n)],
                   fp32_field(rng, 0, 0, n),  # subnormal (or zero)
                   (rng.integers(0, 2, n).astype(np.uint32) << 31) | rng.integers(0, 255, n).astype(np.uint32) << 23],
                  fp32_field(rng, 1, 254, n))
    return a, b, c.astype(np.uint32)


def gen_sparse(rng, n):
    a, b = codes(rng, rng.integers(0, 16, (n, 32))), codes(rng, rng.integers(0, 16, (n, 32)))
    for x in (a, b):
        zero = rng.random((n, 32)) < 0.6
        x[zero] = rng.integers(0, 2, zero.sum()) << 7
    c = fp32_field(rng, 80, 140, n)
    return a, b, np.where(rng.random(n) < 0.3, 0, c).astype(np.uint32)


def gen_subnormal(rng, n):
    field = lambda: np.where(rng.random((n, 32)) < 0.6, 0, rng.integers(1, 5, (n, 32)))
    a, b = codes(rng, field()), codes(rng, field())
    c = fp32_field(rng, 100, 120, n)
    return a, b, np.where(rng.random(n) < 0.3, 0, c).astype(np.uint32)


SETS = {  # name: (generator, seed, count)
    "bits_s1": (gen_bits, 1, 100_000),
    "normal_s2": (gen_normal, 2, 150_000),
    "wide_s3": (gen_wide, 3, 150_000),
    "cancel_s4": (gen_cancel, 4, 150_000),
    "cspecial_s5": (gen_cspecial, 5, 100_000),
    "sparse_s6": (gen_sparse, 6, 100_000),
    "subnormal_s7": (gen_subnormal, 7, 100_000),
}


def records(a, b, c):
    return np.concatenate([a, b, np.asarray(c, "<u4")[:, None].view(np.uint8)], 1).tobytes()


def read_vec(path):
    """(header, a, b, c, columns) of an fp8-vectors-v1 file, plain or gzip."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    nl = raw.index(b"\n")
    hdr = json.loads(raw[:nl])
    n, body = hdr["count"], np.frombuffer(raw, np.uint8, offset=nl + 1)
    rec = body[:68 * n].reshape(n, 68)
    cols = {name: body[68 * n + 4 * n * k:68 * n + 4 * n * (k + 1)].view("<u4").astype(np.uint32)
            for k, name in enumerate(hdr["columns"])}
    return hdr, rec[:, :32], rec[:, 32:64], rec[:, 64:].copy().view("<u4")[:, 0].astype(np.uint32), cols


def main(argv):
    if argv[:1] == ["gen"] and len(argv) == 2:
        out = Path(argv[1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "directed.bin").write_bytes(records(*directed()[1:]))
        for name, (gen, seed, n) in SETS.items():
            (out / f"{name}.bin").write_bytes(records(*gen(np.random.default_rng(seed), n)))
        return 0
    if argv[:1] == ["records"] and len(argv) == 3:
        _, a, b, c, _ = read_vec(argv[1])
        Path(argv[2]).write_bytes(records(a, b, c))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

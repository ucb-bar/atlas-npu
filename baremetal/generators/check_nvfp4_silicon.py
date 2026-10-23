#!/usr/bin/env python3
"""Check the NVFP4 MXU arithmetic (npu_model/configs/nvfp4.py) against NVIDIA silicon outputs.

    python3 -I baremetal/generators/check_nvfp4_silicon.py            # nvfp4_silicon/*.vec.gz
    python3 -I baremetal/generators/check_nvfp4_silicon.py FILE ...   # .vec or .vec.gz, kind 0

Each vector is one 64-element NVFP4 dot product plus an FP32 C (format: nvfp4_silicon/README.md).
nvfp4.dot() is compared with every silicon column (rtx5090, thor); any NaN equals any NaN.
A disagreement on a vector of the known gap (an all-zero 16-element block sets the alignment;
docs/nvfp4-mxu.md, and ucb-bar/nvfp4@0c611e1 docs/architecture.html section 4, "Model accuracy")
is listed separately. Any other disagreement makes the exit status 1. The 1-ULP differences
documented there are all MXFP4, so here they are other disagreements.
"""
import gzip
import importlib.util
import json
import struct
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "nvfp4", Path(__file__).resolve().parents[2] / "npu-model" / "npu_model" / "configs" / "nvfp4.py")
nvfp4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nvfp4)

VECTORS = Path(__file__).resolve().parent / "nvfp4_silicon"
SILICON = ("rtx5090", "thor")
ZERO_BLOCK = "all-zero 16-element block sets the alignment"


def read_vec(path):
    """(header, vectors, columns) of an nvfp4-vectors-v1 file, plain or gzip."""
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    nl = raw.index(b"\n")
    hdr = json.loads(raw[:nl])
    n, body = hdr["count"], raw[nl + 1:]
    nib = lambda bs: [(bs[i // 2] >> 4 * (i % 2)) & 15 for i in range(64)]
    vecs = []
    for i in range(n):
        r = body[76 * i:76 * i + 76]
        vecs.append(dict(a=nib(r[0:32]), b=nib(r[32:64]), sa=list(r[64:68]), sb=list(r[68:72]),
                         c=struct.unpack("<I", r[72:76])[0]))
    cols = {}
    for k, name in enumerate(hdr["columns"]):
        off = 76 * n + 4 * n * k
        cols[name] = list(struct.unpack(f"<{n}I", body[off:off + 4 * n]))
    return hdr, vecs, cols


def is_nan(x):
    return (x >> 23) & 255 == 255 and x & 0x7F_FFFF != 0


def same(x, y):
    return x == y or (is_nan(x) and is_nan(y))


def zero_block_sets_alignment(v):
    """Some 16-element block has only zero products, and its scale exponent e_A + e_B is above
    those of C and of every block with a nonzero product, so it alone sets emax
    (docs/nvfp4-mxu.md, Arithmetic)."""
    def e(x):  # UE4M3 exponent after normalization (docs/nvfp4-mxu.md)
        x &= 127
        return (x >> 3) - 7 if x >> 3 else (-6 if x & 7 else -126)
    c = v["c"]
    if any(x & 127 == 127 for x in v["sa"] + v["sb"]) or (c >> 23) & 255 == 255:
        return False
    zero = [all(x & 7 == 0 or y & 7 == 0 for x, y in zip(v["a"][16 * g:16 * g + 16], v["b"][16 * g:16 * g + 16]))
            for g in range(4)]
    if all(zero) or not any(zero):
        return False
    xs = [e(v["sa"][g]) + e(v["sb"][g]) for g in range(4)]
    top = max([max((c >> 23) & 255, 1) - 127] + [xs[g] for g in range(4) if not zero[g]])
    return any(zero[g] and xs[g] > top for g in range(4))


def describe(v):
    nib = lambda xs: sum((x & 15) << 4 * i for i, x in enumerate(xs))
    return (f"a={nib(v['a']):064x} b={nib(v['b']):064x} sa={bytes(v['sa']).hex()} "
            f"sb={bytes(v['sb']).hex()} c={v['c']:08x}")


def main(paths):
    known, other, total, n_gap = [], [], 0, 0
    for path in paths:
        hdr, vecs, cols = read_vec(path)
        name = hdr.get("name", Path(path).stem)
        if hdr["kind"] != 0:
            print(f"{name}: kind {hdr['kind']} is not NVFP4, skipped")
            continue
        full = hdr.get("indices", range(len(vecs)))  # index in the full set, for subsets
        sil = [c for c in SILICON if c in cols]
        model = [nvfp4.dot(v["a"], v["b"], v["sa"], v["sb"], v["c"]) for v in vecs]
        gap = [zero_block_sets_alignment(v) for v in vecs]
        n_gap += sum(gap)
        ok = dict.fromkeys(sil, 0)
        for i, m in enumerate(model):
            bad = [c for c in sil if not same(m, cols[c][i])]
            for c in sil:
                ok[c] += c not in bad
            if bad:
                cls = ZERO_BLOCK if gap[i] else None
                outs = " ".join(f"{c}={cols[c][i]:08x}" for c in sil)
                (known if cls else other).append(
                    f"  {name} #{full[i]}: model={m:08x} {outs} [{cls or 'unexplained'}]\n    {describe(vecs[i])}")
        total += len(vecs) * len(sil)
        print(f"{name:20s} {len(vecs):>7} vectors" + "".join(f"   model == {c} {ok[c]:>7}" for c in sil))
    print(f"vectors where an all-zero block sets the alignment: {n_gap}")
    for title, rows in (("known gaps (documented)", known), ("other disagreements", other)):
        print(f"{title}: {len(rows)} vectors")
        for row in rows:
            print(row)
    print(f"{'PASS' if not other else 'FAIL'}: {total} comparisons; vectors off silicon: "
          f"{len(known)} known gap, {len(other)} other")
    return 1 if other else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(VECTORS.glob("*.vec*"))))

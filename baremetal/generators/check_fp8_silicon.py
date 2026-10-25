#!/usr/bin/env python3
"""Check the FP8 MXU arithmetic (npu_model/configs/fp8_mma.py) against NVIDIA silicon outputs.

    python3 -I baremetal/generators/check_fp8_silicon.py            # fp8_silicon/*.vec.gz
    python3 -I baremetal/generators/check_fp8_silicon.py FILE ...   # .vec or .vec.gz

Each vector is one 32-element E4M3 dot product plus an FP32 C (format: fp8_silicon/README.md).
fp8_mma.dot() is compared bit for bit with every silicon column; NaN is 0x7FFFFFFF in both.
Any disagreement makes the exit status 1.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VECTORS = Path(__file__).resolve().parent / "fp8_silicon"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fp8_mma = load("fp8_mma", ROOT / "npu-model" / "npu_model" / "configs" / "fp8_mma.py")
fp8_vectors = load("fp8_vectors", VECTORS / "fp8_vectors.py")


def main(paths):
    bad, total = 0, 0
    for path in paths:
        hdr, a, b, c, cols = fp8_vectors.read_vec(path)
        name = hdr.get("name", Path(path).stem)
        full = hdr.get("indices", range(len(c)))  # index in the full set, for subsets
        model = fp8_mma.dot(a, b, c)
        print(f"{name:14s} {len(c):>7} vectors" + "".join(
            f"   model == {col} {(model == out).sum():>7}" for col, out in cols.items()))
        for col, out in cols.items():
            for i in (model != out).nonzero()[0]:
                bad += 1
                print(f"  {name} #{full[i]}: model={model[i]:08x} {col}={out[i]:08x}\n"
                      f"    a={bytes(a[i]).hex()} b={bytes(b[i]).hex()} c={c[i]:08x}")
            total += len(c)
    print(f"{'PASS' if not bad else 'FAIL'}: {total} comparisons, {bad} disagreements")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(VECTORS.glob("*.vec*"))))

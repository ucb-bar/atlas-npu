#!/usr/bin/env python3
"""Check the FP8 and BF16 conversions (npu_model/configs/cvt.py) against NVIDIA silicon outputs.

    python3 -I baremetal/generators/check_cvt_silicon.py            # fp8_silicon/*.cvt.gz
    python3 -I baremetal/generators/check_cvt_silicon.py FILE ...

Each file holds the outputs of one conversion (format: fp8_silicon/README.md). The model is
compared bit for bit with the rtx5090 column. For a runs file, the model is compared at the first
and the last input of every run. Any disagreement makes the exit status 1.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VECTORS = Path(__file__).resolve().parent / "fp8_silicon"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cvt = load("cvt", ROOT / "npu-model" / "npu_model" / "configs" / "cvt.py")
cvt_vectors = load("cvt_vectors", VECTORS / "cvt_vectors.py")


def vectors(hdr, cols):
    """(x, e, model, rtx5090) of every vector of one file; x is the input bits."""
    n, name, gpu = hdr["count"], hdr["name"], cols["rtx5090"]
    i = np.arange(n, dtype=np.uint32)
    if name == "vpack":
        x, e = i & 0xFFFF, i >> 16
        return x, e, cvt.e4m3_rn_satfinite(x << 16, e), gpu
    if name == "vunpack":
        x, e = i & 0xFF, i >> 8
        return x, e, cvt.bf16_rne(cvt.e4m3_to_f32(x), e), gpu
    if name == "push":
        return i, np.full(n, 127), cvt.e4m3_to_f32(i), gpu
    if name == "pop":
        return cols["x"], cols["e"], cvt.e4m3_rn_satfinite(cols["x"], cols["e"]), gpu
    x = np.concatenate([cols["x"], np.append(cols["x"][1:] - 1, 0xFFFF_FFFF)])
    model = {"e4m3_runs": cvt.e4m3_rn_satfinite, "bf16_runs": cvt.bf16_rne}[name](x)
    return x, np.full(2 * n, 127), model, np.tile(gpu, 2)


def main(paths):
    bad, total = 0, 0
    for path in paths:
        hdr, cols = cvt_vectors.read_cvt(path)
        x, e, model, gpu = vectors(hdr, cols)
        same = (model == gpu).sum()
        print(f"{hdr['name']:10s} {len(gpu):>9} vectors   model == rtx5090 {same:>9}")
        for k in (model != gpu).nonzero()[0]:
            bad += 1
            print(f"  {hdr['name']} x={int(x[k]):08x} e={int(e[k])}: "
                  f"model={int(model[k]):x} rtx5090={int(gpu[k]):x}")
        total += len(gpu)
    print(f"{'PASS' if not bad else 'FAIL'}: {total} comparisons, {bad} disagreements")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or sorted(VECTORS.glob("*.cvt.gz"))))

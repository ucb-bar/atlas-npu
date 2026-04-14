#!/usr/bin/env python3
"""
gen_vpu_test_vectors.py — Per-family wrapper around
`software_models.vpu.gen_vectors` that produces the six golden files
consumed by the new `VectorEngineTop<Family>VectorTest.scala` suites.

Mirrors `scripts/gen_mxu_vectors.py` in shape: thin wrapper, imports the
functional model out of `baremetal/generators/software_models/`, writes
into `src/test/resources/vpu_test_vectors/`.

Usage (from project root):
    python3 scripts/gen_vpu_test_vectors.py
    python3 scripts/gen_vpu_test_vectors.py --num 30 --seed 12345
    python3 scripts/gen_vpu_test_vectors.py --families binary unary_math

Output files (one per family):
    src/test/resources/vpu_test_vectors/vpu_binary_vectors.txt
    src/test/resources/vpu_test_vectors/vpu_unary_simple_vectors.txt
    src/test/resources/vpu_test_vectors/vpu_unary_math_vectors.txt
    src/test/resources/vpu_test_vectors/vpu_row_reduce_vectors.txt
    src/test/resources/vpu_test_vectors/vpu_col_reduce_vectors.txt
    src/test/resources/vpu_test_vectors/vpu_vli_vectors.txt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Put baremetal/generators on sys.path so `software_models.vpu...` resolves
# exactly the way `scripts/gen_mxu_vectors.py` reaches `software_models.mxu*`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_GENERATORS_DIR = _REPO_ROOT / "baremetal" / "generators"
sys.path.insert(0, str(_GENERATORS_DIR))

from software_models.vpu.gen_vectors import main as gen_vectors_main  # noqa: E402

# Family → ops mapping lives here (NOT in gen_vectors.py) so the low-level
# CLI stays reusable and the family map is the wrapper's concern only.
FAMILIES: dict[str, list[str]] = {
    "binary":       ["add", "sub", "mul", "pairmax", "pairmin"],
    "unary_simple": ["mov", "relu", "rcp"],
    "unary_math":   ["sqrt", "log", "exp", "exp2", "square", "cube", "sin", "cos", "tanh"],
    "row_reduce":   ["rsum", "rmin", "rmax"],
    "col_reduce":   ["csum", "cmin", "cmax"],
    "vli":          ["vliOne", "vliRow", "vliCol", "vliAll"],
}

_DEFAULT_OUT_DIR = _REPO_ROOT / "src" / "test" / "resources" / "vpu_test_vectors"


def _run_family(family: str, out_dir: Path, num: int, seed: int, num_lanes: int) -> None:
    ops = FAMILIES[family]
    out_path = out_dir / f"vpu_{family}_vectors.txt"
    argv = [
        "--out", str(out_path),
        "--num", str(num),
        "--num-lanes", str(num_lanes),
        "--seed", str(seed),
        "--ops", *ops,
    ]
    print(f"[gen_vpu_test_vectors] family={family} → {out_path}")
    rc = gen_vectors_main(argv)
    if rc != 0:
        raise SystemExit(f"gen_vectors_main returned {rc} for family={family}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gen_vpu_test_vectors.py",
        description="Generate per-family VPU golden vector files.",
    )
    ap.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR),
                    help="Directory to write vpu_<family>_vectors.txt files into")
    ap.add_argument("--num", type=int, default=30,
                    help="Random cases per family (default 30)")
    ap.add_argument("--num-lanes", type=int, default=16,
                    help="Lanes per instruction")
    ap.add_argument("--seed", type=int, default=12345,
                    help="Seed for both random and numpy RNG")
    ap.add_argument("--families", nargs="+", choices=list(FAMILIES.keys()),
                    default=list(FAMILIES.keys()),
                    help="Subset of families to generate (default: all)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for family in args.families:
        _run_family(family, out_dir, args.num, args.seed, args.num_lanes)

    print(f"[gen_vpu_test_vectors] wrote {len(args.families)} families to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

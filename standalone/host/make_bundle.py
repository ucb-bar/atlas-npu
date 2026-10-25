"""Assemble the baremetal tests into host-run bundles for run_tests.py.

    cd baremetal
    uv run --locked --project ../npu-model --python 3.14 python ../standalone/host/make_bundle.py \
        --out ../build/bundle/orig [--no-delay] [--gen-python PY] [test ...]

A bundle is one JSON file per test: the program words, the DRAM preload and check words (from
the test's generator, else its inline @DRAM / @CHECK_DRAM lines), and its @PERF directives.
This is the data the Chipyard C harness embeds. --no-delay assembles each test with every
DELAY removed, which the interlocked core must not need. Generators run with --gen-python (they
need torch).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BAREMETAL = Path(__file__).resolve().parents[2] / "baremetal"
sys.path.insert(0, str(BAREMETAL))
import atlas_asm  # noqa: E402
from npu_model.isa import ISA_VERSION  # noqa: E402

LABELS = re.compile(r"^(\s*(?:[A-Za-z_.$][\w.$]*:\s*)*)(.*)$")


def strip_delays(source):
    """Remove every DELAY. Returns (source, removed)."""
    out, removed = [], 0
    for line in source.splitlines():
        code = line.split("#", 1)[0]
        labels, rest = LABELS.match(code).groups()
        if rest.strip().lower().split()[:1] == ["delay"]:
            out.append(labels.rstrip())
            removed += 1
            continue
        out.append(line)
    return "\n".join(out) + "\n", removed


def generator_for(test):
    """The Makefile's rule: gen_<test>.py, else gen_<test with mxu1_ -> mxu_>.py."""
    for base in (test, re.sub(r"mxu1_", "mxu_", test)):
        path = BAREMETAL / "generators" / f"gen_{base}.py"
        if path.exists():
            return path
    return None


def golden_for(test, source, gen_python, cache):
    gen = generator_for(test)
    if gen is None:
        inline = atlas_asm._parse_inline_golden(source, 32)
        return None if inline is None else inline[1:]
    cached = cache / f"{test}.json"
    if not cached.exists():
        out = subprocess.run([gen_python, str(gen)], cwd=BAREMETAL, check=True,
                             capture_output=True, text=True).stdout
        cached.write_text(out)
    dram_base = atlas_asm._parse_dram_base(source)
    return atlas_asm._parse_golden_json(str(cached), dram_base if dram_base is not None else 0, 32)


def pairs(addrs, words):
    """[[addr, [words]]] runs of consecutive words."""
    out = []
    for a, w in zip(addrs, words):
        if out and out[-1][0] + 4 * len(out[-1][1]) == a:
            out[-1][1].append(w)
        else:
            out.append([a, [w]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tests", nargs="*", help="test names (default: every assembly/*.S)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-delay", action="store_true")
    ap.add_argument("--gen-python", default=sys.executable)
    ap.add_argument("--gen-cache", default=None, help="directory for generator JSON (default <out>/../gen)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(a.gen_cache) if a.gen_cache else out.parent / "gen"
    cache.mkdir(parents=True, exist_ok=True)
    tests = a.tests or sorted(p.stem for p in (BAREMETAL / "assembly").glob("*.S"))

    failed = []
    for t in tests:
        source = (BAREMETAL / "assembly" / f"{t}.S").read_text()
        removed = 0
        try:
            golden = golden_for(t, source, a.gen_python, cache)
            if a.no_delay:
                source, removed = strip_delays(source)
            program = [w & 0xFFFF_FFFF for w in atlas_asm.assemble(source)]
        except Exception as e:  # report and continue with the other tests
            failed.append(t)
            print(f"  {t}: {type(e).__name__}: {e}")
            continue
        timeout = re.search(r"^\s*#\s*@TIMEOUT\s+(\d+)", source, re.M)
        bundle = {
            "name": t,
            "isa": list(ISA_VERSION),
            "variant": "no-delay" if a.no_delay else "orig",
            "delays_removed": removed,
            "program": program,
            "preload": pairs(golden[0], golden[1]) if golden else [],
            "check": pairs(golden[2], golden[3]) if golden else [],
            "perf": {"threshold": atlas_asm._parse_perf_threshold(source),
                     "report": atlas_asm._parse_perf_report(source),
                     "macs_mxu1": atlas_asm._parse_perf_macs(source)},
            "timeout": int(timeout.group(1)) if timeout else None,
        }
        (out / f"{t}.json").write_text(json.dumps(bundle))
        print(f"  {t}: {len(program)} words, {sum(len(w) for _, w in bundle['preload'])} preload, "
              f"{sum(len(w) for _, w in bundle['check'])} check"
              + (f", {removed} DELAY removed" if a.no_delay else ""))
    print(f"{len(tests) - len(failed)}/{len(tests)} bundled into {out}")
    if failed:
        print("failed: " + " ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Run bundled baremetal tests (make_bundle.py) on the standalone SoC.

    python standalone/host/run_tests.py --sim build/vl_soc/atlas_sim -j 32 --out OUT BUNDLE.json ...
    python standalone/host/run_tests.py --board [--port TTY] --out OUT BUNDLE.json ...

--sim runs each test in a fresh Verilator process (in parallel with -j). An RTL assertion, which
with the interlock means a hazard got through, fails the test. --board runs the tests one after
another over UART TSI. Results: OUT/results.csv and a summary on stdout.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_host import Atlas, LinkError, Result, SimLink, TsiLink  # noqa: E402

FIELDS = ["name", "variant", "status", "cycles", "instret", "dbg0", "dbg1", "util", "checked", "mismatches",
          "wall_s", "detail"]


def run_sim(args):
    binary, path, budget = args
    test = json.loads(Path(path).read_text())
    t0 = time.time()
    link = None
    try:
        link = SimLink(binary)
        r = Atlas(link).run(test, budget)
    except LinkError as e:
        r = Result(test["name"], "ERROR", variant=test.get("variant", ""), detail=" ".join(str(e).split())[:400])
    finally:
        if link:
            link.close()
    return r, time.time() - t0


def row(r, wall):
    d = {k: getattr(r, k) for k in FIELDS if hasattr(r, k)}
    d["util"] = " ".join(f"{k}={v}%" for k, v in r.util.items())
    d["wall_s"] = f"{wall:.1f}"
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", nargs="+")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", metavar="BINARY")
    g.add_argument("--board", action="store_true")
    ap.add_argument("--port")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--budget", type=float, default=None,
                    help="completion wait: cycles for --sim (default 4e6), seconds for --board (default 20)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(a.bundles)
    rows = []
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()

        def report(i, r, wall):
            rows.append(r)
            w.writerow(row(r, wall))
            f.flush()
            util = " ".join(f"{k}={v}%" for k, v in r.util.items())
            print(f"[{i + 1}/{len(paths)}] {r.status:7s} {r.name} ({r.variant})  cycles={r.cycles} {util} {r.detail[:100]}",
                  flush=True)

        if a.sim:
            budget = a.budget or 4e6
            with ProcessPoolExecutor(max_workers=a.jobs) as pool:
                for i, (r, wall) in enumerate(pool.map(run_sim, [(a.sim, p, budget) for p in paths])):
                    report(i, r, wall)
        else:
            budget = a.budget or 20.0
            link = TsiLink(a.port)
            try:
                atlas = Atlas(link)
                print(f"board on {link.port}: DRAM {atlas.dram_bytes >> 20} MiB, features 0x{atlas.features:x}")
                for i, p in enumerate(paths):
                    test = json.loads(Path(p).read_text())
                    t0 = time.time()
                    try:
                        r = atlas.run(test, budget)
                    except Exception as e:
                        r = Result(test["name"], "ERROR", variant=test.get("variant", ""),
                                   detail=f"{type(e).__name__}: {e}")
                    report(i, r, time.time() - t0)
            finally:
                link.close()

    passed = sum(r.status == "PASS" for r in rows)
    print(f"\nSUMMARY: {passed}/{len(rows)} passed")
    for r in rows:
        if r.status != "PASS":
            print(f"  {r.status:7s} {r.name} ({r.variant}): {r.detail[:160]}")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()

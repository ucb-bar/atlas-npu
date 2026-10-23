"""Run bundled baremetal tests (make_bundle.py) on the standalone SoC.

    python standalone/host/run_tests.py --sim build/vl_soc/atlas_sim -j 32 --out OUT BUNDLE.json ...
    python standalone/host/run_tests.py --board [--port TTY] --out OUT BUNDLE.json ...
    python standalone/host/run_tests.py --tsi-sim build/vl_tsi/atlas_tsi_sim --out OUT BUNDLE.json ...
    python standalone/host/run_tests.py --eth [IP[:PORT]] --out OUT BUNDLE.json ...
    python standalone/host/run_tests.py --eth-sim build/vl_eth/atlas_eth_sim -j 16 --out OUT BUNDLE.json ...

--sim runs each test in a fresh Verilator process (in parallel with -j). An RTL assertion, which
with the interlock means a hazard got through, fails the test. --board runs the tests one after
another over UART TSI; --tsi-sim does the same against the UART-level model. --eth runs them one after
another over TSI over UDP (default 172.20.1.18:21587); --eth-sim runs each test in a fresh Ethernet model
on its own UDP port, in parallel with -j. Results: OUT/results.csv and a summary on stdout.
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
from atlas_host import (ETH_IP_DEFAULT, ETH_PORT, PERF_NAMES, Atlas, EthLink, LinkError, Result,  # noqa: E402
                        SimLink, TsiLink)
import expect  # noqa: E402

FIELDS = ["name", "variant", "status", "cycles", "instret", "dbg0", "dbg1", "util", "checked", "mismatches",
          "run_s", "wall_s", "detail"]


def run_sim(args):
    """One test in a fresh model: --sim (SimLink) or --eth-sim (EthLink). Returns (result, wall, link stats)."""
    kind, binary, path, budget = args
    test = json.loads(Path(path).read_text())
    t0 = time.time()
    link = None
    try:
        link = SimLink(binary) if kind == "sim" else EthLink(cosim=binary)
        r = Atlas(link).run(test, budget)
    except LinkError as e:
        r = Result(test["name"], "ERROR", variant=test.get("variant", ""), detail=" ".join(str(e).split())[:400])
    finally:
        if link:
            link.close()
    return r, time.time() - t0, getattr(link, "stats", {})


def row(r, wall):
    d = {k: getattr(r, k) for k in FIELDS if hasattr(r, k)}
    d["util"] = " ".join(f"{k}={v}%" for k, v in r.util.items())
    d["wall_s"] = f"{wall:.1f}"                 # the whole test, with loading DRAM and IMEM
    d["run_s"] = f"{r.run_s:.4f}"               # from the start to the completion poll
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundles", nargs="*")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sim", metavar="BINARY")
    g.add_argument("--board", action="store_true")
    g.add_argument("--tsi-sim", metavar="BINARY", help="PyUARTSI against the UART-level model, one link")
    g.add_argument("--eth", nargs="?", const=ETH_IP_DEFAULT, metavar="IP[:PORT]",
                   help=f"the board over TSI over UDP (default {ETH_IP_DEFAULT}:{ETH_PORT})")
    g.add_argument("--eth-sim", metavar="BINARY", help="the Ethernet model, one process per test")
    ap.add_argument("--port")
    ap.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--budget", type=float, default=None,
                    help="completion wait: cycles for --sim (default 4e6), seconds otherwise "
                         "(default 20 on the board, 600 on --tsi-sim and --eth-sim)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", choices=sorted(expect.UNITS),
                    help="judge against this configuration: a test needing a unit it lacks must be ILLEGAL")
    a = ap.parse_args()
    if a.eth and (a.eth.endswith(".json") or os.path.isfile(a.eth)):    # "--eth BUNDLE.json ...": no IP given
        a.bundles.insert(0, a.eth)
        a.eth = ETH_IP_DEFAULT
    if not a.bundles:
        ap.error("no bundles")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(a.bundles)
    rows = []
    stats = {}
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

        if a.sim or a.eth_sim:
            kind, binary = ("sim", a.sim) if a.sim else ("eth", a.eth_sim)
            budget = a.budget or (4e6 if a.sim else 600.0)
            with ProcessPoolExecutor(max_workers=a.jobs) as pool:
                jobs = [(kind, binary, p, budget) for p in paths]
                for i, (r, wall, s) in enumerate(pool.map(run_sim, jobs)):
                    report(i, r, wall)
                    for k, v in s.items():
                        stats[k] = stats.get(k, 0) + v
        else:
            budget = a.budget or (600.0 if a.tsi_sim else 20.0)
            if a.eth:
                ip, _, port = a.eth.partition(":")
                try:
                    link = EthLink(ip, int(port or ETH_PORT))
                except LinkError as e:
                    sys.exit(f"link: {e}")
                where = f"{link.name} (UDP)"
            else:
                link = TsiLink(cosim=a.tsi_sim) if a.tsi_sim else TsiLink(a.port)
                where = link.port
            try:
                atlas = Atlas(link)
                print(f"board on {where}: DRAM {atlas.dram_bytes >> 20} MiB, features 0x{atlas.features:x}")
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
                if getattr(link, "read_retries", 0):
                    print(f"link: {link.read_retries} read chunk(s) retried after a lost byte")
                stats = getattr(link, "stats", {})
                link.close()
        if any(stats.values()):
            print("link: " + ", ".join(f"{k} {v}" for k, v in stats.items()))

    if any(r.perf for r in rows):     # the performance counters of every run
        with open(out / "perf.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "variant", "status"] + PERF_NAMES)
            for r in rows:
                w.writerow([r.name, r.variant, r.status] + [r.perf.get(n, "") for n in PERF_NAMES])

    passed = sum(r.status == "PASS" for r in rows)
    print(f"\nSUMMARY: {passed}/{len(rows)} passed")
    if a.config:
        want = {r.name: expect.expected(r.name, a.config) for r in rows}
        illegal = sum(w == "ILLEGAL" for w in want.values())
        bad = [r for r in rows if r.status != want[r.name]]
        print(f"CONFIG {a.config}: {len(rows) - len(bad)}/{len(rows)} as expected "
              f"({len(rows) - illegal} must pass, {illegal} must be illegal)")
        for r in bad:
            print(f"  {r.status:7s} {r.name} ({r.variant}), expected {want[r.name]}: {r.detail[:140]}")
        sys.exit(0 if not bad else 1)
    for r in rows:
        if r.status != "PASS":
            print(f"  {r.status:7s} {r.name} ({r.variant}): {r.detail[:160]}")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()

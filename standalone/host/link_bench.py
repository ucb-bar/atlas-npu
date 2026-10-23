"""Throughput and latency of the host links (atlas_host.py).

    python3 standalone/host/link_bench.py --eth [IP[:PORT]] --uart [TTY]
    python3 standalone/host/link_bench.py --eth-sim build/vl_eth/atlas_eth_sim --tsi-sim build/vl_tsi/atlas_tsi_sim

For each link: write random words into DRAM with write_words (and with write_bulk where the link has
it), read them back and compare, then time single-word reads (sys_regs ID) and writes (SCRATCH).
UART TSI writes are not acknowledged, so their latency is the time to send them. --mb sets the size
(default 16 MiB on Ethernet, 256 KiB on the UART, 64 KiB on the Ethernet model, 1 KiB on the UART
model). Results: a table on stdout.
"""
import argparse
import random
import statistics
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from atlas_host import (ETH_IP_DEFAULT, ETH_PORT, SYS_ETH_DROP, SYS_ETH_DUP, SYS_ETH_RUN, SYS_ETH_RXERR,  # noqa: E402
                        SYS_ETH_STATUS, SYS_ID, SYS_SCRATCH, Atlas, EthLink, TsiLink)

DEFAULT_MB = {"eth": 16, "uart": 0.25, "eth-sim": 1 / 16, "tsi-sim": 1 / 1024}
DEFAULT_SMALL = {"eth": 2000, "uart": 200, "eth-sim": 50, "tsi-sim": 5}


def open_link(kind, arg, a):
    if kind == "eth":
        ip, _, port = arg.partition(":")
        return EthLink(ip, int(port or ETH_PORT), window=a.window, timeout=a.timeout)
    if kind == "eth-sim":
        return EthLink(cosim=arg, window=a.window, timeout=a.timeout)
    if kind == "uart":
        return TsiLink(arg or None)
    return TsiLink(cosim=arg)


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def latency(fn, n):
    """Median and 99th percentile of n calls, in microseconds."""
    t = sorted(timed(fn, i)[1] * 1e6 for i in range(n))
    return statistics.median(t), t[min(n - 1, int(0.99 * n))]


def bench(link, nbytes, addr, small):
    atlas = Atlas(link)
    n = min(nbytes, atlas.dram_bytes) // 4
    base = atlas.dram(addr)
    rng = random.Random(1)
    row = {"MiB": 4 * n / 2**20}
    words = [rng.getrandbits(32) for _ in range(n)]
    _, t = timed(link.write_words, base, words)
    row["write MB/s"] = 4 * n / t / 1e6
    if getattr(link, "bulk", False):
        words = [rng.getrandbits(32) for _ in range(n)]
        _, t = timed(link.write_bulk, base, struct.pack("<%dI" % n, *words))
        row["bulk MB/s"] = 4 * n / t / 1e6
    got, t = timed(link.read_words, base, n)
    row["read MB/s"] = 4 * n / t / 1e6
    row["errors"] = sum(g != w for g, w in zip(got, words)) + abs(len(got) - n)
    row["read us"] = "%.0f / %.0f" % latency(lambda i: link.read_words(SYS_ID, 1), small)
    row["write us"] = "%.0f / %.0f" % latency(lambda i: link.write_words(SYS_SCRATCH, [i]), small)
    if isinstance(link, EthLink):
        row["link"] = ", ".join(f"{k} {v}" for k, v in link.stats.items() if v)
        status, run, dup, drop = (link.read_words(r, 1)[0] for r in (SYS_ETH_STATUS, SYS_ETH_RUN, SYS_ETH_DUP,
                                                                     SYS_ETH_DROP))
        rxerr = link.read_words(SYS_ETH_RXERR, 1)[0]
        row["board"] = (f"ETH_STATUS 0x{status:x}, run {run}, dup {dup}, drop {drop}, "
                        f"rx dropped {rxerr & 0xFFFF}, bad {rxerr >> 16}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eth", nargs="?", const=ETH_IP_DEFAULT, metavar="IP[:PORT]")
    ap.add_argument("--uart", nargs="?", const="", metavar="TTY", help="UART TSI (default: the VCU118's CP2105)")
    ap.add_argument("--eth-sim", metavar="BINARY")
    ap.add_argument("--tsi-sim", metavar="BINARY")
    ap.add_argument("--mb", type=float, help="MiB to write and read per link")
    ap.add_argument("--small", type=int, help="single-word transactions to time")
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x9000_0000, help="DRAM address (default 0x9000_0000)")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=0.05)
    a = ap.parse_args()
    links = [(k, getattr(a, k.replace("-", "_"))) for k in ("eth", "uart", "eth-sim", "tsi-sim")]
    links = [(k, v) for k, v in links if v is not None]
    if not links:
        ap.error("no link: pass --eth, --uart, --eth-sim or --tsi-sim")

    rows = []
    for kind, arg in links:
        link = open_link(kind, arg, a)
        name = f"{kind} {arg if kind == 'eth' else Path(arg).name if arg else 'CP2105'}"
        try:
            r = bench(link, int((a.mb or DEFAULT_MB[kind]) * 2**20), a.addr, a.small or DEFAULT_SMALL[kind])
        finally:
            link.close()
        rows.append((name, r))
        print(f"{name}: {r}", file=sys.stderr, flush=True)

    cols = ["MiB", "write MB/s", "bulk MB/s", "read MB/s", "errors", "read us", "write us"]
    table = [["link"] + cols] + [[name] + [f"{r[c]:.3g}" if isinstance(r.get(c), float) else str(r.get(c, "-"))
                                           for c in cols] for name, r in rows]
    width = [max(len(t[i]) for t in table) for i in range(len(table[0]))]
    for t in table:
        print("  ".join(s.ljust(w) for s, w in zip(t, width)))
    print("(latency: median / 99th percentile)")
    for name, r in rows:
        for k in ("link", "board"):
            if r.get(k):
                print(f"{name} {k}: {r[k]}")
    sys.exit(1 if any(r["errors"] for _, r in rows) else 0)


if __name__ == "__main__":
    main()

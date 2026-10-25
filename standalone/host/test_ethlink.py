"""Tests of EthLink (atlas_host.py) against a model of the board's TSI over UDP target (docs/tsi-over-udp.md).

    python3 standalone/host/test_ethlink.py [-v] [-k PATTERN]
    python3 standalone/host/test_ethlink.py --serve [PORT] [--loss P] [--ready-first]

MockBoard is a UDP server thread on 127.0.0.1 that runs datagrams by the board's rules (eth_tsi) over a
model of the SoC's address map. Net drops, duplicates, reorders and delays datagrams in one direction.
EXEC_CONTROL writes are logged, so a test can check that every write ran exactly once; a write of 1
"runs a kernel" that sets DBG0 to 1 and halts.

--serve runs MockBoard as a stand-in for the Ethernet model (build/vl_eth/atlas_eth_sim): it prints
"PORT <n>" (for port 0) and "READY", and exits on stdin EOF or SIGTERM. run_tests.py --eth-sim and
link_bench.py --eth-sim accept "python3 test_ethlink.py --serve" wrapped in a script.
"""
import argparse
import heapq
import os
import random
import select
import signal
import socket
import struct
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_host  # noqa: E402
from atlas_host import (ATLS_ID, CSR_BASE, CSR_DBG0, CSR_DBG1, CSR_EXEC_CONTROL, CSR_ISA, CSR_STATUS,  # noqa: E402
                        DRAM_BASE, ETH_BADCMD, ETH_CUT, ETH_DENIED, ETH_DROP, ETH_DUP, ETH_NOSYNC, ETH_OVERFLOW,
                        ETH_PAYLOAD, ETH_PORT, ETH_REPLY_WORDS, ETH_SYNC, ETH_WRITE_WORDS, IMEM_BASE, M32,
                        SYS_DRAM_BYTES, SYS_ETH_DROP, SYS_ETH_DUP, SYS_ETH_IP, SYS_ETH_PORT, SYS_ETH_RUN,
                        SYS_ETH_STATUS, SYS_FEATURES, SYS_ID, SYS_LEDS, SYS_RUN_CYCLES, SYS_SCRATCH, VMEM_BASE, Atlas,
                        EthLink, LinkError)

IMEM_BYTES = 128 << 10
VMEM_BYTES = 1536 << 10


class Net:
    """One direction: each datagram is lost with probability loss, sent twice with probability dup, and
    held back up to delay seconds with probability reorder (a duplicate too), after latency seconds."""

    def __init__(self, loss=0.0, dup=0.0, reorder=0.0, delay=0.003, latency=0.0):
        self.loss, self.dup, self.reorder, self.delay, self.latency = loss, dup, reorder, delay, latency

    def copies(self, rng):
        """Delays of the copies that arrive."""
        out = []
        for i in range(2 if rng.random() < self.dup else 1):
            if rng.random() >= self.loss:
                late = i or rng.random() < self.reorder
                out.append(self.latency + (rng.uniform(0, self.delay) if late else 0.0))
        return out


def done_kernel(board):
    """EXEC_CONTROL = 1: the kernel sets DBG0 to 1 and halts (STATUS halted, reason 2)."""
    board.csr[CSR_DBG0] = 1
    board.csr[CSR_STATUS] = 1 | 2 << 1
    board.sys[SYS_RUN_CYCLES] = 1234


class MockBoard(threading.Thread):
    """The board's protocol engine (spec rules for the FPGA) over sys_regs, IMEM, CSR, VMEM and DRAM.
    exec_s: time to run a request; rx_s: time to receive any request, also one answered DUP or DROP
    (the Ethernet model's GMII time). Requests are served one at a time."""

    def __init__(self, rx=None, tx=None, seed=1, port=0, dram_bytes=16 << 20, exec_s=0.0, rx_s=0.0,
                 kernel=done_kernel, hold=0.0):
        super().__init__(daemon=True)
        self.hold, self._held = hold, None   # hold: a request waits for the next one (the VCU118's receive path)
        self.rx, self.tx = rx or Net(), tx or Net()
        self.rng = random.Random(seed)
        self.exec_s, self.rx_s, self.kernel = exec_s, rx_s, kernel
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.imem, self.vmem, self.dram = bytearray(IMEM_BYTES), bytearray(VMEM_BYTES), bytearray(dram_bytes)
        self.sys = {SYS_ID: ATLS_ID, SYS_DRAM_BYTES: dram_bytes, SYS_FEATURES: 0x7, SYS_ETH_IP: 0x7F00_0001,
                    SYS_ETH_STATUS: 1 << 20 | 1 << 19 | 1 << 18 | 1 << 16 | 1 << 12 | 2 << 10 | 3,
                    SYS_ETH_PORT: ETH_PORT}
        self.csr = {CSR_ISA: 1 << 24 | 0x7}     # ISA 1.0, the units of SYS_FEATURES
        self.exec_log = []          # every value written to EXEC_CONTROL, in order
        self.session = None         # (peer, seq) of the SYNC that opened the session; None: no session
        self.expect = None          # next seq to run in the session
        self.n_exec = self.n_dup = self.n_drop = 0
        self.received = []          # (time, seq) of every request that reached the engine
        self.lock = threading.Lock()
        self._events = []           # (time, n, "rx" | "tx", datagram, peer)
        self._n = 0
        self._halt = threading.Event()
        self.start()

    def stop(self):
        self._halt.set()
        self.join()
        self.sock.close()

    def run(self):
        while not self._halt.is_set():
            wait = 0.02
            if self._events:
                wait = min(wait, max(0.0, self._events[0][0] - time.monotonic()))
            if select.select([self.sock], [], [], wait)[0]:
                while True:
                    try:
                        pkt, peer = self.sock.recvfrom(65536)
                    except (BlockingIOError, ConnectionRefusedError):
                        break
                    if self._held is not None:
                        self._push(0.0, "rx", *self._held)
                        self._held = None
                    if self.rng.random() < self.hold:
                        self._held = (pkt, peer)
                        continue
                    for dt in self.rx.copies(self.rng):
                        self._push(dt, "rx", pkt, peer)
            while self._events and self._events[0][0] <= time.monotonic():
                _, _, kind, pkt, peer = heapq.heappop(self._events)
                if kind == "tx":
                    try:
                        self.sock.sendto(pkt, peer)
                    except OSError:
                        pass
                    continue
                if self.rx_s:
                    time.sleep(self.rx_s)
                reply = self.handle(pkt, peer)
                if reply is not None:
                    for dt in self.tx.copies(self.rng):
                        self._push(dt, "tx", reply, peer)

    def _push(self, dt, kind, pkt, peer):
        heapq.heappush(self._events, (time.monotonic() + dt, self._n, kind, pkt, peer))
        self._n += 1

    def handle(self, pkt, peer):
        """One request from peer (address, port): the reply, or None (malformed)."""
        with self.lock:
            if len(pkt) < 8 or len(pkt) % 4:
                self.n_drop += 1
                return None
            seq, flags = struct.unpack_from("<II", pkt)
            self.received.append((time.monotonic(), seq))
            if flags & ETH_SYNC:
                if self.session != (peer, seq):         # a copy of the session's SYNC does not reopen it
                    self.session, self.expect = (peer, seq), (seq + 1) & M32
                return struct.pack("<4I", seq, ETH_SYNC, self.expect, 0)
            if self.session is None or self.session[0] != peer:
                self.n_drop += 1
                return struct.pack("<4I", seq, ETH_NOSYNC, 0, 0)
            ahead = ((self.expect - seq + 0x8000_0000) & M32) - 0x8000_0000
            if ahead > 0:
                self.n_dup += 1
                return struct.pack("<4I", seq, ETH_DUP, self.expect, 0)
            if ahead < 0:
                self.n_drop += 1
                return struct.pack("<4I", seq, ETH_DROP, self.expect, 0)
            status, data = self.execute(pkt)
            self.expect = (self.expect + 1) & M32
            self.n_exec += 1
        if self.exec_s:
            time.sleep(self.exec_s)
        return struct.pack("<4I", seq, 0, self.expect, status) + data

    def execute(self, pkt):
        """Run the TSI commands of a request: (status, read data)."""
        w = struct.unpack("<%dI" % (len(pkt) // 4), pkt)
        i, nrun, status, out = 2, 0, 0, bytearray()
        while i < len(w):
            if len(w) - i < 5:
                status |= ETH_CUT
                break
            cmd, alo, ahi, llo, lhi = w[i:i + 5]
            i += 5
            if cmd > 1:
                status |= ETH_BADCMD
                break
            addr, n = ahi << 32 | alo, (lhi << 32 | llo) + 1
            if cmd == 1:
                k = min(n, len(w) - i)
                status |= self.write(addr, pkt[4 * i:4 * (i + k)])
                i += k
                if k < n:
                    status |= ETH_CUT
                    break
            else:
                k = min(n, ETH_REPLY_WORDS - len(out) // 4)
                if k < n:
                    status |= ETH_OVERFLOW
                st, data = self.read(addr, k)
                status |= st
                out += data
            nrun += 1
        return status | nrun, bytes(out)

    def _mem(self, a):
        """(memory, offset, bytes to its end) for a, or None for a register or an unmapped address."""
        if IMEM_BASE <= a < IMEM_BASE + IMEM_BYTES:
            return self.imem, a - IMEM_BASE, IMEM_BASE + IMEM_BYTES - a
        if VMEM_BASE <= a < VMEM_BASE + VMEM_BYTES:
            return self.vmem, a - VMEM_BASE, VMEM_BASE + VMEM_BYTES - a
        if DRAM_BASE <= a <= M32:
            o = (a - DRAM_BASE) % len(self.dram)      # aliased, like the SoC's DRAM
            return self.dram, o, min(len(self.dram) - o, M32 + 1 - a)
        return None

    def write(self, addr, data):
        status, pos = 0, 0
        while pos < len(data):
            a = (addr + pos) & ~3
            if a >> 32:                 # not on the bus: dropped
                break
            m = self._mem(a)
            if m:
                mem, o, room = m
                k = min(room, len(data) - pos)
                mem[o:o + k] = data[pos:pos + k]
                pos += k
            else:
                status |= self._write_reg(a, struct.unpack_from("<I", data, pos)[0])
                pos += 4
        return status

    def read(self, addr, n):
        status, out = 0, bytearray()
        while len(out) < 4 * n:
            a = (addr + len(out)) & ~3
            m = None if a >> 32 else self._mem(a)
            if m:
                mem, o, room = m
                out += mem[o:o + min(room, 4 * n - len(out))]
            else:
                st, v = (0, 0) if a >> 32 else self._read_reg(a)
                status |= st
                out += struct.pack("<I", v)
        return status, bytes(out)

    def _write_reg(self, a, v):
        if a < 0x1000:
            if a in (SYS_SCRATCH, SYS_LEDS, SYS_ETH_IP):
                self.sys[a] = v
            elif a == SYS_RUN_CYCLES:
                self.sys[a] = 0
            return 0
        if CSR_BASE <= a < CSR_BASE + 0x1000:
            self.csr[a] = v
            if a == CSR_EXEC_CONTROL:
                self.exec_log.append(v)
                if v & 1 and self.kernel:
                    self.kernel(self)
            return 0
        return ETH_DENIED

    def _read_reg(self, a):
        if a < 0x1000:
            live = {SYS_ETH_RUN: self.n_exec, SYS_ETH_DUP: self.n_dup, SYS_ETH_DROP: self.n_drop,
                    SYS_ETH_STATUS: self.sys[SYS_ETH_STATUS] | (self.session is not None) << 17}
            return 0, live[a] & M32 if a in live else self.sys.get(a, 0)
        if CSR_BASE <= a < CSR_BASE + 0x1000:
            return 0, self.csr.get(a, 0)
        return ETH_DENIED, 0


def heavy(loss=0.3):
    return dict(rx=Net(loss=loss, dup=0.1, reorder=0.1), tx=Net(loss=loss, dup=0.1, reorder=0.1))


class Raw:
    """A bare UDP client for protocol-level checks of MockBoard."""

    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.connect(("127.0.0.1", port))
        self.sock.settimeout(1.0)

    def ask(self, *words, raw=None):
        self.sock.send(raw if raw is not None else struct.pack("<%dI" % len(words), *words))
        try:
            r = self.sock.recv(65536)
        except socket.timeout:
            return None
        return struct.unpack_from("<4I", r) + (list(struct.unpack_from("<%dI" % (len(r) // 4 - 4), r, 16)),)


class MockBoardTest(unittest.TestCase):
    """The model itself, by the spec's FPGA rules (a reference for the RTL)."""

    def setUp(self):
        self.b = MockBoard()
        self.addCleanup(self.b.stop)
        self.r = Raw(self.b.port)
        self.addCleanup(self.r.sock.close)

    def test_rules(self):
        r, b = self.r, self.b
        self.assertEqual(r.ask(5, 0, 0, 0x8000_0000, 0, 0, 0)[:2], (5, ETH_NOSYNC))
        self.assertEqual(r.ask(100, ETH_SYNC, 1, 2, 3), (100, ETH_SYNC, 101, 0, []))   # payload ignored
        self.assertEqual(r.ask(101, 0, 1, 0x8000_0000, 0, 1, 0, 0xA, 0xB), (101, 0, 102, 1, []))
        self.assertEqual(r.ask(101, 0, 1, 0x8000_0000, 0, 0, 0, 0xC), (101, ETH_DUP, 102, 0, []))
        self.assertEqual(r.ask(104, 0, 0, 0x8000_0000, 0, 0, 0), (104, ETH_DROP, 102, 0, []))
        self.assertEqual(r.ask(102, 0, 0, 0x8000_0000, 0, 1, 0), (102, 0, 103, 1, [0xA, 0xB]))
        self.assertEqual(r.ask(103, 0), (103, 0, 104, 0, []))                       # no commands
        self.assertEqual(r.ask(0xFFFF_FFFF, ETH_SYNC)[2], 0)                        # wraps
        self.assertEqual(r.ask(0, 0)[:3], (0, 0, 1))
        self.assertEqual(r.ask(0xFFFF_FFFF, 0)[:3], (0xFFFF_FFFF, ETH_DUP, 1))
        n = b.n_drop
        self.assertIsNone(r.ask(raw=b"\x01\x00\x00\x00\x00\x00"))                    # < 8 bytes
        self.assertIsNone(r.ask(raw=struct.pack("<II", 1, 0) + b"\x00"))            # not whole words
        self.assertEqual(b.n_drop, n + 2)
        self.assertEqual((b.n_exec, b.n_dup), (4, 2))

    def test_sessions(self):
        """The session belongs to the peer of its SYNC. A copy of that SYNC does not reopen it; any
        other SYNC opens a new one, and other peers get NOSYNC."""
        r, b = self.r, self.b
        r2 = Raw(b.port)
        self.addCleanup(r2.sock.close)
        start = (0, 1, CSR_EXEC_CONTROL, 0, 0, 0, 1)
        self.assertEqual(r.ask(100, ETH_SYNC), (100, ETH_SYNC, 101, 0, []))
        self.assertEqual(r.ask(101, *start), (101, 0, 102, 1, []))
        self.assertEqual(r.ask(100, ETH_SYNC), (100, ETH_SYNC, 102, 0, []))       # a copy: no reopen
        self.assertEqual(r.ask(101, *start)[:3], (101, ETH_DUP, 102))               # so it runs once
        n = b.n_drop
        self.assertEqual(r2.ask(102, *start), (102, ETH_NOSYNC, 0, 0, []))          # another peer
        self.assertEqual((b.n_drop, b.expect), (n + 1, 102))
        self.assertEqual(r2.ask(100, ETH_SYNC)[:3], (100, ETH_SYNC, 101))           # takes the session
        self.assertEqual(r.ask(102, *start)[:2], (102, ETH_NOSYNC))
        self.assertEqual(r.ask(100, ETH_SYNC)[:3], (100, ETH_SYNC, 101))            # takes it back
        self.assertEqual(r.ask(101, *start)[:3], (101, 0, 102))
        self.assertEqual(r.ask(500, ETH_SYNC)[:3], (500, ETH_SYNC, 501))            # another seq: reopens
        self.assertEqual(r.ask(100, ETH_SYNC)[:3], (100, ETH_SYNC, 101))
        self.assertEqual(b.exec_log, [1, 1])
        self.assertEqual(r.ask(101, 0, 0, SYS_ETH_STATUS, 0, 0, 0)[4][0] >> 17 & 1, 1)   # session open

    def test_status(self):
        r = self.r
        r.ask(0, ETH_SYNC)
        # CUT: a write ends early (the words that came are written), then a header ends early
        self.assertEqual(r.ask(1, 0, 1, 0x8000_0010, 0, 3, 0, 7, 8)[3], ETH_CUT)
        self.assertEqual(r.ask(2, 0, 0, 0x8000_0010, 0, 3, 0, 0, 0x8000)[3:], (1 | ETH_CUT, [7, 8, 0, 0]))
        # BADCMD skips the rest
        self.assertEqual(r.ask(3, 0, 2, 0, 0, 0, 0, 0, SYS_ID, 0, 0, 0)[3:], (ETH_BADCMD, []))
        # OVERFLOW: the reply holds REPLY_WORDS words
        s = r.ask(4, 0, 0, 0x8000_0000, 0, 300, 0, 0, 0x8000_0000, 0, 99, 0)
        self.assertEqual((s[3], len(s[4])), (2 | ETH_OVERFLOW, ETH_REPLY_WORDS))
        # DENIED: the hole above VMEM
        self.assertEqual(r.ask(5, 0, 0, VMEM_BASE + VMEM_BYTES, 0, 0, 0)[3:], (1 | ETH_DENIED, [0]))
        # >= 4 GiB: writes dropped, reads 0, not denied
        self.assertEqual(r.ask(6, 0, 1, 0x8000_0000, 1, 0, 0, 5)[3], 1)
        self.assertEqual(r.ask(7, 0, 0, 0x8000_0000, 1, 0, 0, 0, SYS_ID, 0, 0, 0)[3:], (2, [0, ATLS_ID]))


class EthLinkTest(unittest.TestCase):
    """link() gives timeout 5 ms and retries 500. Under heavy() loss (30 % each way) the timeout backs off
    often, and a short one keeps the tests fast. A link gives up after retries x timeout without
    progress: at the default retries that is about 7 tries, too few for this loss."""

    def board(self, **kw):
        b = MockBoard(**kw)
        self.addCleanup(b.stop)
        return b

    def link(self, b, **kw):
        kw.setdefault("timeout", 0.005)
        kw.setdefault("retries", 500)
        lk = EthLink("127.0.0.1", b.port, **kw)
        self.addCleanup(lk.close)
        return lk

    def test_basic(self):
        b = self.board()
        lk = self.link(b, timeout=1.0)
        self.assertEqual(lk.read_words(SYS_ID, 1), [ATLS_ID])
        rng = random.Random(1)
        for addr, n in ((IMEM_BASE, 100), (DRAM_BASE + 0x1000, 1000), (VMEM_BASE + 4, 363), (DRAM_BASE, 1)):
            w = [rng.getrandbits(32) for _ in range(n)]
            lk.write_words(addr, w)
            self.assertEqual(lk.read_words(addr, n), w)
        self.assertEqual(lk.read_words(DRAM_BASE + 0x1000, 0), [])
        lk.write_words(0x1_8000_0000, [1, 2])                       # >= 4 GiB: dropped
        self.assertEqual(lk.read_words(0x1_8000_0000, 2), [0, 0])
        lk.write_bulk(DRAM_BASE + 0x40, struct.pack("<3I", 1, 2, 3))
        self.assertEqual(b.dram[0x40:0x4C], struct.pack("<3I", 1, 2, 3))
        with self.assertRaises(ValueError):
            lk.write_bulk(DRAM_BASE, b"abc")
        with self.assertRaises(LinkError):
            lk.write_words(VMEM_BASE + VMEM_BYTES, [1])             # denied write
        self.assertEqual(lk.read_words(VMEM_BASE + VMEM_BYTES, 1), [0])   # denied read: 0
        self.assertEqual((lk.stats["resends"], lk.stats["timeouts"]), (0, 0))
        self.assertEqual(lk.stats["denied"], 2)

    def test_packing_limits(self):
        b = self.board()
        lk = self.link(b, timeout=1.0, nudge=None)    # counts every datagram sent
        sent = []
        put = lk._put
        lk._put = lambda m: (sent.append(bytes(m)), put(m))
        lk.write_bulk(DRAM_BASE, os.urandom(100_000))
        lk.read_words(DRAM_BASE, 30_000)
        lk.batch([("w", DRAM_BASE + 8 * i, [i]) for i in range(500)]
                 + [("r", DRAM_BASE + 8 * i, 1) for i in range(500)])
        seqs = set()
        for m in sent:
            self.assertLessEqual(len(m), ETH_PAYLOAD)
            w = struct.unpack("<%dI" % (len(m) // 4), m)
            seqs.add(w[0])
            i, nread, kinds = 2, 0, set()
            while i < len(w):
                cmd, n = w[i], w[i + 3] + 1
                kinds.add(cmd)
                self.assertEqual(w[i + 4], 0)
                if cmd == 1:
                    self.assertLessEqual(n, ETH_WRITE_WORDS)
                    i += 5 + n
                else:
                    nread += n
                    i += 5
            self.assertEqual(i, len(w))                             # whole commands only
            self.assertLessEqual(nread, ETH_REPLY_WORDS)
            self.assertEqual(len(kinds), 1)                         # reads or writes, not both
        self.assertEqual(len(seqs), len(sent))                      # no loss: nothing sent twice
        self.assertEqual(max(len(m) for m in sent), ETH_PAYLOAD)
        self.assertEqual(lk.batch([("r", DRAM_BASE + 8 * i, 1) for i in range(500)]), [[i] for i in range(500)])

    def test_window(self):
        b = self.board(tx=Net(latency=0.05))
        lk = self.link(b, window=4, timeout=1.0)
        t0 = time.monotonic()
        lk.write_bulk(DRAM_BASE, os.urandom(4 * ETH_WRITE_WORDS * 12))
        first = {s for t, s in b.received if t0 <= t < t0 + 0.04}   # a nudge copy is not a new datagram
        self.assertEqual(len(first), 4)
        self.assertEqual(lk.stats["datagrams"], 12)

    def test_heavy_loss(self):
        b = self.board(**heavy())
        lk = self.link(b)
        data = os.urandom(256 << 10)
        lk.write_bulk(DRAM_BASE + 0x10_0000, data)
        self.assertEqual(bytes(b.dram[0x10_0000:0x10_0000 + len(data)]), data)
        got = lk.read_words(DRAM_BASE + 0x10_0000, len(data) // 4)
        self.assertEqual(struct.pack("<%dI" % len(got), *got), data)
        prog = [random.getrandbits(32) for _ in range(2000)]
        lk.write_words(IMEM_BASE, prog)
        self.assertEqual(lk.read_words(IMEM_BASE, len(prog)), prog)
        for k in ("resends", "dups", "drops", "timeouts", "reread"):
            self.assertGreater(lk.stats[k], 0, k)

    def test_exactly_once(self):
        b = self.board(**heavy())
        lk = self.link(b)
        for i in range(200):
            lk.write_words(CSR_EXEC_CONTROL, [i << 1])
        ops = []
        for i in range(400):
            ops.append(("w", CSR_EXEC_CONTROL, [1000 + 2 * i]))
            if i % 7 == 0:
                ops.append(("w", DRAM_BASE + 0x2000, list(range(i, i + 50))))
        lk.batch(ops)
        self.assertEqual(b.exec_log, [i << 1 for i in range(200)] + [1000 + 2 * i for i in range(400)])
        self.assertGreater(lk.stats["resends"], 0)
        self.assertGreater(b.n_dup, 0)

    def test_read_write_order(self):
        """A read whose reply is lost is read again; it must not see the writes after it."""
        b = self.board(**heavy(0.35))
        lk = self.link(b)
        a = DRAM_BASE + 0x100
        ops = []
        for i in range(150):
            ops += [("w", a, [i]), ("r", a, 1), ("r", a + 4, 2), ("w", a + 4, [i, ~i & M32])]
        res = lk.batch(ops)
        for i in range(150):
            self.assertEqual(res[2 * i], [i])
            self.assertEqual(res[2 * i + 1], [0, 0] if i == 0 else [i - 1, ~(i - 1) & M32])
        self.assertGreater(lk.stats["reread"], 0)

    def test_seq_wrap(self):
        b = self.board(**heavy())
        lk = self.link(b, seq=0xFFFF_FFF0)
        data = os.urandom(64 * 4 * ETH_WRITE_WORDS)
        lk.write_bulk(DRAM_BASE, data)
        self.assertEqual(bytes(b.dram[:len(data)]), data)
        self.assertLess(b.expect, 0x1000)
        for i in range(50):
            lk.write_words(CSR_EXEC_CONTROL, [i])
        self.assertEqual(b.exec_log, list(range(50)))

    def test_slow_board(self):
        """The board takes longer than the timeout: resends are acknowledged DUP, nothing runs twice."""
        b = self.board(exec_s=0.004, **heavy(0.1))
        lk = self.link(b, timeout=0.005)
        data = os.urandom(48 * 4 * ETH_WRITE_WORDS)
        lk.batch([("w", CSR_EXEC_CONTROL, [7])])
        lk.write_bulk(DRAM_BASE, data)
        lk.batch([("w", CSR_EXEC_CONTROL, [8]), ("w", CSR_EXEC_CONTROL, [9])])
        self.assertEqual(bytes(b.dram[:len(data)]), data)
        self.assertEqual(b.exec_log, [7, 8, 9])
        self.assertEqual(lk.read_words(DRAM_BASE, 1000), list(struct.unpack("<1000I", data[:4000])))

    def test_held_frame(self):
        """Half the requests wait in the board's receive path for the next datagram (as on the VCU118):
        the nudge sends the newest datagram again after 1 ms, so no transaction waits for the timeout."""
        b = self.board(hold=0.5)
        lk = self.link(b, timeout=1.0)
        lk.batch([("w", CSR_EXEC_CONTROL, [5])])
        t0 = time.monotonic()
        for i in range(100):
            lk.write_words(SYS_SCRATCH, [i])
            self.assertEqual(lk.read_words(SYS_SCRATCH, 1), [i])
        self.assertLess(time.monotonic() - t0, 10.0)        # 200 transactions; one timeout would be 1 s
        self.assertEqual(lk.stats["timeouts"], 0)
        self.assertGreater(lk.stats["nudges"], 0)
        self.assertEqual(b.exec_log, [5])
        lk2 = self.link(b, timeout=0.2, nudge=None)          # without the nudge: timeouts
        for i in range(20):
            lk2.read_words(SYS_SCRATCH, 1)
        self.assertGreater(lk2.stats["timeouts"], 0)

    def test_defaults(self):
        b = self.board(**heavy(0.1))
        lk = EthLink("127.0.0.1", b.port)
        self.addCleanup(lk.close)
        self.assertEqual((lk.window, lk.timeout, lk.retries), (16, 0.05, 50))
        data = os.urandom(32 << 10)
        lk.write_bulk(DRAM_BASE, data)
        self.assertEqual(bytes(b.dram[:len(data)]), data)

    def test_atlas_run(self):
        b = self.board(**heavy(0.2))
        lk = self.link(b)
        atlas = Atlas(lk)
        self.assertEqual(atlas.dram_bytes, len(b.dram))
        words = [random.getrandbits(32) for _ in range(5000)]
        test = {"name": "t", "preload": [[0x9000_0000, words]], "program": [0x13] * 64,
                "check": [[0x9000_0000, words]]}
        r = atlas.run(test, 5.0)
        self.assertEqual((r.status, r.checked, r.dbg0, r.cycles), ("PASS", 5000, 1, 1234), r.detail)
        self.assertEqual(b.exec_log, [0, 1, 0])
        test["check"][0][1] = words[:-1] + [words[-1] ^ 1]
        self.assertEqual(atlas.run(test, 5.0).status, "FAIL")

    def test_poll(self):
        b = self.board(kernel=None)
        lk = self.link(b)
        v, waited = lk.poll(CSR_DBG1, 0.1)
        self.assertEqual(v, 0)
        self.assertGreaterEqual(waited, 0.1)
        threading.Timer(0.05, lambda: b.csr.__setitem__(CSR_DBG1, 5)).start()
        v, waited = lk.poll(CSR_DBG1, 5.0)
        self.assertEqual(v, 5)
        self.assertLess(waited, 1.0)
        b.csr[CSR_STATUS] = 1 | 1 << 1                              # halted, illegal instruction
        self.assertEqual(lk.poll(CSR_DBG0, 5.0)[0], 0)

    def test_no_answer(self):
        quiet = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        quiet.bind(("127.0.0.1", 0))
        self.addCleanup(quiet.close)
        closed = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        closed.bind(("127.0.0.1", 0))
        port = closed.getsockname()[1]
        closed.close()
        with mock.patch.object(atlas_host, "ETH_OPEN_S", 0.3):
            t0 = time.monotonic()
            with self.assertRaisesRegex(LinkError, r"no answer .* to SYNC in 0\.\d s \(\d+ tries\); check the cable"):
                EthLink("127.0.0.1", quiet.getsockname()[1], timeout=0.02, retries=5)
            self.assertLess(time.monotonic() - t0, 1.0)
            with self.assertRaisesRegex(LinkError, "port is closed"):
                EthLink("127.0.0.1", port, timeout=0.02, retries=5)

    def test_open_budget(self):
        """No answer: SYNC goes out for ETH_OPEN_S = 6 s (the copper link comes up about 3 s after
        programming or CPU_RESET), the wait backing off from timeout to 10 timeout."""
        quiet = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        quiet.bind(("127.0.0.1", 0))
        quiet.settimeout(0.1)
        self.addCleanup(quiet.close)
        arrived, stop = [], threading.Event()

        def listen():
            while not stop.is_set():
                try:
                    quiet.recv(64)
                    arrived.append(time.monotonic())
                except socket.timeout:
                    pass
        t = threading.Thread(target=listen, daemon=True)
        t.start()
        t0 = time.monotonic()
        with self.assertRaisesRegex(LinkError, r"to SYNC in \d+\.\d s .* about 3 s after programming or CPU_RESET"):
            EthLink("127.0.0.1", quiet.getsockname()[1])
        waited = time.monotonic() - t0
        stop.set()
        t.join()
        gaps = [y - x for x, y in zip(arrived, arrived[1:])]
        self.assertGreaterEqual(waited, 6.0)
        self.assertLess(waited, 7.5)
        self.assertLess(gaps[0], 0.2)
        self.assertGreater(min(gaps[-3:]), 0.4)
        self.assertLess(max(gaps), 0.8)

    def test_board_stops(self):
        b = MockBoard()
        lk = self.link(b, retries=5)
        lk.write_words(DRAM_BASE, [1])
        b.stop()
        t0 = time.monotonic()
        with self.assertRaisesRegex(LinkError, "no answer"):
            lk.read_words(DRAM_BASE, 1)
        self.assertLess(time.monotonic() - t0, 1.0)

    def test_board_reset(self):
        b = self.board()
        lk = self.link(b)
        lk.write_words(DRAM_BASE, [1])
        b.session = None
        with self.assertRaisesRegex(LinkError, "no session"):
            lk.write_words(CSR_EXEC_CONTROL, [1])
        self.assertEqual(b.exec_log, [])

    def test_stale_sync(self):
        """A late copy of the SYNC does not reopen the session: nothing runs twice, and the link goes on."""
        b = self.board()
        lk = self.link(b, seq=0x5000)
        for i in range(5):
            lk.write_words(CSR_EXEC_CONTROL, [i])
        self.assertEqual(b.handle(struct.pack("<II", 0x5000, ETH_SYNC), lk._sock.getsockname()),
                         struct.pack("<4I", 0x5000, ETH_SYNC, 0x5006, 0))
        lk.write_words(CSR_EXEC_CONTROL, [5])
        self.assertEqual(b.exec_log, list(range(6)))

    def test_late_sync_lost_reply(self):
        """A late copy of the SYNC arrives right after the first datagram ran, and that datagram's reply
        is lost: the datagram sent again is answered DUP, it does not run twice."""
        b = self.board()
        lk = self.link(b, seq=0x5000)
        handle, late = b.handle, []

        def late_sync(pkt, peer):
            r = handle(pkt, peer)
            if not late and struct.unpack_from("<I", r, 4)[0] == 0:
                late.append(handle(struct.pack("<II", 0x5000, ETH_SYNC), peer))
                return None
            return r
        b.handle = late_sync
        lk.write_words(CSR_EXEC_CONTROL, [1])
        self.assertEqual(b.exec_log, [1])
        self.assertEqual(late, [struct.pack("<4I", 0x5000, ETH_SYNC, 0x5002, 0)])
        self.assertGreater(lk.stats["dups"], 0)

    def test_takeover(self):
        """A second host opens its own session (two_clients.py): the first gets LinkError, never a write
        acknowledged that did not run, and the second goes on."""
        for seq_a, seq_b in ((1000, 5000), (5000, 1000)):
            with self.subTest(seq_a=seq_a, seq_b=seq_b):
                b = self.board()
                a = self.link(b, seq=seq_a, retries=5)
                a.write_words(CSR_EXEC_CONTROL, [1])
                other = self.link(b, seq=seq_b, retries=5)
                with self.assertRaisesRegex(LinkError, "no session open for this host.* another SYNC"):
                    a.write_words(CSR_EXEC_CONTROL, [2])
                with self.assertRaisesRegex(LinkError, "another SYNC"):
                    a.read_words(SYS_ID, 1)
                other.write_words(CSR_EXEC_CONTROL, [3])
                self.assertEqual(b.exec_log, [1, 3])

    def test_session_restarted(self):
        """A SYNC at another seq from this link's own address and port restarts the session. The board's
        expect is then newer (DUP) or older (DROP) than the link's seqs: LinkError, never a write
        acknowledged that did not run or reads sent again without end."""
        for offset, op in ((0x100, "w"), (0x3000_0000, "r"), (-0x100, "w")):
            with self.subTest(offset=offset, op=op):
                b = self.board()
                a = self.link(b, seq=0x1000_0000, retries=5)
                a.write_words(CSR_EXEC_CONTROL, [1])
                b.handle(struct.pack("<II", (0x1000_0000 + offset) & M32, ETH_SYNC), a._sock.getsockname())
                t0 = time.monotonic()
                with self.assertRaisesRegex(LinkError, "another SYNC restarted the session"):
                    if op == "w":
                        a.write_words(CSR_EXEC_CONTROL, [2])
                    else:
                        a.read_words(SYS_ID, 1)
                self.assertLess(time.monotonic() - t0, 1.0)
                self.assertEqual(b.exec_log, [1])

    def test_read_data_lost(self):
        """The reply of a read is lost every time: LinkError after retries reads again, not a loop."""
        b = self.board()
        lk = self.link(b, retries=5)
        handle = b.handle
        b.handle = lambda pkt, peer: (lambda r: None if r is not None and len(r) > 16 else r)(handle(pkt, peer))
        with self.assertRaisesRegex(LinkError, "read datagram at 0x0 was lost 6 times"):
            lk.read_words(SYS_ID, 1)
        self.assertEqual(lk.stats["reread"], 5)

    def test_dup_drop(self):
        """Every reply comes twice and the first datagram is lost: the duplicated DROP replies make one
        go-back, not two."""
        b = self.board(tx=Net(dup=1.0, delay=0.0))
        lk = self.link(b, timeout=1.0)
        handle, lost = b.handle, []

        def lose_first(pkt, peer):
            if not lost and len(pkt) > 100:
                lost.append(pkt)
                return None
            return handle(pkt, peer)
        b.handle = lose_first
        data = os.urandom(16 * 4 * ETH_WRITE_WORDS)
        lk.write_bulk(DRAM_BASE, data)
        self.assertEqual(bytes(b.dram[:len(data)]), data)
        self.assertEqual((lk.stats["resends"], lk.stats["timeouts"]), (16, 0), lk.stats)

    def test_slow_model(self):
        """Receiving a request, also one answered DUP, takes longer than the timeout (the Ethernet model
        under load): the timeout backs off, so the copies sent again do not pile up on the board."""
        b = self.board(rx_s=0.03)
        lk = self.link(b, window=4, timeout=0.02, retries=50)
        data = os.urandom(32 * 4 * ETH_WRITE_WORDS)
        lk.write_bulk(DRAM_BASE, data)
        lk.write_words(CSR_EXEC_CONTROL, [1])
        self.assertEqual(bytes(b.dram[:len(data)]), data)
        self.assertEqual(b.exec_log, [1])
        self.assertLess(lk.stats["resends"], lk.stats["datagrams"], lk.stats)

    def test_error_mid_batch(self):
        """A denied write raises LinkError; the datagrams in flight still run once, and the link goes on.
        (Under loss the denial can go unseen: a DUP reply has no status.)"""
        b = self.board()
        lk = self.link(b, timeout=1.0)
        ops = [("w", CSR_EXEC_CONTROL, [i]) if i % 2 else ("w", DRAM_BASE + 0x1000 * i, list(range(400)))
               for i in range(40)]
        ops.insert(20, ("w", VMEM_BASE + VMEM_BYTES, [1]))
        with self.assertRaisesRegex(LinkError, "denied"):
            lk.batch(ops)
        self.assertGreater(len(lk._fly), 0)
        b.rx, b.tx = heavy()["rx"], heavy()["tx"]
        lk.timeout, lk._rto, lk._rto_max = 0.005, 0.005, 0.08
        lk.write_words(CSR_EXEC_CONTROL, [100])
        self.assertEqual(lk.read_words(SYS_ID, 1), [ATLS_ID])
        log = b.exec_log
        self.assertEqual(log[-1], 100)
        self.assertEqual(log[:-1], sorted(set(log[:-1])))          # in order, none twice
        self.assertEqual(log[:10], list(range(1, 20, 2)))           # everything before the denied write ran

    def test_interrupted(self):
        """A call stopped with datagrams in flight: the next call finishes them first."""
        b = self.board(**heavy())
        lk = self.link(b)

        def ops():
            for i in range(10):
                yield from lk._pack([("w", CSR_EXEC_CONTROL, struct.pack("<I", i))])
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            lk._run(ops())
        lk.write_words(CSR_EXEC_CONTROL, [100])
        self.assertEqual(b.exec_log, list(range(10)) + [100])

    def test_cosim_launcher(self):
        """EthLink(cosim=...) with the PORT and READY lines in either order."""
        for order in ([], ["--ready-first"]):
            lk = EthLink(cosim=[sys.executable, Path(__file__).resolve(), "--serve"] + order)
            try:
                self.assertEqual(lk.read_words(SYS_ID, 1), [ATLS_ID])
                self.assertEqual(lk.ip, "127.0.0.1")
                self.assertNotEqual(lk.port, 0)
                self.assertEqual(lk.timeout, atlas_host.ETH_SIM_TIMEOUT)
            finally:
                lk.close()
            self.assertIsNotNone(lk.sim._p.poll())
        with self.assertRaisesRegex(LinkError, "exited"):
            EthLink(cosim=[sys.executable, "-c", "print('PORT 1'); import sys; sys.exit(3)"])
        with self.assertRaisesRegex(LinkError, "stopped at cycle 42"):
            EthLink(cosim=[sys.executable, "-c", "print('X 42', flush=True); import time; time.sleep(1)"])

    def test_throughput(self):
        b = self.board()
        lk = self.link(b)
        data = os.urandom(8 << 20)
        t0 = time.perf_counter()
        lk.write_bulk(DRAM_BASE, data)
        tw = time.perf_counter() - t0
        t0 = time.perf_counter()
        got = lk.read_words(DRAM_BASE, len(data) // 4)
        tr = time.perf_counter() - t0
        self.assertEqual(struct.pack("<%dI" % len(got), *got), data)
        print(f"\n  localhost model: write {len(data) / tw / 1e6:.1f} MB/s, read {len(data) / tr / 1e6:.1f} MB/s, "
              f"{lk.stats}", file=sys.stderr)


def serve(argv):
    ap = argparse.ArgumentParser(prog="test_ethlink.py --serve")
    ap.add_argument("port", nargs="?", type=int, default=int(os.environ.get("ETH_SIM_PORT", ETH_PORT)))
    ap.add_argument("--loss", type=float, default=float(os.environ.get("ETH_SIM_LOSS", 0)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("ETH_SIM_SEED", 1)))
    ap.add_argument("--dram-mb", type=int, default=16)
    ap.add_argument("--ready-first", action="store_true", help="print READY before PORT")
    a = ap.parse_args(argv)
    net = [Net(loss=a.loss, dup=a.loss / 3, reorder=a.loss / 3) for _ in range(2)]
    b = MockBoard(rx=net[0], tx=net[1], seed=a.seed, port=a.port, dram_bytes=a.dram_mb << 20)
    lines = [f"PORT {b.port}"] if a.port == 0 else []
    lines = ["READY"] + lines if a.ready_first else lines + ["READY"]
    print("\n".join(lines), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        sys.stdin.read()
    finally:
        print(f"Q exec {b.n_exec} dup {b.n_dup} drop {b.n_drop}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--serve"]:
        serve(sys.argv[2:])
    else:
        unittest.main()

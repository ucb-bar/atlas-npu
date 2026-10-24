"""Host driver for the standalone Atlas SoC (standalone/rtl/atlas_soc.sv).

The host takes the place of the SoC's CPU. For each kernel it writes the inputs into DRAM and
the program into IMEM, starts the scalar core through EXEC_CONTROL, waits for the program's
completion flag in DBG0, stops the core and reads the results back. This is the sequence of the
Chipyard C harness that baremetal/atlas_asm.py generates.

The links share one interface (write_words / read_words / poll):

    SimLink(binary)        the Verilator model, build/vl_soc/atlas_sim (standalone/sim)
    TsiLink(port, baud)    the VCU118 over UART TSI with PyUARTSI, on the CP2105 port
    TsiLink(cosim=binary)  PyUARTSI against the UART-level model, build/vl_tsi/atlas_tsi_sim
    EthLink(ip, port)      the VCU118 over TSI over UDP, on its 1000BASE-T port (docs/tsi-over-udp.md)
    EthLink(cosim=binary)  the Ethernet model, build/vl_eth/atlas_eth_sim, on a local UDP port
"""
import glob
import os
import queue
import random
import socket
import struct
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field

IMEM_BASE = 0x0002_0000
CSR_BASE = 0x0004_0000
VMEM_BASE = 0x2000_0000
DRAM_BASE = 0x8000_0000

CSR_CYCLE = CSR_BASE + 0x00
CSR_INSTCNT = CSR_BASE + 0x04
CSR_STATUS = CSR_BASE + 0x08
CSR_ILLEGAL_PC = CSR_BASE + 0x0C
CSR_DBG0 = CSR_BASE + 0x10
CSR_DBG1 = CSR_BASE + 0x14
CSR_EXEC_CONTROL = CSR_BASE + 0x18

# Performance counters (CSRFile.scala, PerfCounters): 64-bit, counting while the core runs.
CSR_PERF_RESET = CSR_BASE + 0x100     # write 1 to bit i: clear counter i
CSR_PERF_INHIBIT = CSR_BASE + 0x104   # bit i = 1: counter i stops
CSR_PERF_NUM = CSR_BASE + 0x108
CSR_PERF_BASE = CSR_BASE + 0x200      # counter i at + 8 i (low word), + 8 i + 4 (high word)
PERF_NAMES = ["cycles", "instret", "stall_unit", "stall_data", "stall_dma", "stall_delay",
              "mxu0_busy", "mxu0_feed", "mxu1_busy", "mxu1_feed", "vpu_busy", "xlu_busy",
              "vload_busy", "vstore_busy", "dma_busy", "dma_beats"]

SYS_ID = 0x00
SYS_SCRATCH = 0x04
SYS_FLAGS = 0x08
SYS_CLK_HZ = 0x14
SYS_BAUD = 0x18
SYS_DRAM_BYTES = 0x1C
SYS_ATLAS = 0x20
SYS_FEATURES = 0x24
SYS_LEDS = 0x28
SYS_RUN_CYCLES = 0x2C
SYS_ETH_IP = 0x30
SYS_ETH_STATUS = 0x34
SYS_ETH_RUN = 0x38
SYS_ETH_DUP = 0x3C
SYS_ETH_DROP = 0x40
SYS_ETH_PORT = 0x44
SYS_ETH_MAC_LO = 0x48
SYS_ETH_MAC_HI = 0x4C
SYS_ETH_RXERR = 0x50
SYS_ETH_RXFRAMES = 0x54
ATLS_ID = 0x4154_4C53

# The VCU118's CP2105, interface 1: the UART on AW25/BB21.
VCU118_TSI_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_*-if01-port0"
TSI_BAUD = 921_600

CHUNK_WORDS = 256

# TSI over UDP (docs/tsi-over-udp.md)
ETH_IP_DEFAULT = "172.20.1.18"
ETH_PORT = 21587                    # 0x5453, "TS"
ETH_PAYLOAD = 1472                  # UDP payload bytes in a 1500-byte IPv4 packet
ETH_REPLY_WORDS = 364               # read words in one reply
ETH_WRITE_WORDS = (ETH_PAYLOAD - 8 - 20) // 4   # 361: the largest write command
ETH_SIM_TIMEOUT = 3.0               # least timeout on the Ethernet model: simulated time is slow
ETH_OPEN_S = 6.0                    # least time to send SYNC: the copper link takes ~3 s to come up
ETH_SYNC, ETH_DUP, ETH_NOSYNC, ETH_DROP = 1, 2, 4, 8                        # reply flags
ETH_CUT, ETH_BADCMD, ETH_OVERFLOW, ETH_DENIED = 1 << 16, 1 << 17, 1 << 18, 1 << 19   # reply status
M32 = 0xFFFF_FFFF


class LinkError(RuntimeError):
    """The link failed, or the simulation stopped (for example on an RTL assertion)."""


class SimLink:
    """The Verilator model, driven over its stdin/stdout protocol (see standalone/sim/atlas_sim.cpp)."""

    def __init__(self, binary):
        self.messages = []
        self._p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.bulk = "B" in self._reply("READY")    # the model has the DRAM backdoor

    def _reply(self, prefix):
        """Return the next protocol line starting with prefix; keep any other output as messages."""
        while True:
            line = self._p.stdout.readline()
            if not line:
                self._p.wait()
                raise LinkError("simulation exited (code %s):\n%s"
                                % (self._p.returncode, "".join(self.messages[-20:])))
            if line.startswith(prefix):
                return line.split()[1:]
            if line.startswith("X "):
                self._p.wait()
                raise LinkError("simulation stopped at cycle %d:\n%s"
                                % (int(line.split()[1]), "".join(self.messages[-20:])))
            self.messages.append(line)

    def _send(self, text):
        try:
            self._p.stdin.write(text + "\n")
            self._p.stdin.flush()
        except BrokenPipeError as e:
            raise LinkError("simulation closed its input:\n%s" % "".join(self.messages[-20:])) from e

    def write_words(self, addr, words):
        for i in range(0, len(words), CHUNK_WORDS):
            chunk = words[i:i + CHUNK_WORDS]
            self._send("W %x %s" % (addr + 4 * i, " ".join("%x" % w for w in chunk)))
            line = self._p.stdout.readline()
            while line and not line[:2] in ("K\n", "E "):
                if line.startswith("X "):
                    self._p.wait()
                    raise LinkError("simulation stopped:\n%s" % "".join(self.messages[-20:]))
                self.messages.append(line)
                line = self._p.stdout.readline()
            if not line:
                raise LinkError("simulation exited:\n%s" % "".join(self.messages[-20:]))
            if line.startswith("E"):
                raise LinkError("write denied at 0x%s" % line.split()[1])

    def write_bulk(self, addr, data):
        """DRAM bytes through the model's backdoor (no simulated time)."""
        import tempfile
        with tempfile.NamedTemporaryFile(prefix="atlas_dram_", delete=False) as f:
            f.write(data)
        try:
            self._send("B %x %s" % (addr, f.name))
            self._reply("K")
        finally:
            os.unlink(f.name)

    def read_words(self, addr, n):
        out = []
        for i in range(0, n, CHUNK_WORDS):
            k = min(CHUNK_WORDS, n - i)
            self._send("R %x %x" % (addr + 4 * i, k))
            out += [int(w, 16) for w in self._reply("D")]
        return out

    def poll(self, addr, budget):
        """Read addr until it is nonzero or the core halts; budget in cycles.
        Returns (value, cycles waited)."""
        self._send("P %x %x" % (addr, int(budget)))
        value, cycles, _ = self._reply("P")
        return int(value, 16), int(cycles, 16)

    def close(self):
        if self._p.poll() is None:
            try:
                self._send("Q")
                self._p.wait(timeout=10)
            except (LinkError, subprocess.TimeoutExpired):
                self._p.kill()


class CosimTransport:
    """A PyUARTSI transport backed by the UART-level model (standalone/sim/tsi_cosim.cpp)."""

    def __init__(self, binary):
        self._p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def write_all(self, data):
        self._p.stdin.write(f"W {data.hex()}\n")

    def read_exact(self, size):
        from pyuartsi.exceptions import TransportTimeoutError
        self._p.stdin.write(f"R {size}\n")
        self._p.stdin.flush()
        reply = self._p.stdout.readline().split()
        if not reply or reply[0] != "D":
            raise TransportTimeoutError(f"co-simulation produced no reply for {size} bytes: {reply}")
        return bytes.fromhex(reply[1])

    def close(self):
        if self._p.poll() is None:
            self._p.stdin.write("Q\n")
            self._p.stdin.flush()
            self._p.wait()


class TsiLink:
    """UART TSI with PyUARTSI: the board's CP2105 port, or the UART-level model (cosim=binary)."""

    def __init__(self, port=None, baud=TSI_BAUD, cosim=None):
        from pyuartsi import UARTTSI
        self.read_retries = 0
        if cosim is not None:
            self.port = f"cosim:{cosim}"
            self._t = UARTTSI("cosim", baud, cflush_addr=0, transport=CosimTransport(cosim))
            return
        if port is None:
            ports = glob.glob(VCU118_TSI_PORT)
            if len(ports) != 1:
                raise SystemExit(f"found {len(ports)} VCU118 CP2105 if01 ports; pass --port")
            port = ports[0]
        self.port = port
        self._t = UARTTSI(port, baud, cflush_addr=0, timeout=10.0)

    def write_words(self, addr, words):
        for i in range(0, len(words), 1024):
            chunk = words[i:i + 1024]
            self._t.write_bytes(addr + 4 * i, struct.pack("<%dI" % len(chunk), *chunk))

    def read_words(self, addr, n):
        """Reads are idempotent, so a chunk whose reply lost a byte on the UART is read again."""
        from pyuartsi.exceptions import TransportTimeoutError
        out = []
        for i in range(0, n, 1024):
            k = min(1024, n - i)
            for attempt in range(3):
                try:
                    out += struct.unpack("<%dI" % k, self._t.read_bytes(addr + 4 * i, 4 * k))
                    break
                except TransportTimeoutError:
                    if attempt == 2:
                        raise
                    self.read_retries += 1
        return out

    def poll(self, addr, budget):
        """Read addr until it is nonzero or the core halts; budget in seconds.
        Returns (value, seconds waited)."""
        t0 = time.time()
        while True:
            value = self.read_words(addr, 1)[0]
            if value or time.time() - t0 > budget:
                return value, time.time() - t0
            status = self.read_words(CSR_STATUS, 1)[0]
            if status & 1 and status & 6:
                return value, time.time() - t0

    def close(self):
        self._t.close()


def _seq_diff(a, b):
    """a - b for 32-bit sequence numbers, signed."""
    return ((a - b + 0x8000_0000) & M32) - 0x8000_0000


class EthSim:
    """The Ethernet model (standalone/sim/eth_cosim.cpp) as a child process. It is asked for a free
    UDP port (0, in argv and ETH_SIM_PORT) and prints "PORT <n>" and "READY", in either order.
    binary may also be a command list."""

    def __init__(self, binary, wait=600.0):
        cmd = [str(c) for c in binary] if isinstance(binary, (list, tuple)) else [str(binary)]
        self.name = " ".join(cmd)
        self.messages = deque(maxlen=200)
        self.stopped = None                 # the "X <cycle>" line of an RTL assertion
        self._lines = queue.Queue()
        self._p = subprocess.Popen(cmd + ["0"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1,
                                   env=dict(os.environ, ETH_SIM_PORT="0"))
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        port, ready, end = None, False, time.monotonic() + wait
        try:
            while port is None or not ready:
                try:
                    line = self._lines.get(timeout=max(0.0, end - time.monotonic()))
                except queue.Empty:
                    if ready:               # no PORT line: the model took the default port
                        break
                    raise LinkError(f"{self.name}: no READY after {wait:.0f} s")
                if line is None:
                    try:
                        self._p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                    self.check()
                    raise LinkError(f"{self.name} closed its output before READY")
                f = line.split()
                if f[:1] == ["PORT"] and len(f) > 1:
                    port = int(f[1])
                elif f[:1] == ["READY"]:
                    ready = True
                    end = min(end, time.monotonic() + 2.0)
                elif f[:1] == ["X"]:
                    self.check()
        except BaseException:
            self.close()
            raise
        self._lines = None
        self.port = port or ETH_PORT

    def _read(self):
        for line in self._p.stdout:
            self.messages.append(line)
            if line.startswith("X ") and self.stopped is None:
                self.stopped = line
            q = self._lines
            if q is not None:
                q.put(line)
        self._p.stdout.close()
        q = self._lines
        if q is not None:
            q.put(None)

    def check(self):
        """Raise LinkError if the model stopped on an assertion or exited."""
        if self.stopped is None and self._p.poll() is None:
            return
        self._p.poll()
        self._reader.join(1.0)
        tail = "".join(list(self.messages)[-20:])
        if self.stopped:
            raise LinkError("simulation stopped at cycle %s:\n%s" % (self.stopped.split()[1], tail))
        raise LinkError("simulation exited (code %s):\n%s" % (self._p.returncode, tail))

    def close(self):
        """Close the model's input (it exits on EOF), then SIGTERM, then SIGKILL."""
        try:
            self._p.stdin.close()
        except OSError:
            pass
        try:
            self._p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._p.terminate()
            try:
                self._p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._p.kill()
                self._p.wait()
        self._reader.join(1.0)


class _Datagram:
    """TSI commands of one kind (reads or writes) for one request. reads: (buffer, byte offset, words)
    per read command. seq is given when it is first sent."""
    __slots__ = ("body", "ncmd", "reads", "nread", "addr", "rereads", "seq", "sent", "tries", "replies", "done")

    def __init__(self, body, ncmd, reads, addr, rereads=0):
        self.body, self.ncmd, self.reads, self.addr, self.rereads = body, ncmd, reads, addr, rereads
        self.nread = sum(r[2] for r in reads)
        self.seq, self.sent, self.tries, self.replies, self.done = None, 0.0, 0, 0, False


class EthLink:
    """TSI over UDP (docs/tsi-over-udp.md): the board's Ethernet port, or the Ethernet model (cosim=binary).

    Up to window datagrams are unacknowledged. A lost request or reply is sent again (go-back-N); the
    board runs each sequence number once, so a write never runs twice. A read whose reply was lost is
    sent again as a new datagram, and a datagram with writes waits until every earlier read has its
    data. Each call returns when the board has acknowledged all its datagrams: then its writes are done.
    The timeout follows the round trip and doubles at each expiry, up to 16 timeout (30 s on the model).
    When nothing arrives for nudge seconds after the last datagram, the newest datagram in flight is
    sent once more: on the VCU118 a lone frame can wait in the receive path until the next frame comes
    (docs/tsi-over-udp.md), and the board answers the copy DUP. Not on the model.
    The SYNC that opens the session is sent for at least ETH_OPEN_S. LinkError when nothing is
    acknowledged for retries x timeout seconds (and at least 3 expiries), when the board has no session
    for this link (it was reset, or another host sent SYNC), when the data of a read is lost more than
    retries times, and for a denied write whose reply arrives (a DUP reply has no status). On the model
    the timeout is at least ETH_SIM_TIMEOUT. stats counts datagrams, resends, DUP and DROP replies,
    timeouts, reads sent again, replies with a denied access and nudges."""

    def __init__(self, ip=ETH_IP_DEFAULT, port=ETH_PORT, window=16, timeout=0.05, retries=50, cosim=None,
                 seq=None, nudge=0.001):
        self._sock = self.sim = None
        if cosim is not None:
            self.sim = EthSim(cosim)
            ip, port, timeout = "127.0.0.1", self.sim.port, max(timeout, ETH_SIM_TIMEOUT)
        self.ip, self.port = ip, port
        self.name = f"{ip}:{port}" if self.sim is None else f"{self.sim.name} (UDP {port})"
        self.window, self.timeout, self.retries = max(1, int(window)), timeout, retries
        self.nudge = nudge if self.sim is None else None
        self.bulk = True
        self.stats = dict.fromkeys(("datagrams", "resends", "dups", "drops", "timeouts", "reread", "denied",
                                    "nudges"), 0)
        self._rto, self._srtt, self._rttvar = timeout, None, 0.0
        self._rto_max = 16 * timeout if self.sim is None else max(30.0, timeout)
        self._refused, self._error, self._behind = 0, None, None
        self._fly, self._nreads = {}, 0         # see _go
        self._heard = 0.0                       # time of the last datagram sent or received
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
                self._sock.setsockopt(socket.SOL_SOCKET, opt, 1 << 20)
            try:
                self._sock.connect((ip, port))
            except OSError as e:
                raise LinkError(f"{ip}:{port}: {e}") from e
            self._open(seq)
        except BaseException:
            self.close()
            raise

    def write_words(self, addr, words):
        if len(words):
            self._run(self._pack([("w", addr, struct.pack("<%dI" % len(words), *words))]))

    def write_bulk(self, addr, data):
        """Bytes, a whole number of little-endian words (the same pipelined path as write_words)."""
        data = memoryview(data).cast("B")
        if len(data) % 4:
            raise ValueError("write_bulk takes whole 32-bit words")
        if len(data):
            self._run(self._pack([("w", addr, data)]))

    def read_words(self, addr, n):
        return self.batch([("r", addr, n)])[0]

    def batch(self, ops):
        """Run ("w", addr, words) and ("r", addr, n) in order, pipelined. Returns each read's words."""
        cmds, outs = [], []
        for op in ops:
            if op[0] == "w":
                if len(op[2]):
                    cmds.append(("w", op[1], struct.pack("<%dI" % len(op[2]), *op[2])))
            else:
                buf = bytearray(4 * op[2])
                outs.append(buf)
                if op[2]:
                    cmds.append(("r", op[1], op[2], buf))
        self._run(self._pack(cmds))
        return [list(struct.unpack("<%dI" % (len(b) // 4), b)) for b in outs]

    def poll(self, addr, budget):
        """Read addr until it is nonzero or the core halts; budget in seconds.
        Returns (value, seconds waited)."""
        t0 = time.time()
        while True:
            (value,), (status,) = self.batch([("r", addr, 1), ("r", CSR_STATUS, 1)])
            if value or time.time() - t0 > budget or (status & 1 and status & 6):
                return value, time.time() - t0

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self.sim is not None:
            self.sim.close()

    @staticmethod
    def _pack(cmds):
        """Datagrams for ("w", addr, bytes) and ("r", addr, words, buffer), in order. A datagram holds
        reads or writes, not both, within the payload and reply limits; a long command is split."""
        parts, reads, ncmd, size, nread, kind, first = [], [], 0, 8, 0, None, 0
        for c in cmds:
            total = len(c[2]) // 4 if c[0] == "w" else c[2]
            done = 0
            while done < total:
                if c[0] == "w":
                    room = (ETH_PAYLOAD - size - 20) // 4
                else:
                    room = ETH_REPLY_WORDS - nread if size + 20 <= ETH_PAYLOAD else 0
                if kind != c[0] or room < min(total - done, 32):
                    if parts:
                        yield _Datagram(b"".join(parts), ncmd, reads, first)
                    parts, reads, ncmd, size, nread, kind = [], [], 0, 8, 0, c[0]
                    continue
                k = min(room, total - done)
                a = c[1] + 4 * done
                if not ncmd:
                    first = a
                parts.append(struct.pack("<5I", c[0] == "w", a & M32, a >> 32 & M32, k - 1, 0))
                if c[0] == "w":
                    parts.append(c[2][4 * done:4 * (done + k)])
                    size += 4 * k
                else:
                    reads.append((c[3], 4 * done, k))
                    nread += k
                size += 20
                ncmd += 1
                done += k
        if parts:
            yield _Datagram(b"".join(parts), ncmd, reads, first)

    def _open(self, seq):
        """SYNC at a random sequence number; the session's first datagram is seq + 1. The SYNC is sent
        again, waiting from timeout up to 10 timeout, for at least ETH_OPEN_S and retries x timeout."""
        seq = random.SystemRandom().getrandbits(32) if seq is None else seq & M32
        t0, wait, tries = time.monotonic(), self.timeout, 0
        while True:
            self._put(struct.pack("<II", seq, ETH_SYNC))
            tries += 1
            end = time.monotonic() + wait
            while (pkt := self._get(end - time.monotonic())) is not None:
                if len(pkt) >= 16:
                    s, flags, expect, _ = struct.unpack_from("<4I", pkt)
                    if s == seq and flags & ETH_SYNC:
                        if expect != (seq + 1) & M32:
                            raise LinkError(f"{self.name}: SYNC at {seq:#x} answered with expect {expect:#x}")
                        self._next = expect
                        return
            self._check()
            waited = time.monotonic() - t0
            if waited > max(ETH_OPEN_S, self.retries * self.timeout):
                raise LinkError(self._silent("SYNC", waited, tries))
            wait = min(2 * wait, 10 * self.timeout)

    def _run(self, datagrams):
        """Send the datagrams in order, at most window unacknowledged, until the board has run them all.
        If this stops early (an error), the datagrams in flight stay and are finished by the next call."""
        try:
            self._go(iter(datagrams))
        except BaseException:
            for d in self._fly.values():
                d.done = True           # no reads sent again for them
            raise

    def _go(self, src):
        todo = deque()          # to send next: reads to send again, then the next datagram of src
        fly = self._fly         # sent, not acknowledged: seq -> datagram, oldest first (consecutive seqs)
        idle = 0                # timeouts since the last progress
        recover = None          # expect of the last go-back on DROP: one per loss event, until a timeout
        stats = self.stats
        progress = time.monotonic()
        nudged = False          # the newest datagram was sent again since the last datagram sent or received
        while True:
            if not todo:
                d = next(src, None)
                if d is not None:
                    todo.append(d)
            # Writes wait for every earlier read: a read whose reply is lost is read again later,
            # and then must not see later writes.
            while todo and len(fly) < self.window and (todo[0].nread or not self._nreads):
                d = todo.popleft()
                d.seq, self._next = self._next, (self._next + 1) & M32
                fly[d.seq] = d
                self._nreads += d.nread > 0
                stats["datagrams"] += 1
                self._send(d)
                nudged = False
                if not todo:
                    d = next(src, None)
                    if d is not None:
                        todo.append(d)
            if not fly:
                return
            oldest = next(iter(fly.values()))
            deadline = max(oldest.sent, progress) + self._rto
            early = self.nudge is not None and not nudged and self._heard + self.nudge < deadline
            pkt = self._get((self._heard + self.nudge if early else deadline) - time.monotonic())
            if pkt is None and early:
                newest = fly[next(reversed(fly))]
                self._put(struct.pack("<II", newest.seq, 0) + newest.body)
                stats["nudges"] += 1
                nudged = True
                continue
            if pkt is None:
                self._check()
                idle += 1
                stats["timeouts"] += 1
                waited = time.monotonic() - progress
                if idle >= 3 and waited > self.retries * self.timeout:
                    raise LinkError(self._silent("seq %#x (datagram at 0x%x)" % (oldest.seq, oldest.addr),
                                                 waited, oldest.tries))
                # Back off (RFC 6298 5.5): resent datagrams give no RTT sample, so a timeout shorter than
                # the round trip (a slow model) would send the window again and again.
                self._rto = min(2 * self._rto, self._rto_max)
                recover = None
                self._resend(fly)
                nudged = False
                continue
            nudged = False
            while pkt is not None:
                now = self._heard = time.monotonic()
                if len(pkt) < 16 or len(pkt) % 4:
                    pkt = self._get(0)
                    continue
                seq, flags, expect, status = struct.unpack_from("<4I", pkt)
                d = fly.get(seq)
                if flags & (ETH_SYNC | ETH_NOSYNC):
                    if d is not None:
                        raise LinkError(f"{self.name} has no session open for this host: the board was reset or "
                                        "reprogrammed, or another SYNC restarted the session (second host?)")
                    pkt = self._get(0)
                    continue
                if _seq_diff(expect, self._next) > 0:
                    raise LinkError(f"{self.name} expects seq {expect:#x}, beyond every seq sent ({self._next:#x}): "
                                    "another SYNC restarted the session (second host?)")
                if d is not None:
                    d.replies += 1
                if flags & ETH_DUP:
                    stats["dups"] += 1
                elif flags & ETH_DROP:
                    stats["drops"] += 1
                err = None
                if not flags & (ETH_DUP | ETH_DROP) and d is not None and not d.done:   # ran: has its read data
                    if d.tries == 1:
                        self._sample(now - d.sent)
                    try:
                        self._finish(d, status, pkt)
                    except LinkError as e:
                        d.done, err = True, e
                # Every datagram before expect has run.
                again = []
                while fly:
                    s = next(iter(fly))
                    if _seq_diff(expect, s) <= 0:
                        break
                    x = fly.pop(s)
                    if x.nread:
                        self._nreads -= 1
                        if not x.done:
                            if x.rereads >= self.retries:
                                err = err or LinkError(f"{self.name}: the data of the read datagram at "
                                                       f"0x{x.addr:x} was lost {x.rereads + 1} times")
                            again.append(_Datagram(x.body, x.ncmd, x.reads, x.addr, x.rereads + 1))
                    progress, idle, self._behind = now, 0, None
                if err:
                    raise err
                if again:
                    stats["reread"] += len(again)
                    todo.extendleft(reversed(again))
                if flags & ETH_DROP and d is not None and fly:
                    first = next(iter(fly))
                    if first == expect:
                        # The board still waits for the oldest. If this answers d's last copy, which was
                        # sent after the oldest's last copy, that copy was lost: go back to it, once until
                        # the next timeout (a duplicated DROP must not send the window again).
                        if d.replies >= d.tries and expect != recover:
                            recover = expect
                            self._resend(fly)
                    elif _seq_diff(first, expect) > 0:
                        self._behind = expect
                pkt = self._get(0)

    def _finish(self, d, status, pkt):
        """Check the reply of a datagram that ran and store its read data."""
        data = memoryview(pkt)[16:]
        if status & (ETH_CUT | ETH_BADCMD | ETH_OVERFLOW) or status & 0xFFFF != d.ncmd or len(data) != 4 * d.nread:
            raise LinkError("datagram at 0x%x: the board ran %d of %d commands (status 0x%x) and sent %d of %d "
                            "read bytes" % (d.addr, status & 0xFFFF, d.ncmd, status, len(data), 4 * d.nread))
        if status & ETH_DENIED:
            self.stats["denied"] += 1
            if not d.nread:
                raise LinkError("write denied in the datagram at 0x%x (unmapped address)" % d.addr)
        pos = 0
        for buf, off, n in d.reads:
            buf[off:off + 4 * n] = data[pos:pos + 4 * n]
            pos += 4 * n
        d.done = True

    def _send(self, d):
        d.sent = time.monotonic()
        d.tries += 1
        self._put(struct.pack("<II", d.seq, 0) + d.body)

    def _resend(self, fly):
        for d in fly.values():
            self._send(d)
        self.stats["resends"] += len(fly)

    def _sample(self, rtt):
        """Round-trip time of a datagram sent once (RFC 6298 estimator); the timeout follows it."""
        if self._srtt is None:
            self._srtt, self._rttvar = rtt, rtt / 2
        else:
            self._rttvar += (abs(self._srtt - rtt) - self._rttvar) / 4
            self._srtt += (rtt - self._srtt) / 8
        self._rto = min(max(self.timeout, self._srtt + 4 * self._rttvar), self._rto_max)

    def _put(self, msg):
        self._heard = time.monotonic()
        try:
            self._sock.send(msg)
        except OSError as e:            # also the ICMP error of an earlier datagram
            self._note(e)

    def _get(self, timeout):
        """The next datagram from the board, or None after timeout seconds (0: only one already here)."""
        end = time.monotonic() + timeout
        while True:
            try:
                self._sock.settimeout(max(timeout, 0.0))
                return self._sock.recv(65536)
            except (socket.timeout, BlockingIOError):
                return None
            except OSError as e:
                self._note(e)
            timeout = end - time.monotonic()
            if timeout <= 0:
                return None

    def _note(self, e):
        if isinstance(e, ConnectionRefusedError):
            self._refused += 1
            self._check()
        else:
            self._error = e

    def _check(self):
        if self.sim is not None:
            self.sim.check()

    def _silent(self, what, waited, tries):
        msg = f"no answer from {self.name} (UDP) to {what} in {waited:.1f} s ({tries} tries)"
        if self._refused:
            msg += "; the port is closed (ICMP port unreachable)"
        if self._error is not None:
            msg += f"; {self._error}"
        if self._behind is not None:
            msg += (f"; the board waits for seq {self._behind:#x}, before datagrams it acknowledged: "
                    "another SYNC restarted the session (second host?)")
        elif what == "SYNC" and self.sim is None and not self._refused:
            msg += ("; check the cable, the host's address (on the board's subnet) and that the bitstream has "
                    "Ethernet (the copper link comes up about 3 s after programming or CPU_RESET)")
        return msg


def runs(pairs):
    """Group sorted (addr, word) pairs into [addr, [words]] runs of consecutive words."""
    out = []
    for addr, word in pairs:
        if out and out[-1][0] + 4 * len(out[-1][1]) == addr:
            out[-1][1].append(word)
        else:
            out.append([addr, [word]])
    return out


def read_perf(link):
    """The performance counters by name, or {} on a bitstream without them (PERF_NUM reads 0)."""
    if link.read_words(CSR_PERF_NUM, 1)[0] != len(PERF_NAMES):
        return {}
    w = link.read_words(CSR_PERF_BASE, 2 * len(PERF_NAMES))
    return {n: w[2 * i] | w[2 * i + 1] << 32 for i, n in enumerate(PERF_NAMES)}


def utilization(perf):
    """Each counter as % of the run's cycles (dma_beats: bytes per cycle)."""
    c = perf.get("cycles") or 0
    if not c:
        return {}
    u = {n: 100.0 * v / c for n, v in perf.items() if n not in ("cycles", "instret", "dma_beats")}
    u["ipc"] = perf["instret"] / c
    u["dma_bytes_per_cycle"] = 32.0 * perf["dma_beats"] / c
    return u


@dataclass
class Result:
    name: str
    status: str                 # PASS, FAIL, ILLEGAL, HALTED, TIMEOUT, ERROR
    variant: str = ""
    dbg0: int = 0
    dbg1: int = 0
    cycles: int = 0             # sys_regs RUN_CYCLES: cycles from START until the core halted
    instret: int = 0
    core_status: int = 0
    mismatches: int = 0
    checked: int = 0
    util: dict = field(default_factory=dict)
    perf: dict = field(default_factory=dict)   # performance counters (read_perf)
    run_s: float = 0.0                          # host time from the start to the completion poll
    detail: str = ""


class Atlas:
    """The standalone SoC behind a link."""

    def __init__(self, link):
        self.link = link
        ident = link.read_words(SYS_ID, 1)[0]
        if ident != ATLS_ID:
            raise LinkError("sys_regs ID is 0x%08x, not 0x%08x: not the standalone Atlas SoC" % (ident, ATLS_ID))
        self.dram_bytes = link.read_words(SYS_DRAM_BYTES, 1)[0]
        self.features = link.read_words(SYS_FEATURES, 1)[0]

    def dram(self, addr):
        """DRAM is aliased: fold a 64-bit DMA address onto the 32-bit host window."""
        return DRAM_BASE | (addr % self.dram_bytes)

    def run(self, test, budget):
        """Run one bundled test (see make_bundle.py). budget: poll budget (cycles on SimLink, seconds on
        TsiLink and EthLink)."""
        link, name = self.link, test["name"]
        for addr, words in test["preload"]:
            for a, w in runs([(self.dram(addr + 4 * i), x) for i, x in enumerate(words)]):
                if getattr(link, "bulk", False):
                    link.write_bulk(a, struct.pack("<%dI" % len(w), *w))
                else:
                    link.write_words(a, w)

        link.write_words(CSR_EXEC_CONTROL, [0])
        link.write_words(CSR_DBG0, [0])
        link.write_words(CSR_DBG1, [0])
        program = test["program"]
        link.write_words(IMEM_BASE, program)
        if link.read_words(IMEM_BASE, len(program)) != program:
            return Result(name, "ERROR", detail="IMEM readback mismatch")

        link.write_words(CSR_CYCLE, [0, 0])  # CYCLE and INSTCNT then count this run only
        link.write_words(SYS_RUN_CYCLES, [0])  # cycles until the core halts (ECALL)
        link.write_words(CSR_PERF_RESET, [0xFFFF_FFFF])  # every performance counter to 0
        t_start = time.time()
        link.write_words(CSR_EXEC_CONTROL, [1])
        dbg0, waited = link.poll(CSR_DBG0, budget)
        wall = time.time() - t_start
        link.write_words(CSR_EXEC_CONTROL, [0])
        # Read DBG0 again: the poll may have seen it just before the kernel set it and halted.
        _, instret, core_status, illegal_pc, dbg0, dbg1 = link.read_words(CSR_CYCLE, 6)
        cycles = link.read_words(SYS_RUN_CYCLES, 1)[0]
        r = Result(name, "PASS", dbg0=dbg0, dbg1=dbg1, cycles=cycles, instret=instret, core_status=core_status,
                   variant=test.get("variant", ""), perf=read_perf(link), run_s=wall)
        if dbg0 == 0:
            reason = (core_status >> 1) & 3
            if reason == 1:
                r.status, r.detail = "ILLEGAL", f"illegal instruction at word {illegal_pc}"
            elif reason:
                r.status, r.detail = "HALTED", f"halted (reason {reason}) without setting DBG0"
            else:
                r.status, r.detail = "TIMEOUT", f"DBG0 still 0 after {waited}"
            return r

        perf = test.get("perf", {})
        for unit in ("mxu0", "mxu1"):
            macs = perf.get(f"macs_{unit}")
            if macs is not None:
                r.util[unit] = (macs * 100) // (1024 * dbg1) if dbg1 else 0
        threshold = perf.get("threshold")
        if threshold is not None and (dbg1 == 0 or any(u < threshold for u in r.util.values())):
            r.status, r.detail = "FAIL", f"utilization {r.util} below {threshold}%"

        if test["check"]:
            bad = []
            for addr, words in test["check"]:
                for a, want in runs([(self.dram(addr + 4 * i), x) for i, x in enumerate(words)]):
                    got = link.read_words(a, len(want))
                    r.checked += len(want)
                    bad += [(a + 4 * i, w, g) for i, (w, g) in enumerate(zip(want, got)) if w != g]
            r.mismatches = len(bad)
            if bad:
                r.status = "FAIL"
                r.detail = "; ".join("0x%08x want %08x got %08x" % b for b in bad[:4])
        elif dbg0 != 1:
            r.status, r.detail = "FAIL", f"DBG0 = {dbg0}"
        return r

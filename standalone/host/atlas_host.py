"""Host driver for the standalone Atlas SoC (standalone/rtl/atlas_soc.sv).

The host takes the place of the SoC's CPU. For each kernel it writes the inputs into DRAM and
the program into IMEM, starts the scalar core through EXEC_CONTROL, waits for the program's
completion flag in DBG0, stops the core and reads the results back. This is the sequence of the
Chipyard C harness that baremetal/atlas_asm.py generates.

Two links share one interface (write_words / read_words / poll):

    SimLink(binary)        the Verilator model, build/vl_soc/atlas_sim (standalone/sim)
    TsiLink(port, baud)    the VCU118 over UART TSI with PyUARTSI, on the CP2105 port
    TsiLink(cosim=binary)  PyUARTSI against the UART-level model, build/vl_tsi/atlas_tsi_sim
"""
import glob
import struct
import subprocess
import time
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

SYS_ID = 0x00
SYS_FLAGS = 0x08
SYS_CLK_HZ = 0x14
SYS_BAUD = 0x18
SYS_DRAM_BYTES = 0x1C
SYS_ATLAS = 0x20
SYS_FEATURES = 0x24
SYS_RUN_CYCLES = 0x2C
ATLS_ID = 0x4154_4C53

# The VCU118's CP2105, interface 1: the UART on AW25/BB21.
VCU118_TSI_PORT = "/dev/serial/by-id/usb-Silicon_Labs_CP2105_Dual_USB_to_UART_Bridge_Controller_*-if01-port0"
TSI_BAUD = 921_600

CHUNK_WORDS = 256


class LinkError(RuntimeError):
    """The link failed, or the simulation stopped (for example on an RTL assertion)."""


class SimLink:
    """The Verilator model, driven over its stdin/stdout protocol (see standalone/sim/atlas_sim.cpp)."""

    def __init__(self, binary):
        self.messages = []
        self._p = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        self._reply("READY")

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

    def read_words(self, addr, n):
        out = []
        for i in range(0, n, CHUNK_WORDS):
            k = min(CHUNK_WORDS, n - i)
            self._send("R %x %x" % (addr + 4 * i, k))
            out += [int(w, 16) for w in self._reply("D")]
        return out

    def poll(self, addr, budget):
        """Read addr until it is nonzero; budget in cycles. Returns (value, cycles waited)."""
        self._send("P %x %x" % (addr, int(budget)))
        value, cycles = self._reply("P")
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
        out = []
        for i in range(0, n, 1024):
            k = min(1024, n - i)
            out += struct.unpack("<%dI" % k, self._t.read_bytes(addr + 4 * i, 4 * k))
        return out

    def poll(self, addr, budget):
        """Read addr until it is nonzero; budget in seconds. Returns (value, seconds waited)."""
        t0 = time.time()
        while True:
            value = self.read_words(addr, 1)[0]
            if value or time.time() - t0 > budget:
                return value, time.time() - t0

    def close(self):
        self._t.close()


def runs(pairs):
    """Group sorted (addr, word) pairs into [addr, [words]] runs of consecutive words."""
    out = []
    for addr, word in pairs:
        if out and out[-1][0] + 4 * len(out[-1][1]) == addr:
            out[-1][1].append(word)
        else:
            out.append([addr, [word]])
    return out


@dataclass
class Result:
    name: str
    status: str                 # PASS, FAIL, TIMEOUT, ERROR
    variant: str = ""
    dbg0: int = 0
    dbg1: int = 0
    cycles: int = 0             # sys_regs RUN_CYCLES: cycles from START until the core halted
    instret: int = 0
    core_status: int = 0
    mismatches: int = 0
    checked: int = 0
    util: dict = field(default_factory=dict)
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
        """Run one bundled test (see make_bundle.py). budget: poll budget (cycles on SimLink, seconds on TsiLink)."""
        link, name = self.link, test["name"]
        for addr, words in test["preload"]:
            for a, w in runs([(self.dram(addr + 4 * i), x) for i, x in enumerate(words)]):
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
        link.write_words(CSR_EXEC_CONTROL, [1])
        dbg0, waited = link.poll(CSR_DBG0, budget)
        link.write_words(CSR_EXEC_CONTROL, [0])
        _, instret, core_status, _, _, dbg1 = link.read_words(CSR_CYCLE, 6)
        cycles = link.read_words(SYS_RUN_CYCLES, 1)[0]
        r = Result(name, "PASS", dbg0=dbg0, dbg1=dbg1, cycles=cycles, instret=instret, core_status=core_status,
                   variant=test.get("variant", ""))
        if dbg0 == 0:
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

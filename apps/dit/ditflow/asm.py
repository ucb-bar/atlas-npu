"""An Atlas assembly builder: static memory maps, register pools and DMA scheduling.

Programs are emitted as npu_model assembly text (the syntax baremetal/assembly uses) and
assembled with npu_model's assembler. They target the interlocked core (pipeline-interlock and
later): the hardware holds each instruction until it can issue without a hazard, so the
builder never emits DELAY. What stays with software is DMA: a DMA.WAIT before anything reads
a buffer a DMA writes, and before a buffer a DMA store reads is overwritten.

Fixed scalar registers:
    x1, x2, x3    DMA operands (VMEM word address, DRAM address, size)
    x4            scalar load data (the timestep, a .vx operand)
    x5, x6        sampler step counter and step count
    x7            byte address of the current step's t in VMEM
    x9            DRAM address of the current step's trace slot
    x10           CSR data
    x19           kernel scratch (constants for ADD)
    x11-x18, x26-x29   loop counters and pointers (Builder.xreg)
    x20 - x25     word addresses of the six 256 KiB VMEM segments (vload/vstore bases)
    x30, x31      BF16 constants of .vx instructions (Builder.xconst)
"""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

from npu_model.util.converter import expand_li

from . import sched
from .layout import BF16_TILE, FP8_TILE, tiles

DRAM_BASE = 0x8000_0000
VMEM_SIZE = 1536 * 1024
SEGMENT = 256 * 1024
SEG_REGS = list(range(20, 26))
DMA_MAX = 4096
DMA_LOAD_CHANNELS = 7          # ch0-ch6 load weights and constants; ch7 stores results
STORE_CHANNEL = 7
CSR_DBG0, CSR_DBG1 = 0x800, 0x801
SCRATCH = 19
LOOP_REGS = [11, 12, 13, 14, 15, 16, 17, 18, 26, 27, 28, 29]
CONST_REGS = (30, 31)
RESIDENT_ERF = 28              # e0-e27 are set once in the header; e28-e31 are set at their use
TWO_SOURCE = {"vadd.bf16", "vsub.bf16", "vmul.bf16", "vmaximum.bf16", "vminimum.bf16"}
BRANCH_REACH = 1024            # instructions a branch can jump back over (4 KiB)


@dataclass(frozen=True)
class Ptr:
    """A VMEM byte address: x[reg] (a word address, as vload operands are) plus `off` bytes."""
    reg: int
    off: int = 0

    def __add__(self, k: int) -> "Ptr":
        return Ptr(self.reg, self.off + k)


@dataclass
class Tensor:
    """A tiled matrix in VMEM (see layout.py)."""
    name: str
    addr: int
    rows: int
    cols: int
    kind: str                  # "bf16" or "fp8"

    @property
    def rt(self) -> int:
        return tiles(self.rows)

    @property
    def ct(self) -> int:
        return tiles(self.cols)

    @property
    def tile_bytes(self) -> int:
        return BF16_TILE if self.kind == "bf16" else FP8_TILE

    @property
    def nbytes(self) -> int:
        return self.rt * self.ct * self.tile_bytes

    def tile(self, r: int, c: int) -> int:
        assert 0 <= r < self.rt and 0 <= c < self.ct, (self.name, r, c)
        return self.addr + (r * self.ct + c) * self.tile_bytes


class VmemAlloc:
    def __init__(self, size: int = VMEM_SIZE):
        self.size, self.top, self.map = size, 0, {}

    def alloc(self, name: str, nbytes: int, align: int = 1024) -> int:
        self.top = -(-self.top // align) * align
        addr = self.top
        self.top += nbytes
        assert self.top <= self.size, f"VMEM full allocating {name}: {self.top} > {self.size}"
        self.map[name] = (addr, nbytes)
        return addr

    def tensor(self, name: str, rows: int, cols: int, kind: str) -> Tensor:
        t = Tensor(name, 0, rows, cols, kind)
        t.addr = self.alloc(name, t.nbytes)
        return t

    def mark(self) -> int:
        return self.top

    def reset(self, mark: int) -> None:
        """Free everything allocated after mark (its users' code must be done by then)."""
        self.top = mark


class DramImage:
    """Initial DRAM contents and output regions, at absolute addresses from DRAM_BASE."""

    def __init__(self, size: int):
        self.size, self.top = size, 0
        self.blobs: list[tuple[str, int, bytes]] = []
        self.regions: dict[str, tuple[int, int]] = {}

    def _place(self, name: str, nbytes: int, align: int) -> int:
        self.top = -(-self.top // align) * align
        addr = DRAM_BASE + self.top
        self.top += nbytes
        assert self.top <= self.size, f"DRAM full placing {name}: {self.top} > {self.size}"
        self.regions[name] = (addr, nbytes)
        return addr

    def add(self, name: str, data: bytes, align: int = 1024) -> int:
        addr = self._place(name, len(data), align)
        self.blobs.append((name, addr, data))
        return addr

    def reserve(self, name: str, nbytes: int, align: int = 1024) -> int:
        return self._place(name, nbytes, align)


class Regs:
    """Matrix-register pool. Pairs start at an even register (the VPU's BF16 tile rule) and come
    from m0-m31, so two pairs never share a physical bank (a two-source VPU instruction cannot
    read two pairs of one bank). Single registers come from m32-m63 first. Each request takes the
    register freed longest ago, which leaves the scheduler (sched.py) independent registers."""

    def __init__(self, n: int = 64):
        self.freed = {r: 0 for r in range(n)}     # free register -> when it was released
        self.n, self.clock = n, 0

    def one(self) -> int:
        free = [r for r in self.freed if r >= self.n // 2] or list(self.freed)
        r = min(free, key=lambda r: (self.freed[r], r))
        del self.freed[r]
        return r

    def pair(self) -> int:
        free = [r for r in range(0, self.n, 2) if r in self.freed and r + 1 in self.freed]
        if not free:
            raise RuntimeError("out of register pairs")
        low = [r for r in free if r < self.n // 2] or free
        r = min(low, key=lambda r: (max(self.freed[r], self.freed[r + 1]), r))
        del self.freed[r], self.freed[r + 1]
        return r

    def release(self, *regs: int, pairs: tuple[int, ...] = ()) -> None:
        for r in regs + tuple(p + i for p in pairs for i in (0, 1)):
            assert r not in self.freed
            self.clock += 1
            self.freed[r] = self.clock


class Prog:
    def __init__(self):
        self.lines: list[str] = []
        self._labels = 0

    def op(self, text: str) -> None:
        self.lines.append("    " + text)

    def comment(self, text: str) -> None:
        self.lines.append("    # " + text)

    def new_label(self, stem: str) -> str:
        self._labels += 1
        return f"{stem}_{self._labels}"

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def instructions_since(self, label: str) -> int:
        """The number of instructions from label to the end of the program."""
        n = 0
        for line in self.lines[self.lines.index(f"{label}:") + 1:]:
            s = line.split("#", 1)[0].split()
            if s and not s[0].endswith(":"):
                n += len(expand_li(0, int(s[2], 0))) if s[0] == "LI" else 1
        return n

    def source(self) -> str:
        return "\n".join(self.lines) + "\n"


class Builder:
    """Prog plus the helpers every kernel uses. finish() reorders the program's basic blocks
    (sched.py) unless schedule is False."""

    def __init__(self, dram_size: int = 4 << 20, schedule: bool = True):
        self.schedule = schedule
        self.prog = Prog()
        self.regs = Regs()
        self.vmem = VmemAlloc()
        self.dram = DramImage(dram_size)
        self.dma = Dma(self)
        self._erf: dict[int, int] = {}
        self._next_dynamic_erf = RESIDENT_ERF
        self._header = None
        self.xval: dict[int, int | None] = {}      # known values of x1-x3 and the constant registers
        self.const_use = dict.fromkeys(CONST_REGS, 0)
        self.xfree = list(LOOP_REGS)
        self.rows = 32                              # the row count M (vsetm)
        self.label_rows: dict[str, int] = {}        # M at each label

    # -- program structure -------------------------------------------------------------------
    def begin(self) -> None:
        self.prog.comment("generated by apps/dit/ditflow (do not edit)")
        for s, r in enumerate(SEG_REGS):
            self.li(r, (s * SEGMENT) // 4)
        self.op("LI x5, 0")
        self.op("dma.config x5, ch0")               # dma.base = 0: DRAM addresses are 32-bit
        self._header = len(self.prog.lines)        # scale registers are set here (finish())

    def finish(self, dbg0: int = 1) -> str:
        self.dma.flush()
        self.li(10, dbg0)
        self.op(f"CSRRW x0, {CSR_DBG0:#x}, x10")
        self.op("ECALL")
        seli = [f"    seli e{idx}, {raw}" for raw, idx in sorted(self._erf.items(), key=lambda kv: kv[1])]
        self.prog.lines[self._header:self._header] = seli
        return sched.schedule(self.prog.source()) if self.schedule else self.prog.source()

    def op(self, text: str) -> None:
        self.prog.op(text)

    def label(self, name: str) -> None:
        """A label that a branch_back may jump to: M there is the M of the text before it."""
        self.prog.label(name)
        self.label_rows[name] = self.rows
        self._rows_after_label()
        for r in self.xval:                         # another path may enter here
            self.xval[r] = None

    def vsetm(self, rows: int) -> None:
        """M = rows for the tensor instructions after it."""
        if rows != self.rows:
            self.op(f"vsetm {rows}")
            self.rows = rows

    def xconst(self, bits: int) -> str:
        """An x-register with the BF16 bits in x[31:16], the scalar operand of a .vx instruction: a
        constant register that holds them already, else the one used longest ago, set by LI."""
        value = (bits & 0xFFFF) << 16
        held = [r for r in CONST_REGS if self.xval.get(r) == value]
        r = held[0] if held else min(CONST_REGS, key=self.const_use.get)
        if not held:
            self.li(r, value)
            self.xval[r] = value
        self.const_use[r] = max(self.const_use.values()) + 1
        return f"x{r}"

    def li(self, r: int, value: int) -> None:
        self.op(f"LI x{r}, {value:#x}")

    def set_x(self, r: int, value: int) -> None:
        """x[r] = value, by ADDI from its last value when that is shorter."""
        old = self.xval.get(r)
        if old is not None and -2048 <= value - old <= 2047:
            if value != old:
                self.op(f"ADDI x{r}, x{r}, {value - old}")
        else:
            self.li(r, value)
        self.xval[r] = value

    # -- scale registers ---------------------------------------------------------------------
    def erf(self, raw: int) -> int:
        """A scale register holding raw ERF value `raw` (2**(raw - 127)). The first 28 values get a
        register each, set once in the program header; later ones are set here, just before use."""
        assert 0 <= raw <= 255, raw
        if raw in self._erf:
            return self._erf[raw]
        if len(self._erf) < RESIDENT_ERF:
            self._erf[raw] = len(self._erf)
            return self._erf[raw]
        e = self._next_dynamic_erf
        self._next_dynamic_erf = RESIDENT_ERF + (e + 1 - RESIDENT_ERF) % (32 - RESIDENT_ERF)
        self.op(f"seli e{e}, {raw}")
        return e

    # -- scalar registers and loops ----------------------------------------------------------
    def xreg(self) -> int:
        return self.xfree.pop(0)

    def branch_back(self, rs1: int, rs2: int, top: str) -> None:
        """Jump to top while x[rs1] != x[rs2]: a BNE, or a BEQ over a JAL if top is out of reach.
        M is set back to its value at top first."""
        self.vsetm(self.label_rows[top])
        if self.prog.instructions_since(top) <= BRANCH_REACH:
            self.op(f"BNE x{rs1}, x{rs2}, {top}")
        else:
            out = self.prog.new_label("loop_end")
            self.op(f"BEQ x{rs1}, x{rs2}, {out}")
            self.op(f"JAL x0, {top}")
            self.prog.label(out)
            self._rows_after_label()
        for r in self.xval:                         # sched.py knows only the values a block sets
            self.xval[r] = None

    def _rows_after_label(self) -> None:
        """The text after a label runs at M = 32 unless a vsetm follows (sched.py): set M there if
        it is not 32."""
        if self.rows != 32:
            self.op(f"vsetm {self.rows}")

    def xrelease(self, *regs: int) -> None:
        for r in regs:
            assert r not in self.xfree and r in LOOP_REGS
            self.xfree.insert(0, r)

    def add_imm(self, rd: int, rs: int, value: int) -> None:
        """x[rd] = x[rs] + value (any 32-bit value)."""
        if -2048 <= value <= 2047:
            if value or rd != rs:
                self.op(f"ADDI x{rd}, x{rs}, {value}")
        else:
            self.li(SCRATCH, value)
            self.op(f"ADD x{rd}, x{rs}, x{SCRATCH}")

    @contextmanager
    def loop(self, n: int, steps=()):
        """Run the body n times (n >= 1). steps: (reg, delta) pairs added at the end of each pass."""
        assert n >= 1
        if n == 1:                                  # no branch; pointers still end one step on
            yield
            for reg, delta in steps:
                self.add_imm(reg, reg, delta)
            return
        cnt = self.xreg()
        self.li(cnt, n)
        top = self.prog.new_label("loop")
        self.label(top)
        yield
        for reg, delta in steps:
            self.add_imm(reg, reg, delta)
        self.op(f"ADDI x{cnt}, x{cnt}, -1")
        self.branch_back(cnt, 0, top)
        self.xrelease(cnt)

    # -- VMEM tensor access ------------------------------------------------------------------
    def _vaddr(self, addr) -> str:
        if isinstance(addr, Ptr):
            assert addr.off % 128 == 0 and -SEGMENT <= addr.off < SEGMENT, addr
            return f"{addr.off // 128}(x{addr.reg})"
        assert addr % 128 == 0 and 0 <= addr < VMEM_SIZE, addr
        s = addr // SEGMENT
        return f"{(addr - s * SEGMENT) // 128}(x{SEG_REGS[s]})"

    def vload(self, m: int, addr: int) -> None:
        self.op(f"vload m{m}, {self._vaddr(addr)}")

    def vload_w(self, m: int, addr) -> None:
        """A vload of all 32 rows at any M, for vmatpush.weight, which reads all of them."""
        rows = self.rows
        self.vsetm(32)
        self.vload(m, addr)
        self.vsetm(rows)

    def vstore(self, m: int, addr: int) -> None:
        self.op(f"vstore m{m}, {self._vaddr(addr)}")

    def vload2(self, m: int, addr: int) -> None:
        self.vload(m, addr)
        self.vload(m + 1, addr + FP8_TILE)

    def vstore2(self, m: int, addr: int) -> None:
        self.vstore(m, addr)
        self.vstore(m + 1, addr + FP8_TILE)

    def vli(self, m: int, bits: int) -> None:
        self.op(f"vli.all m{m}, {bits & 0xFFFF:#06x}")

    def v(self, mnemonic: str, *operands) -> None:
        if mnemonic in TWO_SOURCE:
            a, b = (int(s[1:]) for s in operands[1:])
            assert a == b or (a - b) % 32, f"{mnemonic}: m{a} and m{b} share a physical bank"
        self.op(f"{mnemonic} " + ", ".join(operands))


class Dma:
    """Splits transfers into <= 4 KiB commands and issues them round robin over ch0-ch6.

    Kernels queue the next weight block early and pump() a few commands between compute
    instructions, so a channel is idle again by the time it comes around (issuing to a busy
    channel would stall the core). wait(block) emits a DMA.WAIT on every channel the block used.
    """

    def __init__(self, b: Builder):
        self.b = b
        self.queue: deque = deque()
        self.next_ch = 0
        self.channels: dict[str, set[int]] = {}

    def load(self, block: str, vmem: int, dram: int, nbytes: int) -> None:
        self._split("dma.load", block, vmem, dram, nbytes)

    def _split(self, kind: str, block: str, vmem: int, dram: int, nbytes: int) -> None:
        assert vmem % 32 == 0 and dram % 32 == 0 and nbytes % 32 == 0, (block, vmem, dram, nbytes)
        self.channels.setdefault(block, set())
        for off in range(0, nbytes, DMA_MAX):
            self.queue.append((kind, block, vmem + off, dram + off, min(DMA_MAX, nbytes - off)))

    def pending(self, block: str) -> int:
        return sum(1 for q in self.queue if q[1] == block)

    def pump(self, n: int = 1) -> None:
        for _ in range(min(n, len(self.queue))):
            kind, block, vmem, dram, size = self.queue.popleft()
            ch = self.next_ch
            self.next_ch = (self.next_ch + 1) % DMA_LOAD_CHANNELS
            self._emit(kind, ch, vmem, dram, size)
            self.channels[block].add(ch)

    def _emit(self, kind: str, ch: int, vmem: int, dram: int, size: int) -> None:
        b = self.b
        b.set_x(1, vmem // 4)
        b.set_x(2, dram)
        b.set_x(3, size)
        if kind == "dma.load":
            b.op(f"dma.load x1, x2, x3, ch{ch}")
        else:
            b.op(f"dma.store x2, x1, x3, ch{ch}")

    def flush(self) -> None:
        self.pump(len(self.queue))

    def wait(self, block: str) -> None:
        while any(q[1] == block for q in self.queue):
            self.pump(1)
        for ch in sorted(self.channels.pop(block, set())):
            self.b.op(f"dma.wait ch{ch}")

    def store_now(self, vmem: int, dram: int, nbytes: int) -> None:
        """A store on its own channel, waited for at once (results are small)."""
        for off in range(0, nbytes, DMA_MAX):
            self._emit("dma.store", STORE_CHANNEL, vmem + off, dram + off, min(DMA_MAX, nbytes - off))
        self.b.op(f"dma.wait ch{STORE_CHANNEL}")

    def load_now(self, vmem: int, dram: int, nbytes: int, block: str = "now") -> None:
        self.load(block, vmem, dram, nbytes)
        self.wait(block)

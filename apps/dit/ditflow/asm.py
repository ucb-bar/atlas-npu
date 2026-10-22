"""An Atlas assembly builder: static memory maps, register pools and DMA scheduling.

Programs are emitted as npu_model assembly text (the syntax baremetal/assembly uses) and
assembled with npu_model's assembler. They target the interlocked core (pipeline-interlock and
later): the hardware holds each instruction until it can issue without a hazard, so the
builder never emits DELAY. What stays with software is DMA: a DMA.WAIT before anything reads
a buffer a DMA writes, and before a buffer a DMA store reads is overwritten.

Fixed scalar registers:
    x1, x2, x3    DMA operands (VMEM word address, DRAM address, size)
    x4            scalar load/store data
    x5, x6        sampler step counter and step count
    x7            byte address of the current step's t in VMEM
    x8            byte address of the timestep scratch tile
    x9            DRAM address of the current step's trace slot
    x10           CSR data
    x20 - x25     word addresses of the six 256 KiB VMEM segments (vload/vstore bases)
"""
from collections import deque
from dataclasses import dataclass

from .layout import BF16_TILE, FP8_TILE, tiles

DRAM_BASE = 0x8000_0000
VMEM_SIZE = 1536 * 1024
SEGMENT = 256 * 1024
SEG_REGS = list(range(20, 26))
DMA_MAX = 4096
DMA_LOAD_CHANNELS = 7          # ch0-ch6 load weights and constants; ch7 stores results
STORE_CHANNEL = 7
CSR_DBG0, CSR_DBG1 = 0xC10, 0xC11


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
    """Matrix-register pool. Pairs start at an even register (the VPU's BF16 tile rule)."""

    def __init__(self, n: int = 64):
        self.free_set = set(range(n))
        self.n = n

    def one(self) -> int:
        r = min(self.free_set)
        self.free_set.remove(r)
        return r

    def pair(self) -> int:
        for r in range(0, self.n - 1, 2):
            if r in self.free_set and r + 1 in self.free_set:
                self.free_set -= {r, r + 1}
                return r
        raise RuntimeError("out of register pairs")

    def release(self, *regs: int, pairs: tuple[int, ...] = ()) -> None:
        for r in regs:
            assert r not in self.free_set
            self.free_set.add(r)
        for r in pairs:
            self.release(r, r + 1)


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

    def source(self) -> str:
        return "\n".join(self.lines) + "\n"


class Builder:
    """Prog plus the helpers every kernel uses."""

    def __init__(self, dram_size: int = 4 << 20):
        self.prog = Prog()
        self.regs = Regs()
        self.vmem = VmemAlloc()
        self.dram = DramImage(dram_size)
        self.dma = Dma(self)
        self._erf: dict[int, int] = {}
        self._header = None
        self.xval: dict[int, int | None] = {}      # known values of x1-x3

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
        return self.prog.source()

    def op(self, text: str) -> None:
        self.prog.op(text)

    def label(self, name: str) -> None:
        self.prog.label(name)
        for r in self.xval:                         # another path may enter here
            self.xval[r] = None

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
        """The scale register holding raw ERF value `raw` (2**(raw - 127)); set by finish()."""
        assert 0 <= raw <= 255, raw
        if raw not in self._erf:
            self._erf[raw] = len(self._erf)
            assert len(self._erf) <= 32, "out of scale registers"
        return self._erf[raw]

    # -- VMEM tensor access ------------------------------------------------------------------
    def _vaddr(self, addr: int) -> str:
        assert addr % 128 == 0 and 0 <= addr < VMEM_SIZE, addr
        s = addr // SEGMENT
        return f"{(addr - s * SEGMENT) // 128}(x{SEG_REGS[s]})"

    def vload(self, m: int, addr: int) -> None:
        self.op(f"vload m{m}, {self._vaddr(addr)}")

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

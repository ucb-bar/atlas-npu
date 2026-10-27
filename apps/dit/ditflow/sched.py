"""List scheduling of Atlas assembly inside basic blocks.

The interlocked core issues in program order and holds each instruction until its unit and its
operands are free (docs/interlock.md). So the VPU, the LSU, mxu1 and the DMA overlap only as far
as the program order lets them. schedule() reorders each basic block: list scheduling over the
block's dependency graph, with a model of the cycle in which the core issues each instruction
(Machine).

The dependencies come from OPS, which gives the operands of every instruction ditflow emits:
- registers: m-registers (a BF16 pair is two), mxu1's accumulators and weight slots, scale and
  scalar registers, the DMA base. Read after write, write after read, write after write.
- VMEM: an access through a register of known value (a segment register, or one set earlier in
  the block) has an exact address. Two accesses through one register, unchanged between them,
  compare offsets. Anything else may alias anything.
- DMA: a transfer writes (load) or reads (store) its VMEM range when it issues, and again at the
  DMA.WAIT of its channel. A DMA.WAIT for transfers from an earlier block may touch any address.
  The instructions of one channel keep their order.
Labels, branches, CSR accesses, ECALL and instructions that OPS does not list stay in place and
end a block. vsetm does not end a block. Each instruction keeps the row count M of the last vsetm
before it in the text, or 32 if a label or the start comes first. A branch can enter a label with
any M, so the text must have M = 32 at a label unless a vsetm follows the label (asm.Builder does
this). The scheduled block sets M with a vsetm before each instruction that uses M (ROWS) and needs
another M than the one before it, and at its end if the text after it needs another M.
"""
from bisect import insort
from collections import defaultdict
from dataclasses import dataclass, field

from npu_model.hardware.mxu import mxu_latency
from npu_model.hardware.vpu import vpu_latency

GRANULE = 1024                  # VMEM dependency granule: the tile of one vload or vstore
BANK = 256 * 1024               # VMEM bank (the LSU's vector paths may not share one)
CONTROL = {"BEQ", "BNE", "BLT", "BGE", "BLTU", "BGEU", "JAL", "JALR"}
# Cycles per 32-byte DMA beat, and per transfer. Verilator's DRAM gives a beat per cycle and 8
# cycles of latency, but a transfer into a bank that vloads also read takes up to twice as long
# (the LSU has priority on a bank). The slow case leaves the DMA its slack.
DMA_BEAT, DMA_LATENCY = 2, 16
WINDOW = 48                     # ready instructions considered at each step, in program order
ROWS = {"vpu", "vpu2", "vload", "vstore", "mm", "pushacc", "pop"}    # units whose commands use M


@dataclass(frozen=True)
class Op:
    unit: str
    roles: str
    lat: int = 1                # cycles from issue to the last write, inclusive (else latency())
    cls: str = ""               # VPU lane box: two ops of one class never share the VPU


def _vpu(roles: str, cls: str, both: bool = False) -> Op:
    return Op("vpu2" if both else "vpu", roles, cls=cls)


# Operand roles: x scalar register, e scale register, m m-register, M BF16 pair (m, m+1), R row 0
# of a BF16 pair (read in the issue cycle only), Q FP32 quad (m .. m+3), a accumulator, w weight
# slot, each followed by r (read), w (write) or rw; v VMEM tile and s scalar VMEM access, both
# imm(x), followed by r or w; c DMA channel; i immediate.
# Latencies: npu-model/docs/rtl-timing.md and npu_model/hardware (vpu.py, mxu.py). A reader of a
# scalar load's register issues 4 cycles after the load.
OPS = {
    "LI": Op("alu", "xw i"), "ADDI": Op("alu", "xw xr i"), "ADD": Op("alu", "xw xr xr"),
    "seli": Op("alu", "ew i"), "dma.config": Op("alu", "xr c"),
    "lw": Op("alu", "xw sr", 4),
    "vload": Op("vload", "mw vr", 35), "vstore": Op("vstore", "mr vw", 35),
    "dma.load": Op("dma", "xr xr xr c"), "dma.store": Op("dma", "xr xr xr c"),
    "dma.wait": Op("wait", "c"),
    "vadd.bf16": _vpu("Mw Mr Mr", "addsub", both=True),
    "vsub.bf16": _vpu("Mw Mr Mr", "addsub", both=True),
    "vmul.bf16": _vpu("Mw Mr Mr", "mul", both=True),
    "vminimum.bf16": _vpu("Mw Mr Mr", "min", both=True),
    "vmaximum.bf16": _vpu("Mw Mr Mr", "max", both=True),
    "vadd.bf16.vx": _vpu("Mw Mr xr", "addsub"), "vsub.bf16.vx": _vpu("Mw Mr xr", "addsub"),
    "vmul.bf16.vx": _vpu("Mw Mr xr", "mul"), "vminimum.bf16.vx": _vpu("Mw Mr xr", "min"),
    "vmaximum.bf16.vx": _vpu("Mw Mr xr", "max"),
    "vadd.bf16.vr": _vpu("Mw Mr Rr", "addsub"), "vsub.bf16.vr": _vpu("Mw Mr Rr", "addsub"),
    "vmul.bf16.vr": _vpu("Mw Mr Rr", "mul"), "vminimum.bf16.vr": _vpu("Mw Mr Rr", "min"),
    "vmaximum.bf16.vr": _vpu("Mw Mr Rr", "max"),
    "vredsum.row.bf16": _vpu("Mw Mr", "addsub", both=True),
    "vredmax.row.bf16": _vpu("Mw Mr", "max", both=True),
    "vredsum.bf16": _vpu("Mw Mr", "csum"),
    "vsquare.bf16": _vpu("Mw Mr", "sqcb"), "vcube.bf16": _vpu("Mw Mr", "sqcb"),
    "vexp.bf16": _vpu("Mw Mr", "exp"), "vtanh.bf16": _vpu("Mw Mr", "tanh"),
    "vsin.bf16": _vpu("Mw Mr", "sincos"), "vcos.bf16": _vpu("Mw Mr", "sincos"),
    "vrecip.bf16": _vpu("Mw Mr", "rcp"), "vsqrt.bf16": _vpu("Mw Mr", "sqrt"),
    "vgelu.bf16": _vpu("Mw Mr", "gelu"),
    "vli.all": _vpu("Mw i", "vli"), "vli.col": _vpu("mw i", "vli"),
    "vtrpose.xlu": Op("xlu", "mw mr", 66),
    "vmatpush.weight.mxu1": Op("push", "ww mr", 33),
    "vmatpush.acc.bf16.mxu1": Op("pushacc", "aw Mr", 33),
    "vmatpush.acc.fp8.mxu1": Op("pushacc", "aw mr", 33),
    "vmatpush.acc.fp32.mxu1": Op("pushacc", "aw Qr", 65),
    "vmatmul.mxu1": Op("mm", "aw mr wr", 35),
    "vmatmul.acc.mxu1": Op("mm", "arw mr wr", 35),
    "vmatpop.fp8.acc.mxu1": Op("pop", "mw ar er", 33),
    "vmatpop.bf16.acc.mxu1": Op("pop", "Mw ar", 33),
    "vmatpop.bf16.scaled.acc.mxu1": Op("pop", "Mw ar er", 33),
    "vmatpop.fp32.acc.mxu1": Op("pop", "Qw ar", 65),
}


@dataclass
class Ins:
    text: str
    name: str
    args: list
    op: Op
    notes: list = field(default_factory=list)        # comment lines placed before it
    words: int = 1
    lat: int = 1
    reads: list = field(default_factory=list)        # registers and other ordered resources
    writes: list = field(default_factory=list)
    mem: list = field(default_factory=list)          # VMEM accesses: (address, write)
    rr: list = field(default_factory=list)           # m-register reservations: (reg, cycles)
    rw: list = field(default_factory=list)
    bank: int | tuple | None = None                  # VMEM bank of a vload or vstore, banks of a DMA
    acc: int = 0
    slot: int = 0
    beats: int = 0
    rows: int = 32                                   # the row count M


def latency(ins: Ins) -> int:
    """Cycles from issue to the last write, inclusive, at the instruction's row count."""
    u = ins.op.unit
    if u in ("vpu", "vpu2"):
        return vpu_latency(ins.name, ins.rows)
    if u in ("mm", "push", "pushacc", "pop"):
        return mxu_latency(ins.name, ins.rows)
    return ins.rows + 3 if u in ("vload", "vstore") else ins.op.lat


def vpu_reads(name: str, rows: int) -> int:
    """Cycles from issue until a VPU instruction has read its streamed source pair, as the RTL
    releases it: a row reduction reads both halves at once, a column reduction reads the pair
    twice, and a .vr reads row 0 of T(vs2) before T(vs1)."""
    if ".row." in name:
        return rows
    if name.startswith("vred"):
        return 4 * rows
    return 2 * rows + 1 if name.endswith(".vr") else 2 * rows


def _num(s: str) -> int:
    return int(s.removeprefix("acc").removeprefix("ch").lstrip("mxew"), 0)


def _mref(s: str) -> tuple[int, int]:
    off, reg = s.rstrip(")").split("(")
    return int(off, 0), _num(reg)


def li_words(value: int) -> int:
    """Words of `LI rd, value` (npu_model's expansion: ADDI, LUI or LUI + ADDI)."""
    v = value & 0xFFFF_FFFF
    v = v - (1 << 32) if v & 0x8000_0000 else v
    return 1 if -2048 <= v <= 2047 or v & 0xFFF == 0 else 2


def parse(line: str) -> tuple[str, list[str]]:
    name, _, rest = line.strip().partition(" ")
    return name, [a.strip() for a in rest.split(",")] if rest.strip() else []


class Block:
    """One basic block: each instruction's effects, then the dependency graph."""

    def __init__(self, code: list[Ins], known: dict[int, int]):
        self.code = code
        self.known = dict(known)                     # scalar registers of known value
        self.version = defaultdict(int)
        self.pending = {}                            # DMA channel -> its VMEM accesses until its wait
        for ins in code:
            self._effects(ins)

    # -- effects -----------------------------------------------------------------------------
    def _addr(self, reg: int, off: int, scale: int, size: int):
        """VMEM bytes [scale * x[reg] + off, + size)."""
        if reg in self.known:
            lo = scale * self.known[reg] + off
            return ("abs", lo, lo + size)
        return ("rel", (reg, self.version[reg], scale), off, off + size)

    def _effects(self, ins: Ins) -> None:
        op = ins.op
        ins.lat = latency(ins)
        if ins.name == "LI":
            ins.words = li_words(int(ins.args[1], 0))
        for role, s in zip(op.roles.split(), ins.args):
            kind, acc = role[0], role[1:]
            if kind in "xe":
                if s != "x0":
                    (ins.reads if acc == "r" else ins.writes).append(s)
            elif kind in "mMRQ":
                n = _num(s)
                regs = [f"m{n + i}" for i in range({"m": 1, "M": 2, "R": 2, "Q": 4}[kind])]
                (ins.reads if acc == "r" else ins.writes).extend(regs)
            elif kind in "aw":
                if kind == "a":
                    ins.acc = _num(s)
                else:
                    ins.slot = _num(s)
                if "r" in acc:
                    ins.reads.append(s)
                if "w" in acc:
                    ins.writes.append(s)
            elif kind in "vs":
                off, reg = _mref(s)
                ins.reads.append(f"x{reg}")
                addr = self._addr(reg, off * 128, 4, GRANULE) if kind == "v" else self._addr(reg, off, 1, 4)
                ins.mem.append((addr, acc == "w"))
                if kind == "v" and addr[0] == "abs":
                    ins.bank = addr[1] // BANK
            elif kind == "c":
                ins.reads.append(s)
                ins.writes.append(s)
        self._dma(ins)
        self._reserve(ins)
        self._propagate(ins)

    def _dma(self, ins: Ins) -> None:
        a = ins.args
        if ins.name == "dma.config":
            ins.writes.append("dmabase")
        elif ins.name in ("dma.load", "dma.store"):
            load = ins.name == "dma.load"
            vm, size = self.known.get(_num(a[0 if load else 1])), self.known.get(_num(a[2]))
            addr = ("abs", 4 * vm, 4 * vm + size) if vm is not None and size is not None else ("any",)
            ins.reads += ["dmabase", "dram"] if load else ["dmabase"]
            ins.writes += [] if load else ["dram"]
            ins.mem.append((addr, load))
            if addr[0] == "abs":
                ins.bank = (addr[1] // BANK, (addr[2] - 1) // BANK)
            ch = _num(a[3])
            if ch in self.pending:
                self.pending[ch].append((addr, load))
            ins.beats = size // 32 if size else 128
            ins.lat = ins.beats * DMA_BEAT + DMA_LATENCY
        elif ins.name == "dma.wait":
            ch = _num(a[0])
            done = self.pending.get(ch)
            ins.mem += [(("any",), True)] if done is None else done
            if done is None or any(load for _, load in done):       # DRAM that its loads read
                ins.reads.append("dram")
            if done is None or not all(load for _, load in done):   # DRAM that its stores write
                ins.writes.append("dram")
            self.pending[ch] = []

    def _reserve(self, ins: Ins) -> None:
        """m-register reservations: (register, cycles from issue until it is free). The VPU holds
        both registers of a pair until it has read (vpu_reads), or written, all 2M rows. An mxu1
        FP32 transfer streams m, m+1 in T+1 .. T+M and m+2, m+3 in T+M+1 .. T+2M; a weight push
        all 32 rows. The XLU reads its source until T+34."""
        op, m = ins.op, ins.rows
        for role, s in zip(op.roles.split(), ins.args):
            if role[0] not in "mMRQ":
                continue
            n, read = _num(s), role[1] == "r"
            if role[0] == "R":
                c = 1
            elif op.unit == "xlu":
                c = 34 if read else op.lat
            elif op.unit in ("vpu", "vpu2") and read:
                c = vpu_reads(ins.name, m)
            elif op.unit in ("vpu", "vpu2", "vload", "vstore"):
                c = ins.lat
            else:                                        # mxu1 streams rows T+1 .. T+M
                c = 32 if op.unit == "push" else m
            if role[0] == "Q":
                regs = [(n + i, m * (1 + i // 2)) for i in range(4)]
            else:
                regs = [(r, c) for r in ([n, n + 1] if role[0] in "MR" else [n])]
            (ins.rr if read else ins.rw).extend(regs)

    def _propagate(self, ins: Ins) -> None:
        a, k = ins.args, self.known
        for w in ins.writes:
            if w[0] != "x":
                continue
            n = _num(w)
            self.version[n] += 1
            if ins.name == "LI":
                k[n] = int(a[1], 0) & 0xFFFF_FFFF
            elif ins.name == "ADDI" and _num(a[1]) in k:
                k[n] = (k[_num(a[1])] + int(a[2], 0)) & 0xFFFF_FFFF
            elif ins.name == "ADD" and _num(a[1]) in k and _num(a[2]) in k:
                k[n] = (k[_num(a[1])] + k[_num(a[2])]) & 0xFFFF_FFFF
            else:
                k.pop(n, None)

    # -- dependencies ------------------------------------------------------------------------
    def graph(self) -> list[list[tuple[int, int]]]:
        """Successors of each instruction, with the cycles the successor waits for it."""
        code = self.code
        preds = [{} for _ in code]

        def edge(i, j, d):
            if i != j and preds[j].get(i, -1) < d:
                preds[j][i] = d

        writer, readers = {}, defaultdict(list)
        tiles = {}                                   # granule -> [last writer, readers since]
        exact, other = [], []                        # accesses: exact addresses, the rest
        for j, ins in enumerate(code):
            for r in ins.reads:
                if r in writer:
                    edge(writer[r], j, code[writer[r]].lat)
            for r in ins.writes:
                if r in writer:
                    edge(writer[r], j, 1)
                for i in readers[r]:
                    edge(i, j, 1)
            for r in ins.reads:
                readers[r].append(j)
            for r in ins.writes:
                writer[r], readers[r] = j, []
            for addr, w in ins.mem:
                if addr[0] == "abs":
                    for g in range(addr[1] // GRANULE, (addr[2] - 1) // GRANULE + 1):
                        t = tiles.setdefault(g, [None, []])
                        if t[0] is not None:
                            edge(t[0], j, code[t[0]].lat)
                        if w:
                            for i in t[1]:
                                edge(i, j, 1)
                            t[0], t[1] = j, []
                        else:
                            t[1].append(j)
                    for a2, w2, i in other:
                        if w or w2:
                            edge(i, j, code[i].lat if w2 else 1)
                    exact.append((addr, w, j))
                else:
                    for a2, w2, i in exact + other:
                        if (w or w2) and _alias(addr, a2):
                            edge(i, j, code[i].lat if w2 else 1)
                    other.append((addr, w, j))
        succ = [[] for _ in code]
        for j, p in enumerate(preds):
            for i, d in p.items():
                succ[i].append((j, d))
        return succ


def _alias(a, b) -> bool:
    if a[0] == "rel" and b[0] == "rel" and a[1] == b[1]:
        return a[2] < b[3] and b[2] < a[3]
    return True


class Machine:
    """The cycle in which the interlocked core issues each instruction.

    It follows the interlock's checks (docs/interlock.md): unit occupancy; m-register
    reservations on the 64 registers and on the 32 physical banks (m and m+32 share ports);
    the VPU's two slots (single-input ops of different lane boxes overlap, two-input ops and row
    reductions take both, .vx and .vr take one, and a .vr waits until the other slot has read its
    last row); mxu1's sequencer (one feed port, one weight-write path, one
    accumulator-push path, accumulator ports); the LSU's VMEM bank rule; DMA channels, with
    transfers at DMA_BEAT cycles per beat."""

    def __init__(self):
        self.f = defaultdict(int)                    # resource -> first cycle it is free
        self.slots = [[0, "", 0], [0, "", 0]]        # VPU: free cycle, lane box ("*": both), reads end
        self.lsu = {"vload": None, "vstore": None}   # VMEM bank of each LSU vector path
        self.dma = 0                                 # cycle the DMA engine has no beats queued

    def earliest(self, ins: Ins, t: int) -> int:
        f, u, b = self.f, ins.op.unit, ins.acc
        e = t
        for r, _ in ins.rr:
            e = max(e, f["w", r], f["pr", r & 31])
        for r, _ in ins.rw:
            e = max(e, f["r", r], f["pw", r & 31])
        for r in ins.reads:
            if r[0] == "x":
                e = max(e, f[r])
        if u in ("vload", "vstore"):
            e = max(e, f[u])
            other = "vstore" if u == "vload" else "vload"
            if self.lsu[other] == ins.bank:
                e = max(e, f[other])
        elif u in ("vpu", "vpu2"):
            (fa, _, _), (fb, cb, rb) = sorted(self.slots)
            if u == "vpu2":
                e = max(e, fb)
            else:
                e = max(e, fa)
                if e < fb and cb in ("*", ins.op.cls):
                    e = fb
                if ins.name.endswith(".vr"):             # it reads row 0 on both read ports
                    e = max(e, rb)
        elif u == "xlu":
            e = max(e, f["xlu"])
        elif u == "mm":
            e = max(e, f["P0"], f["row0", b], f["wpush", ins.slot], f["pushacc", b],
                    f["popfirst", b])
            if ins.name == "vmatmul.acc.mxu1":
                e = max(e, f["pop", b])
        elif u == "push":
            e = max(e, min(f["P0"], f["P1"]), f["wpath"], f["wread", ins.slot])
        elif u == "pushacc":
            e = max(e, f["P0"], f["P1"], f["row0", b], f["accpath"], f["mm", b], f["popfirst", b])
        elif u == "pop":
            e = max(e, f["W0"], f["row0", b], f["pushrow0", b], f["psum", b])
            if ins.op.roles[0] in "MQ":                  # BF16 and FP32 pops take both write ports
                e = max(e, f["W1"])
        elif u in ("dma", "wait"):
            e = max(e, f[ins.args[-1]])
            if u == "dma":                               # nor overtake a vload/vstore on its banks
                for path in ("vload", "vstore") if ins.name == "dma.load" else ("vstore",):
                    b = self.lsu[path]
                    if ins.bank is None or b is None or ins.bank[0] <= b <= ins.bank[1]:
                        e = max(e, f[path])
        return e

    def issue(self, ins: Ins, T: int) -> None:
        f, u, b = self.f, ins.op.unit, ins.acc
        for r, c in ins.rr:
            f["r", r] = max(f["r", r], T + c)
            f["pr", r & 31] = max(f["pr", r & 31], T + c)
        for r, c in ins.rw:
            f["w", r] = max(f["w", r], T + c)
            f["pw", r & 31] = max(f["pw", r & 31], T + c)
        if ins.name == "lw":
            f[ins.args[0]] = T + ins.lat
        if u in ("vload", "vstore"):
            f[u] = T + ins.lat
            self.lsu[u] = ins.bank
        elif u in ("vpu", "vpu2"):
            end = T + ins.lat - 1
            if u == "vpu2":
                self.slots = [[end, "*", end], [end, "*", end]]
            else:
                min(self.slots)[:] = [end, ins.op.cls, T + max((c for _, c in ins.rr), default=0)]
        elif u == "xlu":
            f["xlu"] = T + ins.lat
        elif u == "mm":
            m = ins.rows
            # row 0 lands at T + 3; the next matmul or pop of the accumulator issues from T + 4
            f["P0"], f["row0", b], f["mm", b], f["wread", ins.slot] = T + m, T + 4, T + ins.lat, T + m
            if ins.name == "vmatmul.acc.mxu1":
                f["psum", b] = T + m
        elif u == "push":
            f["P1" if f["P1"] <= T else "P0"] = T + 32
            f["wpath"], f["wpush", ins.slot] = T + 32, T + 32
        elif u == "pushacc":                             # rows land at T+1 .. T+M (FP32: twice)
            f["P0"] = f["P1"] = f["accpath"] = T + ins.lat - 1
            f["pushacc", b] = T + ins.lat                # a matmul waits for the whole push
            # after the last write of row 0
            f["pushrow0", b] = T + (ins.rows + 2 if "fp32" in ins.name else 2)
        elif u == "pop":
            f["W0"] = f["pop", b] = T + ins.lat - 1
            if ins.op.roles[0] in "MQ":
                f["W1"] = T + ins.lat - 1
            if ins.op.roles[0] == "Q":                   # rows are read again in T+M+1 .. T+2M
                f["popfirst", b] = T + ins.rows + 1
        elif u == "dma":
            start = max(T + 1, self.dma)
            self.dma = start + ins.beats * DMA_BEAT
            f[ins.args[-1]] = self.dma + DMA_LATENCY


def _order(block: Block, window: int, rows: int) -> list[Ins]:
    """Greedy list scheduling: of the first `window` ready instructions, issue the one the core
    can issue first, the one with the longest path to the end of the block on a tie. M is `rows`
    at the start; an instruction that needs another M issues one cycle later, after a vsetm."""
    code = block.code
    succ = block.graph()
    height = [0] * len(code)
    npred = [0] * len(code)
    for i in reversed(range(len(code))):
        height[i] = max([code[i].lat] + [d + height[j] for j, d in succ[i]])
        for j, _ in succ[i]:
            npred[j] += 1
    ready = [i for i in range(len(code)) if not npred[i]]
    m, t, out = Machine(), 0, []

    def start(ins: Ins) -> int:
        return t + (ins.op.unit in ROWS and ins.rows != rows)

    while ready:
        e, _, i = min((m.earliest(code[i], start(code[i])), -height[i], i) for i in ready[:window])
        if code[i].op.unit in ROWS:
            rows = code[i].rows
        m.issue(code[i], e)
        t = e + code[i].words
        out.append(code[i])
        ready.remove(i)
        for j, _ in succ[i]:
            npred[j] -= 1
            if not npred[j]:
                insort(ready, j)
    return out


def _constants(lines: list[str]) -> dict[int, int]:
    """Scalar registers set once, by an LI in the first block, and never written again."""
    writes, first, value = defaultdict(int), True, {}
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        name, args = parse(s)
        if s.endswith(":") or name in CONTROL or name not in OPS:
            first = False
            regs = [a for a in args if a.startswith("x")]
        else:
            regs = [a for r, a in zip(OPS[name].roles.split(), args) if r == "xw"]
        for a in regs:
            writes[_num(a)] += 1
            if first and name == "LI":
                value[_num(a)] = int(args[1], 0) & 0xFFFF_FFFF
    return {r: v for r, v in value.items() if writes[r] == 1}


def schedule(source: str, window: int = WINDOW) -> str:
    """Reorder the instructions of each basic block of `source` (npu_model assembly)."""
    lines = source.splitlines()
    consts = _constants(lines)
    out, code, notes = [], [], []
    first, rows, entry = True, 32, 32               # M of the text; M where the block starts

    def flush():
        nonlocal code, first
        m = entry
        for ins in _order(Block(code, {} if first else consts), window, entry) if code else ():
            out.extend(ins.notes)
            if ins.op.unit in ROWS and ins.rows != m:
                m = ins.rows
                out.append(f"    vsetm {m}")
            out.append(ins.text)
        if m is not None and m != rows:
            out.append(f"    vsetm {rows}")
        code, first = [], False

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            notes.append(line)
            continue
        name, args = parse(s)
        if name == "vsetm":
            rows = int(args[0], 0)
            if entry is None:                       # it sets M at the label
                entry = rows
                out.extend(notes + [line])
                notes = []
            continue
        if not s.endswith(":") and name in OPS:
            entry = rows if entry is None else entry
            code.append(Ins(line, name, args, OPS[name], notes, rows=rows))
            notes = []
            continue
        flush()
        out.extend(notes)
        notes = []
        out.append(line)
        if s.endswith(":"):
            rows, entry = 32, None
        else:
            entry = rows
    flush()
    out.extend(notes)
    return "\n".join(out) + "\n"

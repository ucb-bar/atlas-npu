"""Randomized hazard test of the interlock.

Each program is a random stream of dependent tensor instructions with no DELAY: vload, vstore,
VPU operations (with the .vx and .vr forms, and vgelu), MXU push / matmul / pop, and XLU
transposes, over 20 registers (m0-m11 and m32-m39, so m_i and m_i+32 also share a bank) and a few
VMEM tiles. vsetm sets the row count M now and then, to 1, 2, 3, values near the matmul spacing of
mxu1 (4) and of the NVFP4 MXU (8), 31 or 32, and the instructions that follow use rows 0..M-1.
Instructions are biased to read or overwrite what the previous ones touch, so RAW, WAR and WAW
hazards on MREG registers, weight slots, accumulators and VMEM tiles are frequent. A .vx often
reads x4 right after a scalar load writes it. The program ends by storing every register and the
VMEM tiles to DRAM.

The expected DRAM comes from the functional model (funcsim.py), which executes in program
order: on an interlocked core that is the architecture's meaning. A hazard the interlock lets
through gives a mismatch; an RTL assertion stops the Verilator run. --spaced puts DELAY 200
after every tensor instruction, so no two overlap: a spaced program that fails points at the
model's arithmetic, not at the interlock.

Values stay finite: the generator tracks each register's type (FP8, BF16, FP32) and a bound on
its magnitude, and only emits operations whose inputs have the right type and whose result
stays far below the BF16 range. FP8 data never uses the NaN codes; NVFP4 scales stay small.
An instruction of M < 32 rows keeps rows M.. of its destination, so it writes only a destination
of its result's type (the bound becomes the larger one); an accumulator counts the rows 0..k-1
that hold data, and a matmul.acc or pop of M rows needs k >= M.

    python -m ditflow.hazfuzz OUT --count 64 --length 400 --units mxu1          # default
    python -m ditflow.hazfuzz OUT --count 64 --length 400 --units mxu1,nv       # ipt-nvfp4
    python standalone/host/run_tests.py --sim build/vl-default/atlas_sim -j 32 --out RES OUT/*.json
"""
import argparse
import json
import random
from pathlib import Path

import torch
from npu_model.isa import ISA_VERSION

from . import layout as L
from .asm import Builder, FP8_TILE, SEGMENT
from .funcsim import FuncSim

REGS = list(range(12)) + list(range(32, 40))
PAIRS = [r for r in REGS if r % 2 == 0]
QUADS = [r for r in REGS if r % 4 == 0]
DATA_TILES = [0 * SEGMENT + i * FP8_TILE for i in range(12)] + [1 * SEGMENT + i * FP8_TILE for i in range(8)]
SCALE_TILES = [3 * SEGMENT + i * 128 for i in range(8)]        # 128 B: one register's NVFP4 scales
OUT_BASE = 2 * SEGMENT
SCALAR_WORD = 4 * SEGMENT                                     # byte address of the .vx scalar
MAX_BOUND = 60          # log2 of the largest magnitude a BF16 value may reach
FP8_BOUND = 9           # 448 < 2**9
NV_BOUND = 20           # 64 x 6 x 6 x 15 x 15 < 2**20 per NVFP4 matmul

VPU_BINARY = ("vadd.bf16", "vsub.bf16", "vmul.bf16", "vminimum.bf16", "vmaximum.bf16")
SCALARS = (0.5, -1.25, 3.0, 0.0078125)                        # .vx operands, |s| < 2^2
VPU_UNARY = ("vsquare.bf16", "vredsum.row.bf16", "vredmax.row.bf16", "vgelu.bf16")
ROW_COUNTS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 31, 32)


def fp8_codes(n: int, gen: torch.Generator) -> torch.Tensor:
    """Random finite E4M3 codes with |value| <= 15 (exponent field <= 10), both signs."""
    exp = torch.randint(0, 11, (n,), generator=gen)
    man = torch.randint(0, 8, (n,), generator=gen)
    sign = torch.randint(0, 2, (n,), generator=gen)
    return (sign << 7 | exp << 3 | man).to(torch.uint8)


def ue4m3_codes(n: int, gen: torch.Generator) -> torch.Tensor:
    """Random UE4M3 scales in [2^-6, 15] (exponent field 1-10), never NaN."""
    exp = torch.randint(1, 11, (n,), generator=gen)
    man = torch.randint(0, 8, (n,), generator=gen)
    return (exp << 3 | man).to(torch.uint8)


class Gen:
    def __init__(self, seed: int, length: int, units=("mxu1",), spaced: bool = False):
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.length = length
        self.fp8_units = [u for u in units if u == "mxu1"]
        self.nv = "nv" in units
        self.spaced = spaced
        self.b = Builder(schedule=False)
        self.b.begin()
        self.rtype: dict[int, str] = {}        # per register: "fp8", "bf16" or "fp32"
        self.rbound: dict[int, float] = {}     # and a log2 magnitude bound
        self.ttype: dict[int, str] = {}        # VMEM tiles
        self.tbound: dict[int, float] = {}
        # Per MXU: accumulator bounds (None: undefined), the rows 0..k-1 that hold data, and weight
        # slots written.
        self.acc = {u: {0: None, 1: None} for u in self.fp8_units + (["nv"] if self.nv else [])}
        self.arows = {u: {0: 0, 1: 0} for u in self.acc}
        self.wslot = {u: {0: False, 1: False} for u in self.acc}
        self.m = 32                            # the row count M
        self.recent: list[int] = []            # registers the last instructions touched
        self.stats = {"ops": 0}

    # -- bookkeeping --------------------------------------------------------------------------
    def touch(self, *regs):
        self.recent = (list(regs) + self.recent)[:6]

    def pick(self, cands, bias=0.6):
        cands = list(cands)
        if not cands:
            return None
        near = [c for c in cands if c in self.recent or (c ^ 1) in self.recent or (c + 32) % 64 in self.recent]
        if near and self.rng.random() < bias:
            return self.rng.choice(near)
        return self.rng.choice(cands)

    def typed(self, regs, t, n=1, max_bound=MAX_BOUND):
        return [r for r in regs if all(self.rtype.get(r + i) == t and self.rbound[r + i] <= max_bound for i in range(n))]

    def bf16_pairs(self, max_bound=MAX_BOUND):
        return self.typed(PAIRS, "bf16", 2, max_bound)

    def fp8_regs(self):
        return self.typed(REGS, "fp8")

    def set_regs(self, r, n, t, bound):
        for i in range(n):
            self.rtype[r + i], self.rbound[r + i] = t, bound

    def dests(self, regs, t, n=1):
        """The destinations of n registers for a result of type t: at M < 32 only those of type t."""
        return regs if self.m == 32 else [r for r in regs if all(self.rtype.get(r + i) == t for i in range(n))]

    def write_regs(self, r, n, t, bound):
        """Rows 0..M-1 of r .. r+n-1 get type t and values below 2^bound."""
        if self.m < 32:
            bound = max([bound] + [self.rbound[r + i] for i in range(n)])
        self.set_regs(r, n, t, bound)

    def acc_ready(self, u, a):
        """Rows 0..M-1 of the accumulator hold data."""
        return self.acc[u][a] is not None and self.arows[u][a] >= self.m

    def write_acc(self, u, a, bound):
        """Rows 0..M-1 of the accumulator get values below 2^bound."""
        if self.m < self.arows[u][a]:
            bound = max(bound, self.acc[u][a])
        self.acc[u][a], self.arows[u][a] = bound, max(self.arows[u][a], self.m)

    def count(self, key):
        self.stats[key] = self.stats.get(key, 0) + 1
        self.stats["ops"] += 1

    def spacer(self):
        if self.spaced:
            self.b.op("DELAY 200")

    def emit(self, mnemonic, *operands):
        self.b.v(mnemonic, *operands)
        self.count(mnemonic)
        self.spacer()

    # -- data ---------------------------------------------------------------------------------
    def preload(self):
        b = self.b
        for i, t in enumerate(DATA_TILES):
            if i % 2 == 0:
                data = fp8_codes(1024, self.gen).numpy().tobytes()
                self.ttype[t], self.tbound[t] = "fp8", FP8_BOUND
            else:
                v = torch.randn(32, 16, generator=self.gen) * 2
                data = v.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
                self.ttype[t], self.tbound[t] = "bf16", 4
            addr = b.dram.add(f"tile{i}", data)
            b.dma.load(f"in{i}", t, addr, FP8_TILE)
        if self.nv:
            addr = b.dram.add("scales", ue4m3_codes(128 * len(SCALE_TILES), self.gen).numpy().tobytes())
            b.dma.load("scales", SCALE_TILES[0], addr, 128 * len(SCALE_TILES))
            b.dma.wait("scales")
        for i in range(len(DATA_TILES)):
            b.dma.wait(f"in{i}")
        # Every register gets defined data (and NVFP4 scales) before the random part.
        for j, r in enumerate(REGS):
            t = DATA_TILES[j % len(DATA_TILES)]
            b.vload(r, t)
            self.set_regs(r, 1, self.ttype[t], self.tbound[t])
            if self.nv:
                b.op(f"vload.scale m{r}, {b._vaddr(SCALE_TILES[j % len(SCALE_TILES)])}")
        for e, raw in enumerate((121, 124, 127, 129)):
            b._erf[raw] = e
        b.li(19, SCALAR_WORD)

    # -- one random instruction -----------------------------------------------------------------
    def step(self):
        if self.rng.random() < 0.05:
            self.m = self.rng.choice(ROW_COUNTS)
            self.b.op(f"vsetm {self.m}")
            self.count("vsetm")
            return True
        r = self.rng.random()
        if r < 0.17:
            return self.op_vload()
        if r < 0.27:
            return self.op_vstore()
        if r < 0.52:
            return self.op_vpu()
        if r < 0.90:
            if self.nv and self.rng.random() < 0.45:
                return self.op_nv()
            return self.op_mxu(self.rng.choice(self.fp8_units))
        return self.op_xlu()

    def op_vload(self):
        t = self.rng.choice(DATA_TILES)
        dst = self.pick(self.dests(REGS, self.ttype[t]))
        if dst is None:
            return False
        self.b.vload(dst, t)
        self.count("vload")
        self.spacer()
        self.write_regs(dst, 1, self.ttype[t], self.tbound[t])
        self.touch(dst)
        return True

    def op_vstore(self):
        src = self.pick(REGS)
        tiles = [t for t in DATA_TILES if self.m == 32 or self.ttype[t] == self.rtype[src]]
        if not tiles:
            return False
        t = self.rng.choice(tiles)
        self.b.vstore(src, t)
        self.count("vstore")
        self.spacer()
        self.ttype[t] = self.rtype[src]
        self.tbound[t] = self.rbound[src] if self.m == 32 else max(self.tbound[t], self.rbound[src])
        self.touch(src)
        return True

    def op_vpu(self):
        rng = self.rng
        kind = rng.random()
        if kind < 0.1:
            dst = self.pick(self.dests(PAIRS, "bf16", 2))
            if dst is None:
                return False
            val = rng.choice((0.5, -1.25, 3.0, 0.0078125))
            self.b.vli(dst, L.bf16_bits(val))
            self.count("vli.all")
            self.spacer()
            self.write_regs(dst, 2, "bf16", 2)
            self.touch(dst, dst + 1)
            return True
        if kind < 0.65:
            op = rng.choice(VPU_BINARY)
            form = rng.choice(("", ".vx", ".vr"))
            srcs = self.bf16_pairs()
            a, c = self.pick(srcs), self.pick(srcs)
            # Two different source pairs in one physical bank are an illegal .bf16 instruction.
            if a is None or (form == "" and a % 32 == c % 32 and a != c):
                return False
            ba, bc = self.rbound[a], 2 if form == ".vx" else self.rbound[c]
            bound = ba + bc if op == "vmul.bf16" else max(ba, bc) + (1 if op in ("vadd.bf16", "vsub.bf16") else 0)
            if bound > MAX_BOUND:
                return False
            if form == ".vx":
                self.b.li(4, L.bf16_bits(rng.choice(SCALARS)) << 16 | rng.getrandbits(16))
                if rng.random() < 0.5:                           # through VMEM: the .vx waits for x4
                    self.b.op("SW x4, 0(x19)")
                    self.b.li(4, 0)
                    self.b.op("LW x4, 0(x19)")
            dst = self.pick(self.dests(PAIRS, "bf16", 2))
            if dst is None:
                return False
            self.emit(op + form, f"m{dst}", f"m{a}", "x4" if form == ".vx" else f"m{c}")
            self.write_regs(dst, 2, "bf16", bound)
            self.touch(dst, dst + 1, a, a + 1, *(() if form == ".vx" else (c, c + 1)))
            return True
        op = rng.choice(VPU_UNARY)
        a = self.pick(self.bf16_pairs())
        if a is None:
            return False
        ba = self.rbound[a]
        bound = {"vsquare.bf16": 2 * ba, "vredsum.row.bf16": ba + 5, "vredmax.row.bf16": ba,
                 "vgelu.bf16": ba}[op]
        if bound > MAX_BOUND:
            return False
        dst = self.pick(self.dests(PAIRS, "bf16", 2))
        if dst is None:
            return False
        self.emit(op, f"m{dst}", f"m{a}")
        self.write_regs(dst, 2, "bf16", max(bound, 1))
        self.touch(dst, dst + 1, a, a + 1)
        return True

    def op_mxu(self, u):
        rng = self.rng
        acc, ws = self.acc[u], self.wslot[u]
        kind = rng.random()
        if kind < 0.2:                                           # weight push
            src = self.pick(self.fp8_regs())
            if src is None:
                return False
            w = rng.randrange(2)
            self.emit(f"vmatpush.weight.{u}", f"w{w}", f"m{src}")
            ws[w] = True
            self.touch(src)
            return True
        if kind < 0.55:                                          # matmul
            src = self.pick(self.fp8_regs())
            w, a = rng.randrange(2), rng.randrange(2)
            if src is None or not ws[w]:
                return False
            new = 2 * FP8_BOUND + 5
            if rng.random() < 0.5 and self.acc_ready(u, a) and acc[a] < MAX_BOUND - 1:
                self.emit(f"vmatmul.acc.{u}", f"acc{a}", f"m{src}", f"w{w}")
                acc[a] = max(acc[a], new) + 1
            else:
                self.emit(f"vmatmul.{u}", f"acc{a}", f"m{src}", f"w{w}")
                self.write_acc(u, a, new)
            self.touch(src)
            return True
        if kind < 0.7:                                           # accumulator push: BF16 pair or FP32 quad
            a = rng.randrange(2)
            if rng.random() < 0.6:
                src = self.pick(self.bf16_pairs())
                if src is None:
                    return False
                self.emit(f"vmatpush.acc.bf16.{u}", f"acc{a}", f"m{src}")
                self.write_acc(u, a, self.rbound[src])
                self.touch(src, src + 1)
            else:
                src = self.pick(self.typed(QUADS, "fp32", 4))
                if src is None:
                    return False
                self.emit(f"vmatpush.acc.fp32.{u}", f"acc{a}", f"m{src}")
                self.write_acc(u, a, self.rbound[src])
                self.touch(*range(src, src + 4))
            return True
        a = rng.randrange(2)                                     # pops
        if not self.acc_ready(u, a):
            return False
        r = rng.random()
        if r < 0.5:
            return self.pop_bf16(u, a, scaled=r >= 0.3)
        if r < 0.75:
            dst = self.pick(self.dests(REGS, "fp8"))
            if dst is None:
                return False
            self.emit(f"vmatpop.fp8.acc.{u}", f"m{dst}", f"acc{a}", f"e{rng.randrange(4)}")
            self.write_regs(dst, 1, "fp8", FP8_BOUND)
            self.touch(dst)
            return True
        return self.pop_fp32(u, a)

    def pop_bf16(self, u, a, scaled):
        """vmatpop.bf16.acc, or vmatpop.bf16.scaled.acc with e0-e3 (2^-6 .. 2^2)."""
        dst = self.pick(self.dests(PAIRS, "bf16", 2))
        if dst is None:
            return False
        bound = max(self.acc[u][a] + 2 * scaled, 1)
        if scaled:
            self.emit(f"vmatpop.bf16.scaled.acc.{u}", f"m{dst}", f"acc{a}", f"e{self.rng.randrange(4)}")
        else:
            self.emit(f"vmatpop.bf16.acc.{u}", f"m{dst}", f"acc{a}")
        self.write_regs(dst, 2, "bf16", bound)
        self.touch(dst, dst + 1)
        return True

    def pop_fp32(self, u, a):
        dst = self.pick(self.dests(QUADS, "fp32", 4))
        if dst is None:
            return False
        self.emit(f"vmatpop.fp32.acc.{u}", f"m{dst}", f"acc{a}")
        self.write_regs(dst, 4, "fp32", max(self.acc[u][a], 1))
        self.touch(*range(dst, dst + 4))
        return True

    def op_nv(self):
        """NVFP4 MXU. Any register is valid E2M1 data, and every register's scales stay valid."""
        rng, b = self.rng, self.b
        acc, ws = self.acc["nv"], self.wslot["nv"]
        kind = rng.random()
        if kind < 0.12:                                          # scale transfers
            r, t = self.pick(REGS), rng.choice(SCALE_TILES)
            op = "vload.scale" if rng.random() < 0.6 else "vstore.scale"
            b.op(f"{op} m{r}, {b._vaddr(t)}")
            self.count(op)
            self.spacer()
            self.touch(r)
            return True
        if kind < 0.3:                                           # weight push
            src = self.pick(REGS)
            w = rng.randrange(2)
            self.emit("vmatpush.weight.nv", f"w{w}", f"m{src}")
            ws[w] = True
            self.touch(src)
            return True
        if kind < 0.6:                                           # matmul
            src = self.pick(REGS)
            w, a = rng.randrange(2), rng.randrange(2)
            if not ws[w]:
                return False
            if rng.random() < 0.5 and self.acc_ready("nv", a) and acc[a] < MAX_BOUND - 1:
                self.emit("vmatmul.acc.nv", f"acc{a}", f"m{src}", f"w{w}")
                acc[a] = max(acc[a], NV_BOUND) + 1
            else:
                self.emit("vmatmul.nv", f"acc{a}", f"m{src}", f"w{w}")
                self.write_acc("nv", a, NV_BOUND)
            self.touch(src)
            return True
        if kind < 0.75:                                          # accumulator push: BF16 pair or FP32 quad
            a = rng.randrange(2)
            if rng.random() < 0.5:
                src = self.pick(self.bf16_pairs())
                if src is None:
                    return False
                self.emit("vmatpush.acc.bf16.nv", f"acc{a}", f"m{src}")
                self.write_acc("nv", a, self.rbound[src])
                self.touch(src, src + 1)
            else:
                src = self.pick(self.typed(QUADS, "fp32", 4))
                if src is None:
                    return False
                self.emit("vmatpush.acc.fp32.nv", f"acc{a}", f"m{src}")
                self.write_acc("nv", a, self.rbound[src])
                self.touch(*range(src, src + 4))
            return True
        a = rng.randrange(2)                                     # pops
        if not self.acc_ready("nv", a):
            return False
        r = rng.random()
        if r < 0.5:
            return self.pop_bf16("nv", a, scaled=r >= 0.25)
        return self.pop_fp32("nv", a)

    def op_xlu(self):
        src = self.pick(self.fp8_regs())
        if src is None:
            return False
        dst = self.pick(REGS)
        self.emit("vtrpose.xlu", f"m{dst}", f"m{src}")
        self.set_regs(dst, 1, "fp8", FP8_BOUND)
        self.touch(dst, src)
        return True

    # -- the whole program --------------------------------------------------------------------
    def build(self):
        self.preload()
        n = 0
        while n < self.length:
            if self.step():
                n += 1
        b = self.b
        # Accumulators: pop the rows of each that hold data into a pair, then store every
        # register (and scales) at 32 rows.
        pair = 0
        for u, accs in self.acc.items():
            for a, bound in accs.items():
                if bound is not None:
                    b.op(f"vsetm {self.arows[u][a]}")
                    self.emit(f"vmatpop.bf16.acc.{u}", f"m{PAIRS[pair]}", f"acc{a}")
                    pair += 1
        b.op("vsetm 32")
        for j, r in enumerate(REGS):
            b.vstore(r, OUT_BASE + j * FP8_TILE)
        n_out = len(REGS) * FP8_TILE
        if self.nv:
            for j, r in enumerate(REGS):
                b.op(f"vstore.scale m{r}, {b._vaddr(OUT_BASE + n_out + j * 128)}")
            n_out += len(REGS) * 128
        out_regs = b.dram.reserve("out.regs", n_out)
        b.dma.store_now(OUT_BASE, out_regs, n_out)
        outs = [(OUT_BASE, out_regs, n_out)]
        for s in (0, 1):
            n_tiles = sum(1 for t in DATA_TILES if t // SEGMENT == s)
            d = b.dram.reserve(f"out.vmem{s}", n_tiles * FP8_TILE)
            b.dma.store_now(s * SEGMENT, d, n_tiles * FP8_TILE)
            outs.append((s * SEGMENT, d, n_tiles * FP8_TILE))
        if self.nv:
            d = b.dram.reserve("out.scales", 128 * len(SCALE_TILES))
            b.dma.store_now(SCALE_TILES[0], d, 128 * len(SCALE_TILES))
            outs.append((SCALE_TILES[0], d, 128 * len(SCALE_TILES)))
        return b.finish(), outs


def make(seed: int, length: int, units=("mxu1",), spaced: bool = False) -> dict:
    from .flow import assemble, preload
    g = Gen(seed, length, units, spaced)
    source, outs = g.build()
    sim = FuncSim(source, g.b.dram.blobs, g.b.dram.size)
    sim.run()
    assert sim.dbg[0] == 1
    check = []
    for _, dram, nbytes in outs:
        data = sim.read_dram(dram, nbytes)
        words = torch.frombuffer(bytearray(data), dtype=torch.int32).numpy().astype("uint32").tolist()
        check.append([dram, words])
    words = assemble(source)
    tag = "+".join(units) + (" spaced" if spaced else "")
    return {"name": f"hazfuzz_{seed:04d}", "isa": list(ISA_VERSION), "variant": f"{length} ops {tag}",
            "program": words, "preload": preload(g.b.dram.blobs), "check": check, "perf": {}, "timeout": None,
            "source": source, "stats": g.stats}


def directed_vload_war() -> dict:
    """vload into the second register of a pair that a VPU instruction still reads.

    vadd.bf16 m8, m2, m6 reads m2 for 32 cycles, then m3. A vload of m3 right after it must wait
    for that read (WAR); if it does not, m9 = Z + B_hi instead of A_hi + B_hi."""
    from .flow import assemble, preload
    b = Builder(schedule=False)
    b.begin()
    gen = torch.Generator().manual_seed(1)
    tiles = {}
    for i, name in enumerate(("a0", "a1", "b0", "b1", "z")):
        v = torch.randn(32, 16, generator=gen) * 2
        tiles[name] = i * FP8_TILE
        addr = b.dram.add(name, v.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
        b.dma.load_now(tiles[name], addr, FP8_TILE, block=name)
    b.vload(2, tiles["a0"]); b.vload(3, tiles["a1"])
    b.vload(6, tiles["b0"]); b.vload(7, tiles["b1"])
    b.v("vadd.bf16", "m8", "m2", "m6")
    b.vload(3, tiles["z"])
    b.vstore(8, OUT_BASE); b.vstore(9, OUT_BASE + FP8_TILE); b.vstore(3, OUT_BASE + 2 * FP8_TILE)
    out = b.dram.reserve("out", 3 * FP8_TILE)
    b.dma.store_now(OUT_BASE, out, 3 * FP8_TILE)
    source = b.finish()
    sim = FuncSim(source, b.dram.blobs, b.dram.size)
    sim.run()
    words = torch.frombuffer(bytearray(sim.read_dram(out, 3 * FP8_TILE)), dtype=torch.int32).numpy().astype("uint32").tolist()
    return {"name": "hazard_vload_war_vpu_pair", "isa": list(ISA_VERSION), "variant": "directed",
            "program": assemble(source), "preload": preload(b.dram.blobs), "check": [[out, words]], "perf": {},
            "timeout": None, "source": source}


def directed_rows(units=("mxu1",)) -> dict:
    """Instructions of a few rows (vsetm) where the interlock must order them: matmul chains on one
    accumulator with M below the matmul spacing of mxu1 (4) and of the NVFP4 MXU (8), on
    alternating accumulators, pops right after them, VPU instructions that read the rows of an
    M-row result, a 32-row reader of an M-row result, and vload and vstore of M lines next to VPU
    reads of their register."""
    from .flow import assemble, preload
    b = Builder(schedule=False)
    b.begin()
    gen = torch.Generator().manual_seed(2)
    # Tiles 0-3: FP8 (m0-m2 A, m3 weights); tiles 4-7: BF16 (m4-m7, and every other register the
    # program writes, so that all their rows hold data).
    nv = "nv" in units
    out_regs = list(range(4, 32)) + ([32, 33, 34, 35] if nv else [])
    for i in range(8):
        data = (fp8_codes(1024, gen).numpy().tobytes() if i < 4
                else (torch.randn(32, 16, generator=gen) * 2).to(torch.bfloat16).view(torch.int16).numpy().tobytes())
        b.dma.load_now(i * FP8_TILE, b.dram.add(f"t{i}", data), FP8_TILE, block=f"t{i}")
    for r in range(4):
        b.vload(r, r * FP8_TILE)
    for r in out_regs:
        b.vload(r, (4 + r % 4) * FP8_TILE)
    code = ["vmatpush.weight.mxu1 w0, m3", "vmatpush.acc.bf16.mxu1 acc0, m4", "vmatpush.acc.bf16.mxu1 acc1, m6",
            "vsetm 1",
            "vmatmul.acc.mxu1 acc0, m0, w0", "vmatmul.acc.mxu1 acc0, m1, w0", "vmatmul.acc.mxu1 acc0, m2, w0",
            "vmatpop.bf16.acc.mxu1 m8, acc0",
            "vsetm 2",
            "vmatmul.acc.mxu1 acc1, m0, w0", "vmatmul.acc.mxu1 acc0, m1, w0", "vmatmul.acc.mxu1 acc1, m2, w0",
            "vmatmul.acc.mxu1 acc0, m0, w0", "vmatpop.fp32.acc.mxu1 m12, acc1", "vmatmul.mxu1 acc1, m1, w0",
            "vsetm 3",
            "vmatpop.bf16.acc.mxu1 m10, acc0", "vadd.bf16 m16, m10, m4", "vmul.bf16 m18, m16, m16",
            "vsetm 32",
            "vmov m20, m18",
            "vsetm 5",
            "vredsum.bf16 m22, m4", f"vload m5, {b._vaddr(6 * FP8_TILE)}", f"vstore m18, {b._vaddr(7 * FP8_TILE)}",
            "vsetm 7",
            "vsquare.bf16 m24, m6", f"vload m7, {b._vaddr(4 * FP8_TILE)}", "vmatpop.bf16.acc.mxu1 m26, acc1"]
    if nv:
        scales = b.dram.add("scales", ue4m3_codes(4 * 128, gen).numpy().tobytes())
        b.dma.load_now(SCALE_TILES[0], scales, 4 * 128, block="scales")
        for r in range(4):
            b.op(f"vload.scale m{r}, {b._vaddr(SCALE_TILES[r])}")
        code += ["vsetm 32", "vmatpush.weight.nv w1, m3", "vmatpush.acc.bf16.nv acc0, m4",
                 "vsetm 6",
                 "vmatmul.acc.nv acc0, m0, w1", "vmatmul.acc.nv acc0, m1, w1", "vmatmul.nv acc1, m2, w1",
                 "vmatmul.acc.nv acc0, m2, w1", "vmatpop.bf16.acc.nv m28, acc0",
                 "vsetm 1",
                 "vmatmul.acc.nv acc1, m0, w1", "vmatmul.acc.nv acc1, m1, w1", "vmatpop.fp32.acc.nv m32, acc1",
                 "vsetm 7",
                 "vmatmul.acc.nv acc0, m1, w1", "vmatmul.acc.nv acc0, m0, w1", "vmatpop.bf16.acc.nv m30, acc0"]
    for line in code:
        b.op(line)
    b.op("vsetm 32")
    for j, r in enumerate(out_regs):
        b.vstore(r, OUT_BASE + j * FP8_TILE)
    # The registers, and tile 7, which holds 5 rows of m18 (vstore at M = 5).
    checks = []
    for k, (vmem, n) in enumerate([(OUT_BASE, len(out_regs) * FP8_TILE), (7 * FP8_TILE, FP8_TILE)]):
        dram = b.dram.reserve(f"out{k}", n)
        b.dma.store_now(vmem, dram, n)
        checks.append((dram, n))
    source = b.finish()
    sim = FuncSim(source, b.dram.blobs, b.dram.size)
    sim.run()
    check = [[dram, torch.frombuffer(bytearray(sim.read_dram(dram, n)), dtype=torch.int32).numpy().astype("uint32").tolist()]
             for dram, n in checks]
    return {"name": "hazard_rows" + ("_nv" if nv else ""), "isa": list(ISA_VERSION), "variant": "directed",
            "program": assemble(source), "preload": preload(b.dram.blobs), "check": check, "perf": {},
            "timeout": None, "source": source}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--length", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--units", default="mxu1", help="comma-separated: mxu1, nv")
    ap.add_argument("--spaced", action="store_true", help="DELAY 200 after every tensor instruction")
    ap.add_argument("--directed", action="store_true", help="also write the directed hazard tests")
    a = ap.parse_args(argv)
    units = tuple(a.units.split(","))
    if "mxu1" not in units or not set(units) <= {"mxu1", "nv"}:
        ap.error(f"--units {a.units}: give mxu1 (default) or mxu1,nv (ipt-nvfp4)")
    a.out.mkdir(parents=True, exist_ok=True)
    if a.directed:
        for t in (directed_vload_war(), directed_rows(units)):
            (a.out / f"{t['name']}.json").write_text(json.dumps({k: v for k, v in t.items() if k != "source"}))
            (a.out / f"{t['name']}.S").write_text(t["source"])
    total = {}
    for i in range(a.count):
        t = make(a.seed + i, a.length, units, a.spaced)
        (a.out / f"{t['name']}.json").write_text(json.dumps({k: v for k, v in t.items() if k != "source"}))
        (a.out / f"{t['name']}.S").write_text(t["source"])
        for k, v in t["stats"].items():
            total[k] = total.get(k, 0) + v
    print(f"{a.count} programs in {a.out}:", ", ".join(f"{k} {v}" for k, v in sorted(total.items())))


if __name__ == "__main__":
    main()

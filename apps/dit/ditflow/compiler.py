"""Compile the DiT sampler to one Atlas program with its DRAM image.

The program runs the whole flow-matching sampler: for each of the Euler steps it embeds the
timestep, runs the time MLP and the adaLN projections, the transformer blocks and the output
projection, and integrates x += dt * v. The host only writes the DRAM image (weights,
constants, the noise x0 and the conditioning vector) and reads back x, plus each step's
velocity when tracing.

Numbers on Atlas:
- Matmul inputs are FP8 E4M3 with the per-tensor power-of-two scales of a QuantPlan; mxu1
  accumulates in FP32 over all K tiles and rounds once at the BF16 pop. Everything else is BF16
  on the VPU.
- adaLN is linear in cat(SiLU(time), SiLU(cond)), so its conditioning half (and its bias,
  with the +1 of `x * (1 + scale)` folded in) is computed once before the loop.
- The timestep path and adaLN run on one row (vsetm 1). The timestep is a .vx operand (lw of its
  BF16 bits). adaLN's shift, scale and gate are row 0 of their tiles, which the kernels read with
  .vr. A conditioning row q > 0 of the encoder's tiles moves to row 0 through the MXU (E @ Z^T,
  E the FP8 tile with ones in column q).
- The time MLP's exact GELU is evaluated in its tanh form, the only one the VPU has.
- BF16 keeps 8 significant bits, so x is carried as hi + lo BF16 (a compensated sum) across the
  100 steps. VPU adds round to nearest even, so lo is the exact error of hi + d when |hi| >= |d|
  (Fast2Sum).
"""
import math
from dataclasses import dataclass, field

import torch

from . import layout as L
from . import model as M
from .asm import Builder, Tensor, DRAM_BASE
from .kernels import Kernels, Linear
from .loopk import LoopKernels, Stream
from .quant import QuantPlan


@dataclass
class Compiled:
    d: M.DiTDims
    steps: int
    source: str
    blobs: list
    regions: dict
    manifest: dict = field(default_factory=dict)


class DiTProgram:
    def __init__(self, d: M.DiTDims, plan: QuantPlan, x0: torch.Tensor, cond: torch.Tensor | None,
                 steps: int | None = None, trace: bool = True, dram_size: int = 4 << 20,
                 builder: Builder | None = None, cond_tiles=None, cond_perm=None, state=None,
                 schedule: bool = True):
        """cond: the conditioning vector, from DRAM. Or, when the observation encoder runs in the
        same program: cond_tiles, (VMEM tile, row) per conditioning K tile; cond_perm, the
        conditioning index of each of their columns (-1: none; the state is the last tile); state."""
        assert d.head_dim == 64 and d.horizon <= 64
        self.d, self.plan, self.steps, self.trace = d, plan, steps or d.steps, trace
        self.b = builder or Builder(dram_size, schedule)
        self.cond_tiles, self.cond_perm = cond_tiles, cond_perm
        b = self.b
        self.T, self.H = d.horizon, d.hidden
        self.HT, self.RT = L.tiles(d.hidden), L.tiles(d.horizon)
        a = plan.act
        self.a = a

        # ---------------- DRAM image: weights --------------------------------------------
        p, codes, w = plan.p, plan.codes, plan.weight
        self.lin = {}

        def linear(name, wname, bias, pad_k=None):
            c = codes[wname]
            if pad_k:
                c = L.pad2(c, c.shape[0], pad_k)
            addr = b.dram.add(name, L.weight_blocks(c, bias))
            self.lin[name] = Linear(name, addr, L.tiles(c.shape[0]), L.tiles(c.shape[1]), bias is not None, w[wname])

        linear("time1", "time1.w", p["time1.b"])
        linear("time2", "time2.w", p["time2.b"])
        plus_one = torch.zeros(6 * d.hidden)
        plus_one[d.hidden:2 * d.hidden] = 1.0           # scale_msa
        plus_one[4 * d.hidden:5 * d.hidden] = 1.0       # scale_mlp
        if cond_perm is not None:                         # adaLN columns in the order of cond_tiles
            for l in range(d.layers):
                c = codes[f"{l}.ada_c.w"]
                cp = torch.zeros(c.shape[0], len(cond_perm), dtype=c.dtype)
                for i, src in enumerate(cond_perm):
                    if src >= 0:
                        cp[:, i] = c[:, src]
                codes = dict(codes)
                codes[f"{l}.ada_c.w"] = cp
        for l in range(d.layers):
            linear(f"{l}.ada_c", f"{l}.ada_c.w", p[f"{l}.ada.b"] + plus_one)
            linear(f"{l}.ada_t", f"{l}.ada_t.w", None)
        linear("in", "in.w", p["in.b"])
        for l in range(d.layers):
            for n in ("qkv", "o", "up", "down"):
                linear(f"{l}.{n}", f"{l}.{n}.w", p[f"{l}.{n}.b"])
        out_codes = L.pad2(codes["out.w"], 32, codes["out.w"].shape[1])
        addr = b.dram.add("out", L.weight_blocks(out_codes, L.pad2(p["out.b"].reshape(1, -1), 1, 32)[0]))
        self.lin["out"] = Linear("out", addr, 1, self.HT, True, w["out.w"])

        # ---------------- DRAM image: constants and inputs -------------------------------
        cos, sin = M.rope_tables(d)
        grid = M.time_grid(d)
        dt = (grid[1] - grid[0]).item()
        self.dt_hi = L.bf16_bits(dt)
        self.dt_lo = L.bf16_bits(dt - torch.tensor(dt).to(torch.bfloat16).float().item())
        kmask = None
        if self.T % 32:
            km = torch.zeros(32, 32)
            km[:, self.T % 32:] = -30000.0
            kmask = km
        rowmask = torch.zeros(self.RT * 32, 32)
        rowmask[:self.T] = 1.0
        tbits = grid[:self.steps].to(torch.bfloat16).view(torch.int16).to(torch.int32) << 16    # x[31:16]
        ttable = bytearray(tbits.numpy().tobytes())
        ttable += bytes(-len(ttable) % 32)
        hi = x0[0].to(torch.bfloat16)
        lo = (x0[0] - hi.float()).to(torch.bfloat16)
        consts = {
            "rope_cos": cos, "rope_sin": sin, "freq": M.sinusoidal_freqs(d).reshape(1, -1),
            "rowmask": rowmask, "x_hi": hi, "x_lo": lo,
        }
        if cond_tiles is None:
            consts["cond"] = cond.reshape(1, -1)
        else:
            consts["state"] = state.reshape(1, -1)
        if kmask is not None:
            consts["kmask"] = kmask
        self.vm: dict[str, Tensor] = {}
        self._const_loads = []
        for name, m in consts.items():
            daddr = b.dram.add(name, L.bf16_bytes(m))
            t = b.vmem.tensor(name, m.shape[0], m.shape[1], "bf16")
            self.vm[name] = t
            self._const_loads.append((t.addr, daddr, t.nbytes))
        rows = sorted({q for _, q in cond_tiles or () if q})
        if rows:                                          # FP8 tiles with 1.0 in column q (row q to row 0)
            eq = torch.zeros(32 * len(rows), 32, dtype=torch.uint8)
            for i, q in enumerate(rows):
                eq[32 * i:32 * i + 32, q] = 0x38
            self.eq_rows = rows
            self.vm["e_cols"] = b.vmem.tensor("e_cols", 32 * len(rows), 32, "fp8")
            self._const_loads.append((self.vm["e_cols"].addr, b.dram.add("e_cols", L.fp8_bytes(eq)),
                                      self.vm["e_cols"].nbytes))
        tt_addr = b.dram.add("ttable", bytes(ttable))
        self.vm_ttable = b.vmem.alloc("ttable", len(ttable))
        self._const_loads.append((self.vm_ttable, tt_addr, len(ttable)))

        # ---------------- DRAM: outputs ---------------------------------------------------
        xbytes = self.RT * L.BF16_TILE
        self.out_hi = b.dram.reserve("out.x_hi", xbytes)
        self.out_lo = b.dram.reserve("out.x_lo", xbytes)
        self.out_trace = b.dram.reserve("out.trace", xbytes * self.steps) if trace else None

        # ---------------- VMEM working set -------------------------------------------------
        vt = lambda name, rows, cols, kind: self.vm.setdefault(name, b.vmem.tensor(name, rows, cols, kind))
        C, Tdim = d.cond_dim, d.tdim
        vt("condq", 32, len(cond_perm) if cond_perm is not None else C, "fp8")
        for l in range(d.layers):
            vt(f"{l}.condmod", 32, 6 * d.hidden, "bf16")
            vt(f"{l}.mod", 32, 6 * d.hidden, "bf16")
        vt("embq", 32, Tdim, "fp8")
        vt("t1q", 32, 2 * Tdim, "fp8")
        vt("stq", 32, Tdim, "fp8")
        vt("x", self.T, self.H, "bf16")
        vt("xq", self.T, self.H, "fp8")
        vt("qkv", self.T, 3 * self.H, "bf16")
        vt("attnq", self.T, self.H, "fp8")
        vt("hq", self.T, 4 * self.H, "fp8")
        vt("xin_q", self.T, 32, "fp8")
        vt("vbuf", self.T, 32, "bf16")
        stage = max(lin.block_bytes for lin in self.lin.values())
        self.k = Kernels(b, stage, banked=True)
        self.lk = LoopKernels(b, self.k)

    # -------------------------------------------------------------------------------------
    def chunk(self, l: int, c: int) -> Tensor:
        """Tile view of adaLN output chunk c (shift, scale + 1, gate, shift, scale + 1, gate) of layer l."""
        mod = self.vm[f"{l}.mod"]
        return Tensor(f"{l}.mod.{c}", mod.tile(0, c * self.HT), 32, self.H, "bf16")

    def compile(self, standalone: bool = True) -> Compiled:
        b, k, d, a, vm = self.b, self.k, self.d, self.a, self.vm
        if standalone:
            b.begin()
        b.prog.comment("constants and inputs")
        for vaddr, daddr, n in self._const_loads:
            b.dma.load("const", vaddr, daddr, n)
        b.dma.wait("const")
        self.prologue()
        k.quantize(vm["x_hi"], vm["xin_q"], a["x"])

        b.li(5, 0)
        b.li(6, self.steps)
        b.li(7, self.vm_ttable)
        if self.trace:
            b.li(9, self.out_trace)
        b.label("step")
        b.prog.comment("---- sampler step ----")
        self.time_path()
        self.lin_call("in", vm["xin_q"], a["x"], k.epi_store(vm["x"]))
        for l in range(d.layers):
            self.block(l)
        k.quantize(vm["x"], vm["xq"], a["h"])
        self.lin_call("out", vm["xq"], a["h"], self.epi_euler)
        if self.trace:
            vb = vm["vbuf"]
            b.set_x(1, vb.addr // 4)
            b.set_x(3, vb.nbytes)
            b.op("dma.store x9, x1, x3, ch7")
            b.op("dma.wait ch7")
            b.li(10, vb.nbytes)
            b.op("ADD x9, x9, x10")
        b.op("ADDI x7, x7, 4")
        b.op("ADDI x5, x5, 1")
        b.op("CSRRW x0, 0x801, x5")                     # DBG1: steps done
        b.branch_back(5, 6, "step")
        b.dma.store_now(vm["x_hi"].addr, self.out_hi, vm["x_hi"].nbytes)
        b.dma.store_now(vm["x_lo"].addr, self.out_lo, vm["x_lo"].nbytes)
        source = b.finish()
        manifest = {
            "dims": d.to_dict(), "steps": self.steps, "trace": self.trace,
            "outputs": {"x_hi": self.out_hi, "x_lo": self.out_lo, "trace": self.out_trace,
                        "x_bytes": vm["x_hi"].nbytes},
            "scales": self.plan.to_dict(),
            "dram_bytes": b.dram.top, "vmem_bytes": b.vmem.top,
            "dram_regions": {n: [hex(r[0]), r[1]] for n, r in b.dram.regions.items()},
        }
        return Compiled(d, self.steps, source, b.dram.blobs, b.dram.regions, manifest)

    def lin_call(self, name: str, xq: Tensor, a_x: int, epilogue) -> None:
        lin = self.lin[name]
        self.k.linear(lin, xq, a_x + lin.a_w, epilogue)

    # -------------------------------------------------------------------------------------
    def prologue(self) -> None:
        """condmod[l] = SiLU(cond) @ W_c[l]^T + b[l] (+1 on the scales), in row 0."""
        b, k, a, vm = self.b, self.k, self.a, self.vm
        b.prog.comment("prologue: conditioning half of adaLN")
        condq = vm["condq"]
        if self.cond_tiles is None:
            cond = vm["cond"]
            sources = [(cond.tile(0, j), 0) for j in range(cond.ct)]
        else:
            sources = [(t.tile(0, 0), q) for t, q in self.cond_tiles] + [(vm["state"].tile(0, 0), 0)]
        E = {}
        for j, (src, row) in enumerate(sources):
            p = b.regs.pair()
            b.vload2(p, src)
            k.silu(p)
            if not row:
                k.quant_store(p, 0, condq.tile(0, j), 127 - a["ada_c"])
                b.regs.release(pairs=(p,))
                continue
            if row not in E:
                E[row] = b.regs.one()
                b.vload(E[row], vm["e_cols"].tile(self.eq_rows.index(row), 0))
            zq = k.quant_store(p, 0, None, 127 - a["ada_c"], keep=True)
            zt = b.regs.one()
            b.v("vtrpose.xlu", f"m{zt}", f"m{zq}")
            b.v("vmatpush.weight.mxu1", "w0", f"m{zt}")
            b.v("vmatmul.mxu1", "acc1", f"m{E[row]}", "w0")                 # every row = row `row`
            b.v("vmatpop.fp8.acc.mxu1", f"m{zq}", "acc1", f"e{b.erf(127)}")
            b.vstore(zq, condq.tile(0, j))
            b.regs.release(zq, zt, pairs=(p,))
        b.regs.release(*E.values())
        b.vsetm(1)
        for l in range(self.d.layers):
            lin = self.lin[f"{l}.ada_c"]
            self.lk.linear_rows(lin, lin.addr, condq, a["ada_c"] + lin.a_w,
                                [Stream("cm", vm[f"{l}.condmod"], per_row=False)], self.lk.epi_store("cm"))
        b.vsetm(32)

    def time_path(self) -> None:
        """mod[l] = SiLU(time_mlp(t)) @ W_t[l]^T + condmod[l], in row 0 (M = 1)."""
        b, k, a, vm, d = self.b, self.k, self.a, self.vm, self.d
        b.prog.comment("timestep embedding")
        b.vsetm(1)
        b.op("lw x4, 0(x7)")                            # t in x4[31:16]
        half_tiles = L.tiles(d.tdim // 2)
        embq = vm["embq"]
        for j in range(half_tiles):
            f, s = b.regs.pair(), b.regs.pair()
            b.vload2(f, vm["freq"].tile(0, j))
            b.v("vmul.bf16.vx", f"m{f}", f"m{f}", "x4")
            b.v("vsin.bf16", f"m{s}", f"m{f}")
            k.quant_store(s, 0, embq.tile(0, j), 127 - a["temb"])
            b.v("vcos.bf16", f"m{s}", f"m{f}")
            k.quant_store(s, 1, embq.tile(0, half_tiles + j), 127 - a["temb"])
            b.regs.release(pairs=(f, s))
        self.lin_call("time1", embq, a["temb"], k.epi_quant(vm["t1q"], a["t1"], pre=lambda o, r: k.gelu(o)))

        def gelu_silu(o, r):
            k.gelu(o)
            k.silu(o)
        self.lin_call("time2", vm["t1q"], a["t1"], k.epi_quant(vm["stq"], a["ada_t"], pre=gelu_silu))
        for l in range(d.layers):
            lin = self.lin[f"{l}.ada_t"]
            self.lk.linear_rows(lin, lin.addr, vm["stq"], a["ada_t"] + lin.a_w,
                                [Stream("cm", vm[f"{l}.condmod"], per_row=False), Stream("mod", vm[f"{l}.mod"], per_row=False)],
                                self.epi_add_mod)
        b.vsetm(32)

    def epi_add_mod(self, o: int, r: int, ptrs: dict) -> None:
        b = self.b
        g = b.regs.pair()
        b.vload2(g, ptrs["cm"])
        b.v("vadd.bf16", f"m{o}", f"m{o}", f"m{g}")
        b.vstore2(o, ptrs["mod"])
        b.regs.release(pairs=(g,))

    def block(self, l: int) -> None:
        b, k, a, vm, d = self.b, self.k, self.a, self.vm, self.d
        b.prog.comment(f"---- transformer block {l} ----")
        k.ln_modulate_quant(vm["x"], self.chunk(l, 1), self.chunk(l, 0), vm["xq"], a[f"{l}.ln1"])
        self.lin_call(f"{l}.qkv", vm["xq"], a[f"{l}.ln1"], k.epi_store(vm["qkv"]))
        k.attention(vm["qkv"], vm["rope_cos"], vm["rope_sin"], vm.get("kmask"), vm["attnq"], d.heads,
                    a[f"{l}.q"], a[f"{l}.k"], a[f"{l}.v"], a[f"{l}.p"], a[f"{l}.attn"])
        self.lin_call(f"{l}.o", vm["attnq"], a[f"{l}.attn"], k.epi_residual_gate(vm["x"], self.chunk(l, 2)))
        k.ln_modulate_quant(vm["x"], self.chunk(l, 4), self.chunk(l, 3), vm["xq"], a[f"{l}.ln2"])
        self.lin_call(f"{l}.up", vm["xq"], a[f"{l}.ln2"],
                      k.epi_quant(vm["hq"], a[f"{l}.gelu"], pre=lambda o, r: k.gelu(o)))
        self.lin_call(f"{l}.down", vm["hq"], a[f"{l}.gelu"], k.epi_residual_gate(vm["x"], self.chunk(l, 5)))

    def epi_euler(self, o: int, r: int, n: int) -> None:
        """v = o: trace it, then x += dt * v on the hi/lo pair (padding rows held at 0)."""
        b, k, vm = self.b, self.k, self.vm
        b.vstore2(o, vm["vbuf"].tile(r, 0))
        dd, kk, t = b.regs.pair(), b.regs.pair(), b.regs.pair()
        b.v("vmul.bf16.vx", f"m{dd}", f"m{o}", b.xconst(self.dt_hi))
        b.v("vmul.bf16.vx", f"m{o}", f"m{o}", b.xconst(self.dt_lo))
        b.v("vadd.bf16", f"m{dd}", f"m{dd}", f"m{o}")             # d = dt * v
        b.vload2(kk, vm["rowmask"].tile(r, 0))
        b.v("vmul.bf16", f"m{dd}", f"m{dd}", f"m{kk}")
        b.vload2(kk, vm["x_lo"].tile(r, 0))
        b.v("vadd.bf16", f"m{dd}", f"m{dd}", f"m{kk}")            # d += lo
        b.vload2(kk, vm["x_hi"].tile(r, 0))
        b.v("vadd.bf16", f"m{o}", f"m{kk}", f"m{dd}")             # s = hi + d
        b.v("vsub.bf16", f"m{t}", f"m{o}", f"m{kk}")              # z = s - hi
        b.v("vsub.bf16", f"m{t}", f"m{dd}", f"m{t}")              # lo = d - z
        b.vstore2(o, vm["x_hi"].tile(r, 0))
        b.vstore2(t, vm["x_lo"].tile(r, 0))
        k.quant_store(o, r, vm["xin_q"].tile(r, 0), 127 - self.a["x"])
        b.regs.release(pairs=(dd, kk, t))


def compile_dit(d, plan, x0, cond, steps=None, trace=True, schedule=True) -> Compiled:
    return DiTProgram(d, plan, x0, cond, steps=steps, trace=trace, schedule=schedule).compile()


def read_outputs(c: Compiled, read) -> dict:
    """x (FP32 hi + lo) and the per-step velocities from a reader read(addr, nbytes) -> bytes."""
    d, out = c.d, c.manifest["outputs"]
    n = out["x_bytes"]
    hi = L.from_bf16_bytes(read(out["x_hi"], n), d.horizon, d.action_dim).float()
    lo = L.from_bf16_bytes(read(out["x_lo"], n), d.horizon, d.action_dim).float()
    res = {"x": (hi + lo)[None], "x_hi_bits": read(out["x_hi"], n), "x_lo_bits": read(out["x_lo"], n)}
    if out["trace"] is not None:
        raw = read(out["trace"], n * c.steps)
        res["trace_bits"] = raw
        res["v"] = [L.from_bf16_bytes(raw[i * n:(i + 1) * n], d.horizon, d.action_dim).float()[None]
                    for i in range(c.steps)]
    return res

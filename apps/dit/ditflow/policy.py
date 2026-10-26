"""The whole specialist_dit policy on Atlas: observation encoder, then the DiT sampler.

One program takes the two normalized camera crops, the anchor-frame state and the noise, and
returns the action chunk. The encoder (vision.py) runs first and leaves each view's pooled
tokens in VMEM; the DiT program (compiler.py) then builds its conditioning from them.

The encoder's four layers share one loop body, so each of its tensors has one FP8 scale for all
layers (the largest calibrated range) and each weight type one exponent.
"""
import math

import torch

from . import compiler as C
from . import layout as L
from . import model as M
from . import quant as Q
from . import vision as V
from .asm import Builder, Tensor
from .kernels import Kernels, Linear
from .loopk import LoopKernels, Stream

LAYER_W = ("qkv", "o", "up", "down")
LAYER_P = ("ln1.w", "ln1.b", "ln2.w", "ln2.b", "ls1", "ls2")
E_COLS = 4           # FP8 tiles with 1.0 in column q, q < E_COLS (the DiT replicates row q)


class VisionPlan:
    """FP8 exponents of the encoder: one per tensor type, shared by the layers (and the views)."""

    def __init__(self, p: dict, v: V.VisionDims, act_amax: dict[str, float], margin: float = 2.0):
        self.v = v
        merged: dict[str, float] = {}
        for name, a in act_amax.items():
            key = self.key(name)
            merged[key] = max(merged.get(key, 0.0), a)
        self.act_amax = merged
        self.act = {k: Q.exponent_for(a, margin) for k, a in merged.items()}
        wmax: dict[str, float] = {}
        for name in V.weight_names(v):
            key = self.key(name)
            wmax[key] = max(wmax.get(key, 0.0), float(p[name].abs().max()))
        self.weight = {k: Q.exponent_for(a) for k, a in wmax.items()}
        self.codes = {name: Q.to_fp8(p[name], self.weight[self.key(name)]) for name in V.weight_names(v)}
        self.p = dict(p)
        for name in V.weight_names(v):
            self.p[name] = Q.from_fp8(self.codes[name], self.weight[self.key(name)])

    @staticmethod
    def key(name: str) -> str:
        """'2.qkv.w' -> 'L.qkv.w', '3.ln1' -> 'L.ln1'; others unchanged."""
        head, _, rest = name.partition(".")
        return "L." + rest if head.isdigit() else name

    def q(self, name: str, t: torch.Tensor) -> torch.Tensor:
        return Q.fake(t, self.act[self.key(name)])

    def to_dict(self) -> dict:
        return {"act": self.act, "weight": self.weight}


def random_observation(v: V.VisionDims, d: M.DiTDims, seed: int):
    g = torch.Generator().manual_seed(seed)
    views = [torch.randn(1, 3, s, s, generator=g) for _, s in v.views]
    state = torch.randn(1, v.state_dim, generator=g)
    x0 = torch.randn(1, d.horizon, d.action_dim, generator=g)
    return views, state, x0


class PolicyPlans:
    """Calibrate the encoder on random crops, then the DiT on the conditioning the encoder gives."""

    def __init__(self, vp: dict, dp: dict, v: V.VisionDims, d: M.DiTDims, seeds=(201, 202, 203, 204)):
        amax: dict[str, float] = {}

        def observe(name, t):
            amax[name] = max(amax.get(name, 0.0), float(t.abs().max()))
            return t

        conds = []
        with torch.no_grad():
            for seed in seeds:
                views, state, _ = random_observation(v, d, seed)
                conds.append(V.encode(vp, state, views, v, q=observe))
        self.vision = VisionPlan(vp, v, amax)
        damax: dict[str, float] = {}

        def observe_d(name, t):
            damax[name] = max(damax.get(name, 0.0), float(t.abs().max()))
            return t

        with torch.no_grad():
            for seed, cond in zip(seeds, conds):
                _, _, x0 = random_observation(v, d, seed)
                M.sample(dp, x0, cond, d, q=observe_d)
        self.dit = Q.QuantPlan(dp, d, damax)


def reference(enc, net, v, d, views, state, x0, steps=None):
    """FP32: specialist's PooledVision and ObservationEncoder arithmetic, LeRobot's DiT sampler."""
    cond = V.encode_reference(enc, state, views)
    from .flow import _partial
    x, vs = _partial(net, x0, cond, d, steps or d.steps)
    return cond, x, vs


def fake_quant(plans: PolicyPlans, v, d, views, state, x0, steps=None):
    """The same with Atlas's FP8 points emulated in FP32 (exact GELU kept)."""
    cond = V.encode(plans.vision.p, state, views, v, q=plans.vision.q)
    from .flow import _partial_fq
    x, vs = _partial_fq(plans.dit, x0, cond, d, steps or d.steps)
    return cond, x, vs


# --------------------------------------------------------------------------------------------
# The program
# --------------------------------------------------------------------------------------------

class EncoderEmitter:
    """Emits the encoder for every view into a Builder; leaves pooled tiles in `outs`."""

    def __init__(self, b: Builder, v: V.VisionDims, plan: VisionPlan, views: list[torch.Tensor]):
        self.b, self.v, self.plan = b, v, plan
        h, HT = v.hidden, L.tiles(v.hidden)
        self.HT = HT
        a, w, codes, p = plan.act, plan.weight, plan.codes, plan.p
        self.a = a
        self.lin = {}

        def place(name, wname, bias, wkey=None):
            c = codes[wname]
            addr = b.dram.add(name, L.weight_blocks(c, bias))
            self.lin[name] = Linear(name, addr, L.tiles(c.shape[0]), L.tiles(c.shape[1]), bias is not None,
                                    w[wkey or VisionPlan.key(wname)])
            return addr

        # ---- DRAM: weights (layers contiguous, one stride) and per-layer parameters ----
        place("pe", "pe.w", p["pe.b"])
        self.layer_w = []
        for l in range(v.layers):
            base = None
            for n in LAYER_W:
                addr = place(f"{l}.{n}", f"{l}.{n}.w", p[f"{l}.{n}.b"])
                base = addr if base is None else base
            self.layer_w.append(base)
        self.w_off = {n: self.lin[f"0.{n}"].addr - self.layer_w[0] for n in LAYER_W}
        self.w_stride = self.layer_w[1] - self.layer_w[0] if v.layers > 1 else 0
        for l in range(1, v.layers):
            assert all(self.lin[f"{l}.{n}"].addr - self.layer_w[l] == self.w_off[n] for n in LAYER_W)
        rep = lambda t: L.bf16_bytes(L.replicated(t))
        self.layer_p = [b.dram.add(f"{l}.params", b"".join(rep(p[f"{l}.{n}"]) for n in LAYER_P)) for l in range(v.layers)]
        self.p_stride = self.layer_p[1] - self.layer_p[0] if v.layers > 1 else 0
        self.p_bytes = len(LAYER_P) * HT * L.BF16_TILE
        for n in ("pool.q", "pool.kv", "pool.out", "pool.up", "pool.down", "proj"):
            place(n, f"{n}.w", p[f"{n}.b"])
        self.statics = {n: b.dram.add(n, rep(p[n])) for n in ("norm.w", "norm.b", "pool.ln.w", "pool.ln.b")}
        self.query_dram = b.dram.add("pool.query", L.bf16_bytes(p["pool.query"]))

        # ---- DRAM: per-view inputs and constants ----
        self.view_data = []
        for (name, size), img in zip(v.views, views):
            n = v.patches(size)
            rows = n + v.prefix
            rt = -(-rows // 64) * 2                        # even number of row tiles
            patches = L.pad2(V.patchify(img, v.patch), rt * 32, 3 * v.patch * v.patch)
            cos, sin = V.rope_tables(v, size)
            ctab, stab = torch.ones(rt * 32, 32), torch.zeros(rt * 32, 32)
            ctab[:n], stab[:n] = cos, sin                  # prefix and padding rows: no rotation
            self_mask = torch.zeros(32, rt * 32)
            self_mask[:, rows:] = -30000.0                 # keys: patches and prefix tokens
            pool_mask = torch.zeros(32, rt * 32)
            pool_mask[:, n:] = -30000.0                    # keys: patch tokens only
            pre_tile = n // 32
            assert (n + v.prefix - 1) // 32 == pre_tile, "prefix tokens must share one row tile"
            corr = torch.zeros(32, h)                      # prefix rows: token - bias (their input is 0)
            first = n - pre_tile * 32
            corr[first] = p["cls"] - p["pe.b"]
            corr[first + 1:first + v.prefix] = p["reg"] - p["pe.b"]
            self.view_data.append({
                "name": name, "n": n, "rt": rt, "pre_tile": pre_tile,
                "patches": b.dram.add(f"{name}.patches", L.bf16_bytes(patches)),
                "cos": b.dram.add(f"{name}.cos", L.bf16_bytes(ctab)),
                "sin": b.dram.add(f"{name}.sin", L.bf16_bytes(stab)),
                "self_mask": b.dram.add(f"{name}.self_mask", L.bf16_bytes(self_mask)),
                "pool_mask": b.dram.add(f"{name}.pool_mask", L.bf16_bytes(pool_mask)),
                "corr": b.dram.add(f"{name}.corr", L.bf16_bytes(corr)),
            })
        # ---- VMEM: outputs (kept for the DiT), then the working set ----
        self.outs = [b.vmem.tensor(f"{vd['name']}.pooled", 32, v.token_dim, "bf16") for vd in self.view_data]
        self.mark = b.vmem.mark()
        RT = max(vd["rt"] for vd in self.view_data)
        stage = max(lin.block_bytes for lin in self.lin.values())
        stage = max(stage, (3 * v.patch * v.patch // 32) * L.BF16_TILE // 2 + L.BF16_TILE)
        self.k = Kernels(b, stage)
        assert self.k.stage[1] == self.k.stage[0] + stage
        self.lk = LoopKernels(b, self.k)
        T = lambda name, rows, cols, kind: b.vmem.tensor(name, rows, cols, kind)
        self.x = T("enc.x", RT * 32, h, "bf16")
        self.xq = T("enc.xq", RT * 32, h, "fp8")
        self.big = b.vmem.alloc("enc.big", RT * L.BF16_TILE * 3 * HT)     # qkv | patches FP8 | MLP hidden | pool kv
        self.attnq = T("enc.attnq", RT * 32, h, "fp8")
        self.scratch = {"qbuf": T("enc.qbuf", RT * 32, 64, "fp8"), "kbuf": T("enc.kbuf", RT * 32, 64, "fp8"),
                        "vtbuf": T("enc.vtbuf", 64, RT * 32, "fp8"), "pbuf": T("enc.pbuf", 32, RT * 32, "fp8"),
                        "sbuf": T("enc.sbuf", 32, RT * 32, "bf16")}
        self.cos = T("enc.cos", RT * 32, 32, "bf16")
        self.sin = T("enc.sin", RT * 32, 32, "bf16")
        self.self_mask = T("enc.self_mask", 32, RT * 32, "bf16")
        self.pool_mask = T("enc.pool_mask", 32, RT * 32, "bf16")
        self.corr = T("enc.corr", 32, h, "bf16")
        self.lp = {n: T(f"enc.{n}", 32, h, "bf16") for n in LAYER_P}
        assert self.lp["ls2"].tile(0, 0) == self.lp["ln1.w"].tile(0, 0) + self.p_bytes - HT * L.BF16_TILE
        self.st = {n: T(f"enc.{n}", 32, h, "bf16") for n in self.statics}
        Q = v.queries                                      # the pool's rows: one tile at M = Q
        self.query = T("enc.query", Q, h, "bf16")
        self.qq = T("enc.qq", Q, h, "fp8")
        self.qp = T("enc.qp", Q, h, "bf16")
        self.pattn = T("enc.pattn", Q, h, "fp8")
        self.pooled = T("enc.pooled", Q, h, "bf16")
        self.pm = T("enc.pm", Q, h, "fp8")
        self.ph = T("enc.ph", Q, v.mlp, "fp8")
        self.proj_out = T("enc.proj", Q, v.token_dim, "bf16")

    def big_view(self, name, rows, cols, kind) -> Tensor:
        t = Tensor(name, self.big, rows, cols, kind)
        assert t.nbytes <= 3 * self.HT * L.BF16_TILE * (self.x.rt), name
        return t

    def emit(self) -> None:
        b, v, a, lk, k = self.b, self.v, self.a, self.lk, self.k
        b.prog.comment("==== observation encoder ====")
        for n, t in self.st.items():
            b.dma.load("enc.static", t.addr, self.statics[n], t.nbytes)
        b.dma.load("enc.static", self.query.addr, self.query_dram, self.query.nbytes)
        b.dma.wait("enc.static")
        # Pool queries: input independent, computed once for both views.
        b.vsetm(v.queries)
        lk.quantize_rows(self.query, self.qq, a["pool.qin"])
        lin = self.lin["pool.q"]
        lk.linear_rows(lin, lin.addr, self.qq, a["pool.qin"] + lin.a_w, [Stream("qp", self.qp)], lk.epi_store("qp"))
        b.vsetm(32)
        for vi, vd in enumerate(self.view_data):
            self.emit_view(vi, vd)

    def emit_view(self, vi: int, vd: dict) -> None:
        b, v, a, lk, k = self.b, self.v, self.a, self.lk, self.k
        rows = vd["rt"] * 32
        h, HT = v.hidden, self.HT
        view = lambda t, r=rows: Tensor(t.name, t.addr, r, t.cols, t.kind)
        cview = lambda t, c=rows: Tensor(t.name, t.addr, t.rows, c, t.kind)
        x, xq, attnq = view(self.x), view(self.xq), view(self.attnq)
        cos, sin = view(self.cos), view(self.sin)
        smask, pmask = cview(self.self_mask), cview(self.pool_mask)
        scratch = {"qbuf": view(self.scratch["qbuf"]), "kbuf": view(self.scratch["kbuf"]),
                   "vtbuf": cview(self.scratch["vtbuf"]), "pbuf": cview(self.scratch["pbuf"]),
                   "sbuf": cview(self.scratch["sbuf"])}
        b.prog.comment(f"---- view {vd['name']}: {vd['n']} patches, {vd['rt']} row tiles ----")
        for t, key in ((cos, "cos"), (sin, "sin"), (smask, "self_mask"), (pmask, "pool_mask"), (self.corr, "corr")):
            b.dma.load(f"view{vi}", t.addr, vd[key], t.nbytes)
        b.dma.wait(f"view{vi}")

        # Patches: BF16 rows streamed from DRAM through the staging buffers, quantized to FP8.
        kp = 3 * v.patch * v.patch
        xp = self.big_view("enc.xp", rows, kp, "fp8")
        row_bytes = L.tiles(kp) * L.BF16_TILE
        pd, po = b.xreg(), b.xreg()
        b.li(pd, vd["patches"])
        lk.ptr(po, xp.tile(0, 0))
        stage = k.stage[0]
        with b.loop(vd["rt"], steps=[(pd, row_bytes), (po, xp.ct * L.FP8_TILE // 4)]):
            for i, off in enumerate(range(0, row_bytes, 4096)):
                b.set_x(1, (stage + off) // 4)
                b.add_imm(2, pd, off)
                b.xval[2] = None
                b.set_x(3, min(4096, row_bytes - off))
                b.op(f"dma.load x1, x2, x3, ch{i % 6}")
            for ch in range(min(6, -(-row_bytes // 4096))):
                b.op(f"dma.wait ch{ch}")
            from .asm import Ptr
            for c in range(xp.ct):
                pr = b.regs.pair()
                b.vload2(pr, stage + c * L.BF16_TILE)
                k.quant_store(pr, c % 2, Ptr(po, c * L.FP8_TILE), 127 - a["pe_in"])
                b.regs.release(pairs=(pr,))
        b.xrelease(pd, po)
        lin = self.lin["pe"]
        lk.linear_rows(lin, lin.addr, xp, a["pe_in"] + lin.a_w, [Stream("x", x)], lk.epi_store("x"))
        for c in range(HT):                                             # the prefix tokens
            t, u = b.regs.pair(), b.regs.pair()
            b.vload2(t, x.tile(vd["pre_tile"], c))
            b.vload2(u, self.corr.tile(0, c))
            b.v("vadd.bf16", f"m{t}", f"m{t}", f"m{u}")
            b.vstore2(t, x.tile(vd["pre_tile"], c))
            b.regs.release(pairs=(t, u))

        # Layers: one body, weights and parameters through pointer registers.
        xw, xp_ = b.xreg(), b.xreg()
        b.li(xw, self.layer_w[0])
        b.li(xp_, self.layer_p[0])
        qkv = self.big_view("enc.qkv", rows, 3 * h, "bf16")
        hq = self.big_view("enc.hq", rows, v.mlp, "fp8")
        wl = lambda n: self.lin[f"0.{n}"]
        with b.loop(v.layers, steps=[(xw, self.w_stride), (xp_, self.p_stride)]):
            b.prog.comment("encoder layer")
            lp0 = self.lp["ln1.w"].tile(0, 0)
            for off in range(0, self.p_bytes, 4096):
                b.set_x(1, (lp0 + off) // 4)
                b.add_imm(2, xp_, off)
                b.xval[2] = None
                b.set_x(3, min(4096, self.p_bytes - off))
                b.op(f"dma.load x1, x2, x3, ch{(off // 4096) % 6}")
            for ch in range(min(6, -(-self.p_bytes // 4096))):
                b.op(f"dma.wait ch{ch}")
            lk.layernorm_rows(x, self.lp["ln1.w"], self.lp["ln1.b"], xq, a["L.ln1"], v.eps)
            lk.linear_rows(wl("qkv"), (xw, self.w_off["qkv"]), xq, a["L.ln1"] + wl("qkv").a_w,
                           [Stream("qkv", qkv)], lk.epi_store("qkv"))
            lk.attention_rows(qkv, 0, qkv, HT, 2 * HT, v.heads, (cos, sin), smask, attnq,
                              a["L.q"], a["L.k"], a["L.v"], a["L.p"], a["L.attn"], scratch)
            lk.linear_rows(wl("o"), (xw, self.w_off["o"]), attnq, a["L.attn"] + wl("o").a_w,
                           [Stream("x", x), Stream("ls", self.lp["ls1"], per_row=False)],
                           lk.epi_residual_scale("x", "ls"))
            lk.layernorm_rows(x, self.lp["ln2.w"], self.lp["ln2.b"], xq, a["L.ln2"], v.eps)
            lk.linear_rows(wl("up"), (xw, self.w_off["up"]), xq, a["L.ln2"] + wl("up").a_w,
                           [Stream("hq", hq)], lk.epi_quant("hq", a["L.gelu"], pre=k.gelu))
            lk.linear_rows(wl("down"), (xw, self.w_off["down"]), hq, a["L.gelu"] + wl("down").a_w,
                           [Stream("x", x), Stream("ls", self.lp["ls2"], per_row=False)],
                           lk.epi_residual_scale("x", "ls"))
        b.xrelease(xw, xp_)

        # Final norm, QueryPool, projection.
        lk.layernorm_rows(x, self.st["norm.w"], self.st["norm.b"], xq, a["pool.kvin"], v.eps)
        kv = self.big_view("enc.kv", rows, 2 * h, "bf16")
        lin = self.lin["pool.kv"]
        lk.linear_rows(lin, lin.addr, xq, a["pool.kvin"] + lin.a_w, [Stream("kv", kv)], lk.epi_store("kv"))
        lk.attention_rows(self.qp, 0, kv, 0, HT, v.heads, None, pmask, self.pattn,
                          a["pool.q"], a["pool.k"], a["pool.v"], a["pool.p"], a["pool.attn"],
                          {n: Tensor(t.name, t.addr, v.queries if n == "qbuf" else t.rows, t.cols, t.kind)
                           for n, t in scratch.items()}, q_rows=v.queries)
        b.vsetm(v.queries)
        lin = self.lin["pool.out"]
        lk.linear_rows(lin, lin.addr, self.pattn, a["pool.attn"] + lin.a_w, [Stream("pooled", self.pooled)],
                       lk.epi_store("pooled"))
        lk.layernorm_rows(self.pooled, self.st["pool.ln.w"], self.st["pool.ln.b"], self.pm, a["pool.ln"], 1e-5)
        lin = self.lin["pool.up"]
        lk.linear_rows(lin, lin.addr, self.pm, a["pool.ln"] + lin.a_w, [Stream("ph", self.ph)],
                       lk.epi_quant("ph", a["pool.gelu"], pre=k.gelu))
        lin = self.lin["pool.down"]
        lk.linear_rows(lin, lin.addr, self.ph, a["pool.gelu"] + lin.a_w, [Stream("pooled", self.pooled)],
                       lk.epi_residual_add("pooled"))
        lk.quantize_rows(self.pooled, self.pm, a["proj_in"])
        lin = self.lin["proj"]
        lk.linear_rows(lin, lin.addr, self.pm, a["proj_in"] + lin.a_w, [Stream("proj", self.proj_out)],
                       lk.epi_store("proj"))
        out = self.outs[vi]
        t = b.regs.pair()
        b.vload2(t, self.proj_out.tile(0, 0))
        b.vstore2(t, out.tile(0, 0))
        b.regs.release(pairs=(t,))
        b.vsetm(32)


class PolicyProgram:
    def __init__(self, v: V.VisionDims, vplan: VisionPlan, d: M.DiTDims, dplan: Q.QuantPlan,
                 views, state, x0, steps=None, trace=True, dram_size: int = 64 << 20, schedule: bool = True):
        assert d.cond_dim == v.cond_dim and v.queries <= E_COLS
        self.v, self.d = v, d
        self.b = Builder(dram_size, schedule)
        self.enc = EncoderEmitter(self.b, v, vplan, views)
        self.b.vmem.reset(self.enc.mark)           # the DiT's tensors reuse the encoder's working set
        # Conditioning K tiles, in the order the DiT's adaLN weights are permuted to.
        tiles, perm = [], []
        for vi in range(len(v.views)):
            for qi in range(v.queries):
                tiles.append((self.enc.outs[vi], qi))
                base = v.state_dim + (vi * v.queries + qi) * v.token_dim
                perm += list(range(base, base + v.token_dim))
        perm += list(range(v.state_dim)) + [-1] * (32 - v.state_dim)
        state_tile = state.reshape(1, -1)
        self.dit = C.DiTProgram(d, dplan, x0, None, steps=steps, trace=trace, builder=self.b,
                                cond_tiles=tiles, cond_perm=perm, state=state_tile)

        self.out_pooled = [self.b.dram.reserve(f"out.{t.name}", t.nbytes) for t in self.enc.outs]

    def compile(self) -> C.Compiled:
        b = self.b
        b.begin()
        self.enc.emit()
        for t, addr in zip(self.enc.outs, self.out_pooled):      # the encoder's output, for checking
            b.dma.store_now(t.addr, addr, t.nbytes)
        c = self.dit.compile(standalone=False)
        c.manifest["outputs"]["pooled"] = self.out_pooled
        c.manifest["vision"] = self.v.to_dict()
        return c


def read_cond(c: C.Compiled, v: V.VisionDims, state: torch.Tensor, read) -> torch.Tensor:
    """The conditioning vector Atlas computed: state, then each view's pooled tokens (FP32)."""
    parts = [state.reshape(1, -1).to(torch.bfloat16).float()]
    for addr in c.manifest["outputs"]["pooled"]:
        tile = L.from_bf16_bytes(read(addr, L.BF16_TILE), 32, 32).float()
        parts.append(tile[:v.queries, :v.token_dim].reshape(1, -1))
    return torch.cat(parts, -1)

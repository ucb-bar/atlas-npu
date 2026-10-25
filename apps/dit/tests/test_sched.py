"""The list scheduler (sched.py): a scheduled program computes what the program order computes."""
import math

import torch

from ditflow import compiler as C
from ditflow import layout as L
from ditflow import model as M
from ditflow import quant as Q
from ditflow import sched as S
from ditflow.funcsim import FuncSim
from ditflow.harness import KernelRun
from ditflow.kernels import Linear
from ditflow.loopk import LoopKernels, Stream


class Dma(FuncSim):
    """The functional model with each DMA transfer's source read and destination write at issue
    ("i") or at its channel's DMA.WAIT ("w"). An instruction moved across a wait that fences its
    data then reads or keeps stale data."""
    mode = {"load": "ii", "store": "ii"}

    def __init__(self, *args):
        super().__init__(*args)
        self.inflight = {}

    def _step(self, pc):
        ins, s = self.instrs[pc], self.state
        name = type(ins).__name__
        if name in ("DMA_LOAD", "DMA_STORE"):
            kind = "load" if name == "DMA_LOAD" else "store"
            if self.mode[kind] == "ii":
                return super()._step(pc)
            rd, rs1, size = (s.read_xrf(r) for r in (ins.rd, ins.rs1, ins.rs2))
            if kind == "load":
                src = lambda: s.read_dram(s.dma_dram_address(rs1), size).clone()
                dst = lambda d: s.write_vmem(s.dma_vmem_byte_address(rd), 0, d)
            else:
                src = lambda: s.read_vmem(s.dma_vmem_byte_address(rs1), 0, size).clone()
                dst = lambda d: s.write_dram(s.dma_dram_address(rd), d)
            data = src() if self.mode[kind][0] == "i" else None
            self.inflight.setdefault(int(ins.ch), []).append((src, dst, data))
            self.executed += 1
            return None
        if name == "DMA_WAIT":
            for src, dst, data in self.inflight.pop(int(ins.ch), []):
                dst(src() if data is None else data)
        return super()._step(pc)


MODELS = [{"load": l, "store": s} for l, s in [("ii", "ii"), ("ww", "ww"), ("iw", "iw"), ("ww", "ii"), ("ii", "ww")]]


def outputs(mode, source, blobs, regions):
    sim = type("Sim", (Dma,), {"mode": mode})(source, blobs, 4 << 20)
    sim.run()
    return [sim.read_dram(addr, n) for addr, n in regions]


def check(source, blobs, regions):
    """Program order and the scheduled program agree bit for bit under every DMA timing model."""
    sched = S.schedule(source)
    assert sched != source and sorted(sched.splitlines()) == sorted(source.splitlines())
    gold = outputs(MODELS[0], source, blobs, regions)
    for mode in MODELS:
        assert outputs(mode, source, blobs, regions) == gold
        assert outputs(mode, sched, blobs, regions) == gold, mode


def test_dit_step():
    d = M.SMALL
    p = M.params(M.build(d, seed=0), d)
    plan = Q.QuantPlan(p, d, Q.calibrate(p, d))
    x0, cond = Q.random_inputs(d, 7)
    c = C.compile_dit(d, plan, x0, cond, steps=1, schedule=False)
    lines = [s.strip() for s in c.source.splitlines()]
    names = {S.parse(s)[0] for s in lines if s and not s.startswith("#") and not s.endswith(":")}
    assert names - set(S.OPS) <= S.CONTROL | {"CSRRW", "ECALL"}, "instructions the scheduler keeps in place"
    out = c.manifest["outputs"]
    check(c.source, c.blobs, [(out["x_hi"], out["x_bytes"]), (out["x_lo"], out["x_bytes"]),
                              (out["trace"], out["x_bytes"])])


DRAM = 0x8000_0000


def dram_program(body):
    return f"""    LI x20, 0
    LI x5, 0
    dma.config x5, ch0
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    LI x2, {DRAM:#x}
    LI x3, 1024
{body}
    LI x1, 0x100
    LI x2, {DRAM + 0x1000:#x}
    dma.store x2, x1, x3, ch3
    dma.wait ch3
    ECALL
"""


def test_dram_order():
    """DRAM has no address tracking, so dma.wait orders it: a load of DRAM that a store wrote (RAW),
    and a store to DRAM that a load reads (WAR), stay behind the other transfer's wait."""
    raw = """    LI x1, 0
    dma.store x2, x1, x3, ch1
    dma.wait ch1
    LI x1, 0x100
    dma.load x1, x2, x3, ch2
    dma.wait ch2"""
    war = """    LI x1, 0x100
    dma.load x1, x2, x3, ch1
    dma.wait ch1
    LI x1, 0
    dma.store x2, x1, x3, ch2
    dma.wait ch2"""
    g = torch.Generator().manual_seed(3)
    blobs = [("d", DRAM, bytes(torch.randint(0, 256, (1024,), generator=g, dtype=torch.uint8).tolist()))]
    regions = [(DRAM, 1024), (DRAM + 0x1000, 1024)]
    for body in (raw, war):
        source = dram_program(body)
        sched = S.schedule(source)
        gold = outputs(MODELS[0], source, blobs, regions)
        for mode in MODELS:
            assert outputs(mode, sched, blobs, regions) == gold, (mode, sched)


def test_delay_slot_with_blank_line():
    source = "    BEQ x0, x0, L1\n\n    vmatpop.bf16.acc.mxu1 m10, acc0\n    vexp.bf16 m12, m14\nL1:\n"
    assert S.schedule(source) == source


def test_loop_kernels():
    """Looped linear (streamed rows), attention and LayerNorm: blocks with pointer registers."""
    g = torch.Generator().manual_seed(5)
    rows, K, N = 128, 768, 192
    a_x, a_w = 2, Q.exponent_for(1 / math.sqrt(K))
    X = Q.to_fp8(torch.randn(rows, K, generator=g), a_x)
    W = Q.to_fp8((torch.rand(N, K, generator=g) * 2 - 1) / math.sqrt(K), a_w)
    kr = KernelRun()
    kr.b.schedule = False
    lk = LoopKernels(kr.b, kr.k)
    xq = kr.input_fp8("xq", X)
    lin = Linear("w", kr.b.dram.add("w", L.weight_blocks(W, torch.randn(N, generator=g))), L.tiles(N), L.tiles(K), True, a_w)
    x = kr.input_bf16("x", torch.randn(rows, N, generator=g))
    sc = kr.input_bf16("sc", L.replicated(torch.rand(N, generator=g)))
    lk.linear_rows(lin, lin.addr, xq, a_x + a_w, [Stream("x", x), Stream("sc", sc, per_row=False)],
                   lk.epi_residual_scale("x", "sc"))
    ang = torch.rand(rows, 32, generator=g) * 6
    cos, sin = kr.input_bf16("cos", ang.cos()), kr.input_bf16("sin", ang.sin())
    masks = kr.input_bf16("masks", torch.zeros(32, rows))
    attn = kr.tensor("attn", rows, 64, "fp8")
    scratch = {"qbuf": kr.tensor("qbuf", rows, 64, "fp8"), "kbuf": kr.tensor("kbuf", rows, 64, "fp8"),
               "vtbuf": kr.tensor("vtbuf", 64, rows, "fp8"), "pbuf": kr.tensor("pbuf", 32, rows, "fp8"),
               "sbuf": kr.tensor("sbuf", 32, rows, "bf16")}
    lk.attention_rows(x, 0, x, 2, 4, 1, (cos, sin), masks, attn, 0, 1, 1, -7, 1, scratch)
    ln = kr.tensor("ln", rows, N, "fp8")
    lk.layernorm_rows(x, sc, sc, ln, 1, 1e-5)
    for t in (x, attn, ln):
        kr.output(t)
    kr.run()
    check(kr.source, kr.b.dram.blobs, [(addr, t.nbytes) for t, addr in kr.outputs.values()])

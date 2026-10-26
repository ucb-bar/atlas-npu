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


def test_broadcast_forms():
    """.vx reads its x-register and .vr row 0 of T(vs2) at issue: a later write of either stays
    behind the read, and the scheduled program computes what the program order computes."""
    body = """    LI x1, 0
    dma.load x1, x2, x3, ch0
    LI x1, 0x100
    LI x2, {hi:#x}
    dma.load x1, x2, x3, ch1
    dma.wait ch0
    dma.wait ch1
    vload m0, 0(x20)
    vload m1, 8(x20)
    vload m2, 8(x20)
    vload m3, 0(x20)
    LI x4, 0x3fc01234
    vadd.bf16.vx m4, m0, x4
    LI x4, 0xc0a00000
    vsub.bf16.vx m8, m0, x4
    vmul.bf16.vr m6, m0, m2
    vload m2, 0(x20)
    vmaximum.bf16.vr m10, m2, m2
    vstore m4, 0(x20)
    vstore m5, 8(x20)
    vstore m6, 16(x20)
    vstore m7, 24(x20)
    vstore m8, 32(x20)
    vstore m9, 40(x20)
    vstore m10, 48(x20)
    vstore m11, 56(x20)
    LI x1, 0
    LI x2, {out:#x}
    LI x3, 4096
    dma.store x2, x1, x3, ch3
    LI x1, 0x400
    LI x2, {out2:#x}
    dma.store x2, x1, x3, ch4
    dma.wait ch3
    dma.wait ch4
    ECALL
"""
    g = torch.Generator().manual_seed(4)
    source = f"""    LI x20, 0
    LI x5, 0
    dma.config x5, ch0
    LI x2, {DRAM:#x}
    LI x3, 1024
{body.format(hi=DRAM + 0x400, out=DRAM + 0x1000, out2=DRAM + 0x2000)}"""
    blobs = [("d", DRAM, torch.randn(1024, generator=g).to(torch.bfloat16).view(torch.uint8).numpy().tobytes())]
    check(source, blobs, [(DRAM + 0x1000, 8192)])


def test_vgelu():
    """vgelu.bf16 reads T(vs1) for 64 cycles and writes T(vd) up to 75 cycles after issue: a later
    vload into its source stays behind its reads, a reader of its result behind its writes, and the
    scheduled program computes what the program order computes."""
    g = torch.Generator().manual_seed(5)
    source = f"""    LI x20, 0
    LI x5, 0
    dma.config x5, ch0
    LI x1, 0
    LI x2, {DRAM:#x}
    LI x3, 2048
    dma.load x1, x2, x3, ch0
    dma.wait ch0
    vload m0, 0(x20)
    vload m1, 8(x20)
    vgelu.bf16 m4, m0
    vtanh.bf16 m6, m0
    vload m0, 8(x20)
    vgelu.bf16 m8, m0
    vgelu.bf16 m4, m4
    vstore m4, 0(x20)
    vstore m5, 8(x20)
    vstore m6, 16(x20)
    vstore m7, 24(x20)
    vstore m8, 32(x20)
    vstore m9, 40(x20)
    LI x1, 0
    LI x2, {DRAM + 0x1000:#x}
    LI x3, 6144
    dma.store x2, x1, x3, ch3
    dma.wait ch3
    ECALL
"""
    blobs = [("d", DRAM, (torch.randn(1024, generator=g) * 4).to(torch.bfloat16).view(torch.uint8).numpy().tobytes())]
    check(source, blobs, [(DRAM + 0x1000, 6144)])


def test_instruction_after_a_branch_is_scheduled():
    """No delay slot: the instruction after a branch starts the next block, so it can move."""
    branch, pop, exp = "    BEQ x0, x0, L1\n", "    vmatpop.bf16.acc.mxu1 m10, acc0\n", "    vexp.bf16 m12, m14\n"
    assert S.schedule(branch + pop + exp + "L1:\n") == branch + exp + pop + "L1:\n"


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


def issue_cycles(lines, rows=32):
    """The cycle in which Machine issues each instruction of one block of M = rows, in program order."""
    code = [S.Ins(line, *S.parse(line), S.OPS[S.parse(line)[0]], rows=rows) for line in lines]
    m, t, out = S.Machine(), 0, []
    for ins in S.Block(code, {}).code:
        t = m.earliest(ins, t)
        m.issue(ins, t)
        out.append(t)
        t += ins.words
    return out


def test_machine_models_the_mxu1_fp32_transfers_and_scaled_pop():
    """The interlock's rules for the new mxu1 transfers (docs/fp8-mxu.md)."""
    push, pop = "vmatpush.acc.fp32.mxu1 acc0, m4", "vmatpop.fp32.acc.mxu1 m8, acc0"
    assert issue_cycles([push, "vmatpop.bf16.acc.mxu1 m12, acc0"]) == [0, 34]  # after row 0's second write
    assert issue_cycles([push, "vload m4, 0(x20)"]) == [0, 32]                 # m4, m5 read in T+1 .. T+32
    assert issue_cycles([push, "vload m6, 0(x20)"]) == [0, 64]                 # m6, m7 in T+33 .. T+64
    assert issue_cycles([pop, "vmatpush.acc.bf16.mxu1 acc0, m2"]) == [0, 33]   # after the first phase
    assert issue_cycles([pop, "vmatmul.mxu1 acc0, m1, w0"]) == [0, 33]
    assert issue_cycles([pop, "vexp.bf16 m20, m8"]) == [0, 32]
    assert issue_cycles([pop, "vexp.bf16 m20, m10"]) == [0, 64]
    assert issue_cycles([pop, "vmatpop.fp8.acc.mxu1 m12, acc1, e0"]) == [0, 64]  # both write ports
    assert issue_cycles(["seli e1, 3", "vmatpop.bf16.scaled.acc.mxu1 m8, acc0, e1",
                         "vmatpop.fp8.acc.mxu1 m12, acc1, e0"]) == [0, 1, 33]


def test_vsetm_ends_a_block_and_sets_the_costs_after_it():
    """No instruction moves across vsetm; the instructions after it cost what they cost at its M.
    After a label they cost what they cost at 32 rows, as a branch can enter with any M."""
    pop, exp, vsetm = "    vmatpop.bf16.acc.mxu1 m10, acc0\n", "    vexp.bf16 m12, m14\n", "    vsetm 3\n"
    assert S.schedule(pop + vsetm + exp) == pop + vsetm + exp
    assert S.schedule(vsetm + pop + exp) == vsetm + exp + pop
    xlu = "    vtrpose.xlu m20, m4\n"                     # 66 cycles at any M; vexp 2M + 2
    assert S.schedule(vsetm + exp + xlu) == vsetm + xlu + exp
    assert S.schedule(vsetm + "L1:\n" + exp + xlu) == vsetm + "L1:\n" + exp + xlu
    vpu = ["vadd.bf16 m4, m0, m2", "vexp.bf16 m6, m4"]
    assert issue_cycles(vpu) == [0, 66] and issue_cycles(vpu, rows=3) == [0, 8]          # 2M + 2
    mm = ["vmatmul.mxu1 acc0, m0, w0", "vmatmul.acc.mxu1 acc0, m1, w0", "vmatmul.acc.mxu1 acc1, m1, w0"]
    assert issue_cycles(mm) == [0, 32, 64] and issue_cycles(mm, rows=2) == [0, 4, 6]
    lsu = ["vload m4, 0(x20)", "vexp.bf16 m6, m4"]
    assert issue_cycles(lsu) == [0, 35] and issue_cycles(lsu, rows=5) == [0, 8]          # M + 3
    fp32 = ["vmatpush.acc.fp32.mxu1 acc0, m4", "vmatpop.bf16.acc.mxu1 m12, acc0"]
    assert issue_cycles(fp32, rows=3) == [0, 5]                                          # M + 2


def test_machine_follows_the_rtl():
    """Issue gaps measured on the RTL (Verilator, 2026-10-25), at 32 and at 3 rows."""
    def gap(a, b, rows=32):
        return issue_cycles([a, b], rows)[1]
    assert gap("lh x13, 0(x20)", "ADDI x14, x13, 0") == 4
    for a, reg, at32, at3 in [("vexp.bf16 m20, m10", 11, 64, 6), ("vadd.bf16 m20, m10, m12", 13, 64, 6),
                              ("vgelu.bf16 m20, m10", 11, 64, 6), ("vadd.bf16.vr m20, m10, m12", 11, 65, 7),
                              ("vadd.bf16.vr m20, m10, m12", 12, 1, 1), ("vredsum.bf16 m20, m10", 11, 128, 12),
                              ("vredsum.row.bf16 m20, m10", 11, 32, 3), ("vtrpose.xlu m20, m10", 10, 34, 34)]:
        load = f"vload m{reg}, 0(x20)"                  # the source is free
        assert (gap(a, load), gap(a, load, 3)) == (at32, at3), a
    assert gap("vexp.bf16 m20, m10", "vadd.bf16.vr m22, m12, m14") == 64
    mm, push = "vmatmul.mxu1 acc0, m4, w1", "vmatpush.weight.mxu1 w1, m6"
    assert (gap(push, mm), gap(push, mm, 3), gap(mm, push), gap(mm, push, 3)) == (32, 32, 32, 3)
    for push, at32, at3 in [("vmatpush.acc.fp32.mxu1 acc0, m12", 65, 7), ("vmatpush.acc.bf16.mxu1 acc0, m10", 33, 4)]:
        assert (gap(push, "vmatmul.acc.mxu1 acc0, m4, w0"), gap(push, "vmatmul.mxu1 acc0, m4, w0", 3)) == (at32, at3)


def test_fp32_spill_and_scaled_pop_schedule():
    """A block with the new mxu1 transfers computes the same after scheduling."""
    body = """    LI x1, 0
    LI x3, 4096
    dma.load x1, x2, x3, ch0
    dma.wait ch0
    vload m0, 0(x20)
    vload m1, 8(x20)
    vload m2, 16(x20)
    seli e1, 120
    vmatpush.weight.mxu1 w0, m1
    vmatmul.mxu1 acc0, m0, w0
    vmatpop.fp32.acc.mxu1 m8, acc0
    vmatpush.acc.fp32.mxu1 acc1, m8
    vmatmul.acc.mxu1 acc1, m0, w0
    vmatpop.bf16.scaled.acc.mxu1 m12, acc1, e1
    vexp.bf16 m14, m2
    vstore m8, 32(x20)
    vstore m9, 40(x20)
    vstore m10, 48(x20)
    vstore m11, 56(x20)
    vstore m12, 64(x20)
    vstore m13, 72(x20)
    vstore m14, 80(x20)
    vstore m15, 88(x20)
    LI x1, 1024
    LI x2, 0x80002000
    dma.store x2, x1, x3, ch1
    LI x1, 2048
    LI x2, 0x80003000
    dma.store x2, x1, x3, ch2
    dma.wait ch1
    dma.wait ch2"""
    g = torch.Generator().manual_seed(9)
    blobs = [("d", DRAM, bytes(torch.randint(0, 256, (4096,), generator=g, dtype=torch.uint8).tolist()))]
    check(dram_program(body), blobs, [(DRAM + 0x2000, 8192)])

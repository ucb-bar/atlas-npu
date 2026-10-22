"""Tile kernels for the DiT on Atlas: each emits straight-line code into a Builder.

Every matmul runs on mxu1, the inner-product-tree MXU that all three Atlas configurations have,
as FP8 x FP8 with BF16 accumulation. A Linear's output tile is
    acc = sum_k X_q(k) @ W_q(k)^T          (vmatmul.mxu1, then vmatmul.acc.mxu1 over K tiles)
    y   = acc * 2**(a_x + a_w) + bias      (vmatpop.bf16, vmul, vadd)
and an epilogue consumes y while it is still in registers. BF16 -> FP8 conversion goes through
an accumulator, as npu_model's kernels do: vmatpush.acc.bf16 reads the tile view vmatpop.bf16
and the VPU use, and vmatpop.fp8.acc returns FP8 in the row-major layout vmatmul reads.
"""
import math
from dataclasses import dataclass

from .asm import Builder, Tensor
from .layout import FP8_TILE, bf16_bits, pow2_bits

ONE, HALF, ZERO = bf16_bits(1.0), bf16_bits(0.5), 0
GELU_K1, GELU_K2 = bf16_bits(0.044715), bf16_bits(math.sqrt(2.0 / math.pi))
NEG_BIG = bf16_bits(-30000.0)


@dataclass
class Linear:
    """A Linear's weight blocks in DRAM (layout.weight_blocks)."""
    name: str
    addr: int
    nt: int
    kt: int
    has_bias: bool
    a_w: int

    @property
    def block_bytes(self) -> int:
        return self.kt * FP8_TILE + (2 * FP8_TILE if self.has_bias else 0)

    def block(self, n: int) -> int:
        return self.addr + n * self.block_bytes


class Kernels:
    def __init__(self, b: Builder, stage_bytes: int):
        self.b = b
        self.stage = [b.vmem.alloc("stage0", stage_bytes), b.vmem.alloc("stage1", stage_bytes)]
        self.stage_bytes = stage_bytes

    # -- small helpers -----------------------------------------------------------------------
    def quant_store(self, src_pair: int, acc: int, dst_addr: int | None, raw: int, keep: bool = False) -> int:
        """FP8(src * 2**(raw - 127)) through acc{acc}; vstore'd to dst_addr unless None.
        Returns the FP8 register if keep, else frees it."""
        b = self.b
        q = b.regs.one()
        b.v("vmatpush.acc.bf16.mxu1", f"acc{acc}", f"m{src_pair}")
        b.v("vmatpop.fp8.acc.mxu1", f"m{q}", f"acc{acc}", f"e{b.erf(raw)}")
        if dst_addr is not None:
            b.vstore(q, dst_addr)
        if keep:
            return q
        b.regs.release(q)
        return -1

    def const(self, bits: int) -> int:
        p = self.b.regs.pair()
        self.b.vli(p, bits)
        return p

    def quantize(self, src: Tensor, dst: Tensor, a: int) -> None:
        """dst = FP8(src / 2**a), tile by tile."""
        b = self.b
        assert (src.rt, src.ct) == (dst.rt, dst.ct) and dst.kind == "fp8"
        for i, (r, c) in enumerate((r, c) for r in range(src.rt) for c in range(src.ct)):
            p = b.regs.pair()
            b.vload2(p, src.tile(r, c))
            self.quant_store(p, i % 2, dst.tile(r, c), 127 - a)
            b.regs.release(pairs=(p,))

    # -- Linear --------------------------------------------------------------------------------
    def linear(self, lin: Linear, xq: Tensor, deq: int, epilogue, resident: bool | None = None) -> None:
        """For each output tile (r, n): y = acc * 2**deq + bias, then epilogue(y_pair, r, n)."""
        b, regs = self.b, self.b.regs
        RT, KT, NT = xq.rt, lin.kt, lin.nt
        assert xq.kind == "fp8" and xq.ct == KT and RT <= 2, (lin.name, xq)
        assert lin.block_bytes <= self.stage_bytes
        if resident is None:
            resident = RT * KT <= 24
        b.prog.comment(f"linear {lin.name}: {RT}x{KT} @ {KT}x{NT} tiles, deq 2^{deq}")
        xreg = {}
        if resident:
            for r in range(RT):
                for k in range(KT):
                    xreg[r, k] = regs.one()
                    b.vload(xreg[r, k], xq.tile(r, k))
        else:
            stream = [regs.one() for _ in range(4)]
        wreg = [regs.one(), regs.one()]
        deqt = self.const(pow2_bits(deq))
        bias = regs.pair() if lin.has_bias else None

        def queue(n):
            b.dma.load(f"{lin.name}#{n}", self.stage[n % 2], lin.block(n), lin.block_bytes)

        queue(0)
        cmds = -(-lin.block_bytes // 4096)
        per_k = -(-cmds // KT)
        for n in range(NT):
            b.dma.wait(f"{lin.name}#{n}")
            if n + 1 < NT:
                queue(n + 1)
            base = self.stage[n % 2]
            b.vload(wreg[0], base)
            for k in range(KT):
                slot = k % 2
                b.v("vmatpush.weight.mxu1", f"w{slot}", f"m{wreg[slot]}")
                if k + 1 < KT:
                    b.vload(wreg[(k + 1) % 2], base + (k + 1) * FP8_TILE)
                for r in range(RT):
                    if resident:
                        xm = xreg[r, k]
                    else:
                        xm = stream[(k * RT + r) % 4]
                        b.vload(xm, xq.tile(r, k))
                    b.v("vmatmul.acc.mxu1" if k else "vmatmul.mxu1", f"acc{r}", f"m{xm}", f"w{slot}")
                b.dma.pump(per_k)
            if bias is not None:
                b.vload2(bias, base + KT * FP8_TILE)
            for r in range(RT):
                o = regs.pair()
                b.v("vmatpop.bf16.acc.mxu1", f"m{o}", f"acc{r}")
                b.v("vmul.bf16", f"m{o}", f"m{o}", f"m{deqt}")
                if bias is not None:
                    b.v("vadd.bf16", f"m{o}", f"m{o}", f"m{bias}")
                epilogue(o, r, n)
                regs.release(pairs=(o,))
        regs.release(*(xreg.values() if resident else stream), *wreg, pairs=(deqt,) + ((bias,) if bias is not None else ()))

    # -- epilogues -----------------------------------------------------------------------------
    def epi_store(self, out: Tensor):
        def f(o, r, n):
            self.b.vstore2(o, out.tile(r, n))
        return f

    def epi_quant(self, out: Tensor, a: int, pre=None):
        """Optionally transform y in place (pre(o, r)), then store FP8(y / 2**a)."""
        def f(o, r, n):
            if pre is not None:
                pre(o, r)
            self.quant_store(o, r, out.tile(r, n), 127 - a)
        return f

    def epi_residual_gate(self, x: Tensor, gate: Tensor):
        """x += gate * y, with gate a replicated row of tiles."""
        def f(o, r, n):
            b = self.b
            g = b.regs.pair()
            b.vload2(g, gate.tile(0, n))
            b.v("vmul.bf16", f"m{o}", f"m{o}", f"m{g}")
            b.vload2(g, x.tile(r, n))
            b.v("vadd.bf16", f"m{o}", f"m{o}", f"m{g}")
            b.vstore2(o, x.tile(r, n))
            b.regs.release(pairs=(g,))
        return f

    def epi_add_store(self, add: Tensor, out: Tensor):
        def f(o, r, n):
            b = self.b
            g = b.regs.pair()
            b.vload2(g, add.tile(r, n))
            b.v("vadd.bf16", f"m{o}", f"m{o}", f"m{g}")
            b.vstore2(o, out.tile(r, n))
            b.regs.release(pairs=(g,))
        return f

    # -- pointwise functions on a BF16 pair, in place ------------------------------------------
    def gelu2(self, o: int) -> None:
        """o = 2 * gelu_tanh(o) = o * (1 + tanh(sqrt(2/pi) * (o + 0.044715 o^3)))."""
        b = self.b
        t, k = b.regs.pair(), b.regs.pair()
        b.v("vcube.bf16", f"m{t}", f"m{o}")
        b.vli(k, GELU_K1)
        b.v("vmul.bf16", f"m{t}", f"m{t}", f"m{k}")
        b.v("vadd.bf16", f"m{t}", f"m{t}", f"m{o}")
        b.vli(k, GELU_K2)
        b.v("vmul.bf16", f"m{t}", f"m{t}", f"m{k}")
        b.v("vtanh.bf16", f"m{t}", f"m{t}")
        b.vli(k, ONE)
        b.v("vadd.bf16", f"m{t}", f"m{t}", f"m{k}")
        b.v("vmul.bf16", f"m{o}", f"m{o}", f"m{t}")
        b.regs.release(pairs=(t, k))

    def scale(self, o: int, bits: int) -> None:
        k = self.const(bits)
        self.b.v("vmul.bf16", f"m{o}", f"m{o}", f"m{k}")
        self.b.regs.release(pairs=(k,))

    def silu(self, o: int) -> None:
        """o = o / (1 + exp(-o))."""
        b = self.b
        t, k = b.regs.pair(), b.regs.pair()
        b.vli(k, ZERO)
        b.v("vsub.bf16", f"m{t}", f"m{k}", f"m{o}")
        b.v("vexp.bf16", f"m{t}", f"m{t}")
        b.vli(k, ONE)
        b.v("vadd.bf16", f"m{t}", f"m{t}", f"m{k}")
        b.v("vrecip.bf16", f"m{t}", f"m{t}")
        b.v("vmul.bf16", f"m{o}", f"m{o}", f"m{t}")
        b.regs.release(pairs=(t, k))

    # -- LayerNorm (no affine, eps 1e-6), adaLN modulation, FP8 out ----------------------------
    def ln_modulate_quant(self, x: Tensor, scale1: Tensor, shift: Tensor, out: Tensor, a: int) -> None:
        """out = FP8((LN(x) * scale1 + shift) / 2**a); scale1 = 1 + scale and shift are replicated rows."""
        b, regs = self.b, self.b.regs
        HT = x.ct
        inv_n = bf16_bits(1.0 / x.cols)
        b.prog.comment(f"layernorm+modulate {x.name} -> {out.name}")
        for r in range(x.rt):
            mean, var, xj, t = regs.pair(), regs.pair(), regs.pair(), regs.pair()
            for j in range(HT):                                     # mean
                b.vload2(xj, x.tile(r, j))
                b.v("vredsum.row.bf16", f"m{mean if j == 0 else t}", f"m{xj}")
                if j:
                    b.v("vadd.bf16", f"m{mean}", f"m{mean}", f"m{t}")
            k = self.const(inv_n)
            b.v("vmul.bf16", f"m{mean}", f"m{mean}", f"m{k}")
            for j in range(HT):                                     # variance
                b.vload2(xj, x.tile(r, j))
                b.v("vsub.bf16", f"m{xj}", f"m{xj}", f"m{mean}")
                b.v("vsquare.bf16", f"m{xj}", f"m{xj}")
                b.v("vredsum.row.bf16", f"m{var if j == 0 else t}", f"m{xj}")
                if j:
                    b.v("vadd.bf16", f"m{var}", f"m{var}", f"m{t}")
            b.v("vmul.bf16", f"m{var}", f"m{var}", f"m{k}")
            b.vli(k, bf16_bits(1e-6))
            b.v("vadd.bf16", f"m{var}", f"m{var}", f"m{k}")
            b.v("vsqrt.bf16", f"m{var}", f"m{var}")
            b.v("vrecip.bf16", f"m{var}", f"m{var}")                # 1 / std
            for j in range(HT):                                     # normalize, modulate
                b.vload2(xj, x.tile(r, j))
                b.v("vsub.bf16", f"m{xj}", f"m{xj}", f"m{mean}")
                b.v("vmul.bf16", f"m{xj}", f"m{xj}", f"m{var}")
                b.vload2(t, scale1.tile(0, j))
                b.v("vmul.bf16", f"m{xj}", f"m{xj}", f"m{t}")
                b.vload2(t, shift.tile(0, j))
                b.v("vadd.bf16", f"m{xj}", f"m{xj}", f"m{t}")
                self.quant_store(xj, j % 2, out.tile(r, j), 127 - a)
            regs.release(pairs=(mean, var, xj, t, k))

    # -- RoPE self-attention, one head at a time -----------------------------------------------
    def attention(self, qkv: Tensor, rope_cos: Tensor, rope_sin: Tensor, kmask: Tensor | None,
                  out: Tensor, heads: int, a_q: int, a_k: int, a_v: int, a_p: int, a_out: int) -> None:
        """out = FP8(attention / 2**a_out). qkv holds q | k | v (each heads x 64 wide) per token row.
        a_q is the exponent of q / sqrt(64). kmask (replicated rows) masks the last key tile's padding."""
        b, regs = self.b, self.b.regs
        RT, HT = qkv.rt, qkv.ct // 3
        assert RT <= 2 and HT == 2 * heads, "head_dim must be 64 and the sequence at most 64"
        raw_q, raw_k, raw_v, raw_p = 127 - a_q - 3, 127 - a_k, 127 - a_v, 127 - a_p
        raw_o = 127 + a_p + a_v - a_out
        cos = [regs.pair() for _ in range(RT)]
        sin = [regs.pair() for _ in range(RT)]
        for r in range(RT):
            b.vload2(cos[r], rope_cos.tile(r, 0))
            b.vload2(sin[r], rope_sin.tile(r, 0))
        for h in range(heads):
            b.prog.comment(f"attention head {h}")
            Q, K, VT, P = {}, {}, {}, {}
            for r in range(RT):
                for dst, ct0, raw in ((Q, 0, raw_q), (K, HT, raw_k)):
                    a1, a2, t1, t2 = regs.pair(), regs.pair(), regs.pair(), regs.pair()
                    b.vload2(a1, qkv.tile(r, ct0 + 2 * h))
                    b.vload2(a2, qkv.tile(r, ct0 + 2 * h + 1))
                    b.v("vmul.bf16", f"m{t1}", f"m{a1}", f"m{cos[r]}")
                    b.v("vmul.bf16", f"m{t2}", f"m{a2}", f"m{sin[r]}")
                    b.v("vsub.bf16", f"m{t1}", f"m{t1}", f"m{t2}")     # x1 cos - x2 sin
                    b.v("vmul.bf16", f"m{t2}", f"m{a2}", f"m{cos[r]}")
                    b.v("vmul.bf16", f"m{a2}", f"m{a1}", f"m{sin[r]}")
                    b.v("vadd.bf16", f"m{t2}", f"m{t2}", f"m{a2}")     # x2 cos + x1 sin
                    dst[r, 0] = self.quant_store(t1, 0, None, raw, keep=True)
                    dst[r, 1] = self.quant_store(t2, 1, None, raw, keep=True)
                    regs.release(pairs=(a1, a2, t1, t2))
                for dd in range(2):
                    a1 = regs.pair()
                    b.vload2(a1, qkv.tile(r, 2 * HT + 2 * h + dd))
                    vq = self.quant_store(a1, dd, None, raw_v, keep=True)
                    VT[dd, r] = regs.one()
                    b.v("vtrpose.xlu", f"m{VT[dd, r]}", f"m{vq}")       # rows: head dims, cols: keys
                    regs.release(vq, pairs=(a1,))
            S = {}
            for rk in range(RT):                                         # scores, one key tile at a time
                b.v("vmatpush.weight.mxu1", "w0", f"m{K[rk, 0]}")
                b.v("vmatpush.weight.mxu1", "w1", f"m{K[rk, 1]}")
                for rq in range(RT):
                    b.v("vmatmul.mxu1", f"acc{rq}", f"m{Q[rq, 0]}", "w0")
                    b.v("vmatmul.acc.mxu1", f"acc{rq}", f"m{Q[rq, 1]}", "w1")
                for rq in range(RT):
                    S[rq, rk] = regs.pair()
                    b.v("vmatpop.bf16.acc.mxu1", f"m{S[rq, rk]}", f"acc{rq}")
            regs.release(*Q.values(), *K.values())
            deq = self.const(pow2_bits(a_q + a_k))
            mask = None
            if kmask is not None:
                mask = regs.pair()
                b.vload2(mask, kmask.tile(0, 0))
            for rq in range(RT):                                         # softmax over keys
                m, t = regs.pair(), regs.pair()
                for rk in range(RT):
                    b.v("vmul.bf16", f"m{S[rq, rk]}", f"m{S[rq, rk]}", f"m{deq}")
                    if mask is not None and rk == RT - 1:
                        b.v("vadd.bf16", f"m{S[rq, rk]}", f"m{S[rq, rk]}", f"m{mask}")
                    b.v("vredmax.row.bf16", f"m{m if rk == 0 else t}", f"m{S[rq, rk]}")
                    if rk:
                        b.v("vmaximum.bf16", f"m{m}", f"m{m}", f"m{t}")
                for rk in range(RT):
                    b.v("vsub.bf16", f"m{S[rq, rk]}", f"m{S[rq, rk]}", f"m{m}")
                    b.v("vexp.bf16", f"m{S[rq, rk]}", f"m{S[rq, rk]}")
                for rk in range(RT):                                     # m becomes the row sum
                    b.v("vredsum.row.bf16", f"m{m if rk == 0 else t}", f"m{S[rq, rk]}")
                    if rk:
                        b.v("vadd.bf16", f"m{m}", f"m{m}", f"m{t}")
                b.v("vrecip.bf16", f"m{m}", f"m{m}")
                for rk in range(RT):
                    b.v("vmul.bf16", f"m{S[rq, rk]}", f"m{S[rq, rk]}", f"m{m}")
                    P[rq, rk] = self.quant_store(S[rq, rk], rk % 2, None, raw_p, keep=True)
                    regs.release(pairs=(S[rq, rk],))
                regs.release(pairs=(m, t))
            regs.release(pairs=(deq,) + ((mask,) if mask is not None else ()))
            for dd in range(2):                                          # P @ V
                for rk in range(RT):
                    b.v("vmatpush.weight.mxu1", f"w{rk}", f"m{VT[dd, rk]}")
                for rq in range(RT):
                    for rk in range(RT):
                        b.v("vmatmul.acc.mxu1" if rk else "vmatmul.mxu1", f"acc{rq}", f"m{P[rq, rk]}", f"w{rk}")
                for rq in range(RT):
                    o = regs.one()
                    b.v("vmatpop.fp8.acc.mxu1", f"m{o}", f"acc{rq}", f"e{b.erf(raw_o)}")
                    b.vstore(o, out.tile(rq, 2 * h + dd))
                    regs.release(o)
            regs.release(*VT.values(), *P.values())
        regs.release(pairs=tuple(cos) + tuple(sin))

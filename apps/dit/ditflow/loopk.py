"""Looped tile kernels for long sequences (the vision encoder's 448 and 256 token rows).

The kernels in kernels.py unroll everything; at a few hundred tokens that would not fit the
128 KiB IMEM. These iterate over row tiles, output tiles and key tiles at run time, with tile
addresses relative to pointer registers (asm.Ptr). Arithmetic is the same as in kernels.py.
Row counts are padded to even numbers of tiles (linear_rows works on row pairs), unless they
are one tile.
"""
from dataclasses import dataclass

from .asm import Builder, Ptr, Tensor, SCRATCH, DMA_MAX
from .kernels import Kernels, Linear, NEG_BIG
from .layout import FP8_TILE


@dataclass
class Stream:
    """A tensor an epilogue reads or writes at the current output tile: (row, nt) if per_row,
    else (0, nt) (a replicated row, e.g. a gate)."""
    name: str
    t: Tensor
    per_row: bool = True


class LoopKernels:
    def __init__(self, b: Builder, k: Kernels):
        self.b, self.k = b, k

    def ptr(self, reg: int, addr: int) -> None:
        self.b.li(reg, addr // 4)

    # -- DMA of one weight block, in chunks interleaved with compute --------------------------
    def _block_chunks(self, stage: int, dram_reg: int, off: int, nbytes: int, chans: list[int]):
        """Emitters, one per <= 4 KiB command: stage (VMEM) <- DRAM x[dram_reg] + off."""
        b = self.b
        out = []
        for i, c0 in enumerate(range(0, nbytes, DMA_MAX)):
            size = min(DMA_MAX, nbytes - c0)
            ch = chans[i % len(chans)]

            def emit(c0=c0, size=size, ch=ch):
                b.set_x(1, (stage + c0) // 4)
                b.add_imm(2, dram_reg, off + c0)
                b.xval[2] = None
                b.set_x(3, size)
                b.op(f"dma.load x1, x2, x3, ch{ch}")
            out.append(emit)
        return out

    def _wait(self, chans: list[int]) -> None:
        for ch in chans:
            self.b.op(f"dma.wait ch{ch}")

    # -- Linear over row pairs ----------------------------------------------------------------
    def linear_rows(self, lin: Linear, wbase, xq: Tensor, deq: int, streams: list[Stream], epilogue) -> None:
        """For every output tile (r, n): y = acc * 2**deq + bias, then epilogue(y_pair, r, ptrs) with
        ptrs[name] the Ptr of tile (r, n) (or (0, n)) of each stream. wbase: DRAM address of the
        first weight block, an int or (reg, offset). xq must have one or an even number of row
        tiles."""
        b, k, regs = self.b, self.k, self.b.regs
        RT, KT, NT = xq.rt, lin.kt, lin.nt
        R = min(RT, 2)                                  # row tiles per pass
        assert RT % R == 0 and xq.ct == KT, (lin.name, RT, xq.ct, KT)
        BB = lin.block_bytes
        assert BB <= k.stage_bytes
        resident = R * KT <= 24
        b.prog.comment(f"linear_rows {lin.name}: {RT}x{KT} @ {KT}x{NT}, deq 2^{deq}")
        xrow, xd = b.xreg(), b.xreg()
        srow = {s.name: b.xreg() for s in streams if s.per_row}
        scol = {s.name: b.xreg() for s in streams}
        self.ptr(xrow, xq.tile(0, 0))
        for s in streams:
            if s.per_row:
                self.ptr(srow[s.name], s.t.tile(0, 0))
        wreg = [regs.one(), regs.one()]
        bias = regs.pair() if lin.has_bias else None
        xr = [regs.one() for _ in range(R * KT)] if resident else [regs.one() for _ in range(4)]
        chans = [[0, 1, 2], [3, 4, 5]]
        row_steps = [(xrow, R * KT * FP8_TILE // 4)] + \
                    [(srow[s.name], R * s.t.ct * s.t.tile_bytes // 4) for s in streams if s.per_row]

        def nt_body(par, prefetch_off):
            """Output tile nt = (2 * pass + par): weights in stage[par]; prefetch the block at
            x[xd] + prefetch_off into stage[1 - par]."""
            stage = k.stage[par]
            self._wait(chans[par])
            pending = self._block_chunks(k.stage[1 - par], xd, prefetch_off, BB, chans[1 - par]) \
                if prefetch_off is not None else []
            per_k = -(-len(pending) // KT) if pending else 0
            b.vload_w(wreg[0], stage)
            for kk in range(KT):
                slot = kk % 2
                b.v("vmatpush.weight.mxu1", f"w{slot}", f"m{wreg[slot]}")
                if kk + 1 < KT:
                    b.vload_w(wreg[(kk + 1) % 2], stage + (kk + 1) * FP8_TILE)
                for r in range(R):
                    if resident:
                        xm = xr[r * KT + kk]
                    else:
                        xm = xr[(kk * R + r) % 4]
                        b.vload(xm, Ptr(xrow, (r * KT + kk) * FP8_TILE))
                    b.v("vmatmul.acc.mxu1" if kk else "vmatmul.mxu1", f"acc{r}", f"m{xm}", f"w{slot}")
                for _ in range(per_k):
                    if pending:
                        pending.pop(0)()
            while pending:
                pending.pop(0)()
            if bias is not None:
                b.vload2_row(bias, stage + KT * FP8_TILE)
            for r in range(R):
                o = regs.pair()
                b.v("vmatpop.bf16.scaled.acc.mxu1", f"m{o}", f"acc{r}", f"e{b.erf(127 + deq)}")
                if bias is not None:
                    b.v("vadd.bf16.vr", f"m{o}", f"m{o}", f"m{bias}")
                ptrs = {s.name: Ptr(scol[s.name], (r * s.t.ct * s.t.tile_bytes if s.per_row else 0)
                                    + par * s.t.tile_bytes) for s in streams}
                epilogue(o, r, ptrs)
                regs.release(pairs=(o,))

        with b.loop(RT // R, steps=row_steps):
            if resident:
                for r in range(R):
                    for kk in range(KT):
                        b.vload(xr[r * KT + kk], Ptr(xrow, (r * KT + kk) * FP8_TILE))
            if isinstance(wbase, int):
                b.li(xd, wbase)
            else:
                b.add_imm(xd, wbase[0], wbase[1])
            for s in streams:
                if s.per_row:
                    b.op(f"ADDI x{scol[s.name]}, x{srow[s.name]}, 0")
                else:
                    self.ptr(scol[s.name], s.t.tile(0, 0))
            for emit in self._block_chunks(k.stage[0], xd, 0, BB, chans[0]):
                emit()
            pairs, tail = NT // 2, NT % 2
            if pairs:
                col_steps = [(xd, 2 * BB)] + [(scol[s.name], 2 * s.t.tile_bytes // 4) for s in streams]
                with b.loop(pairs, steps=col_steps):
                    nt_body(0, BB)
                    nt_body(1, 2 * BB if (tail or pairs > 1) else None)
            if tail:
                nt_body(0, None)
            elif pairs > 1:
                self._wait(chans[0])            # the last pass prefetched one block past the end
        regs.release(*wreg, *xr, pairs=(bias,) if bias is not None else ())
        b.xrelease(xrow, xd, *srow.values(), *scol.values())

    # -- epilogues -------------------------------------------------------------------------
    def epi_store(self, name: str):
        def f(o, r, ptrs):
            self.b.vstore2(o, ptrs[name])
        return f

    def epi_quant(self, name: str, a: int, pre=None):
        def f(o, r, ptrs):
            if pre is not None:
                pre(o)
            self.k.quant_store(o, r, ptrs[name], 127 - a)
        return f

    def epi_residual_scale(self, x: str, scale: str):
        """x += scale * y (scale a row, row 0 of its tiles, e.g. LayerScale)."""
        def f(o, r, ptrs):
            b = self.b
            g = b.regs.pair()
            b.vload2_row(g, ptrs[scale])
            b.v("vmul.bf16.vr", f"m{o}", f"m{o}", f"m{g}")
            b.vload2(g, ptrs[x])
            b.v("vadd.bf16", f"m{o}", f"m{o}", f"m{g}")
            b.vstore2(o, ptrs[x])
            b.regs.release(pairs=(g,))
        return f

    def epi_residual_add(self, x: str):
        """x += y."""
        def f(o, r, ptrs):
            b = self.b
            g = b.regs.pair()
            b.vload2(g, ptrs[x])
            b.v("vadd.bf16", f"m{o}", f"m{o}", f"m{g}")
            b.vstore2(o, ptrs[x])
            b.regs.release(pairs=(g,))
        return f

    # -- row-wise kernels ----------------------------------------------------------------------
    def quantize_rows(self, src: Tensor, dst: Tensor, a: int) -> None:
        b = self.b
        assert (src.rt, src.ct) == (dst.rt, dst.ct)
        ps, pd = b.xreg(), b.xreg()
        self.ptr(ps, src.tile(0, 0))
        self.ptr(pd, dst.tile(0, 0))
        with b.loop(src.rt, steps=[(ps, src.ct * src.tile_bytes // 4), (pd, dst.ct * dst.tile_bytes // 4)]):
            for c in range(src.ct):
                p = b.regs.pair()
                b.vload2(p, Ptr(ps, c * src.tile_bytes))
                self.k.quant_store(p, c % 2, Ptr(pd, c * dst.tile_bytes), 127 - a)
                b.regs.release(pairs=(p,))
        b.xrelease(ps, pd)

    def layernorm_rows(self, x: Tensor, w: Tensor, bias: Tensor, out: Tensor, a: int, eps: float) -> None:
        """out = FP8((LN(x) * w + bias) / 2**a) per row tile; w, bias rows (row 0 of their tiles)."""
        b = self.b
        px, po = b.xreg(), b.xreg()
        self.ptr(px, x.tile(0, 0))
        self.ptr(po, out.tile(0, 0))
        b.prog.comment(f"layernorm_rows {x.name} -> {out.name}")
        with b.loop(x.rt, steps=[(px, x.ct * x.tile_bytes // 4), (po, out.ct * out.tile_bytes // 4)]):
            self.k.ln_row(lambda j: Ptr(px, j * x.tile_bytes), lambda j: Ptr(po, j * out.tile_bytes),
                          x.ct, x.cols, w, bias, a, eps)
        b.xrelease(px, po)

    # -- attention -------------------------------------------------------------------------------
    def attention_rows(self, q: Tensor, q_ct0: int, kv: Tensor, k_ct0: int, v_ct0: int, heads: int,
                       rope: tuple[Tensor, Tensor] | None, masks: Tensor, out: Tensor,
                       a_q: int, a_k: int, a_v: int, a_p: int, a_out: int, scratch: dict,
                       q_rows: int | None = None) -> None:
        """Multi-head attention with head_dim 64: out[:, 64h:64h+64] = FP8(softmax(q k^T / 8) v / 2**a_out).
        q rows and kv rows may differ (cross-attention): the query side runs at M = q_rows (default:
        the caller's M). K and V become a weight and V^T, which read 32 rows, so they are computed at
        32 rows: kv's padding rows must give finite K and V, and K = 0 under a padding key's mask
        (rope's padding rows 0). rope (cos, sin per kv/q row, applied to q and k when q is kv) or
        None. masks: one BF16 tile per key tile (row 0), added to
        the scores. scratch: FP8 tensors qbuf [q rows x 64], kbuf [kv rows x 64], vtbuf [64 x kv
        rows] (V^T), pbuf [32 x kv rows], and BF16 sbuf [32 x kv rows]."""
        b, k, regs = self.b, self.k, self.b.regs
        RQ, RK = q.rt, kv.rt
        qbuf, kbuf, vtbuf, pbuf, sbuf = (scratch[n] for n in ("qbuf", "kbuf", "vtbuf", "pbuf", "sbuf"))
        raw_q, raw_k, raw_v, raw_p = 127 - a_q - 3, 127 - a_k, 127 - a_v, 127 - a_p
        raw_o = 127 + a_p + a_v - a_out
        TB, FT = 2048, FP8_TILE
        rows = b.rows
        for h in range(heads):
            b.prog.comment(f"attention_rows head {h}")
            # ---- K and V at 32 rows, then Q, for every row tile ----
            b.vsetm(32)
            self._prep_rows(kv, k_ct0 + 2 * h, kbuf, raw_k, rope)
            self._prep_vt(kv, v_ct0 + 2 * h, vtbuf, raw_v)
            b.vsetm(q_rows or rows)
            self._prep_rows(q, q_ct0 + 2 * h, qbuf, raw_q, rope if q is kv else None)
            # ---- one query tile at a time ----
            pq, po = b.xreg(), b.xreg()
            self.ptr(pq, qbuf.tile(0, 0))
            self.ptr(po, out.tile(0, 2 * h))
            with b.loop(RQ, steps=[(pq, 2 * FT // 4), (po, out.ct * FT // 4)]):
                Q0, Q1 = regs.one(), regs.one()
                b.vload(Q0, Ptr(pq, 0))
                b.vload(Q1, Ptr(pq, FT))
                m, l, t, s = regs.pair(), regs.pair(), regs.pair(), regs.pair()
                b.vli(m, NEG_BIG)
                b.vli(l, 0)
                pk, ps, pm = b.xreg(), b.xreg(), b.xreg()
                self.ptr(pk, kbuf.tile(0, 0))
                self.ptr(ps, sbuf.tile(0, 0))
                self.ptr(pm, masks.tile(0, 0))
                K0, K1 = regs.one(), regs.one()
                with b.loop(RK, steps=[(pk, 2 * FT // 4), (ps, TB // 4), (pm, TB // 4)]):   # scores, row max
                    b.vload_w(K0, Ptr(pk, 0))
                    b.vload_w(K1, Ptr(pk, FT))
                    b.v("vmatpush.weight.mxu1", "w0", f"m{K0}")
                    b.v("vmatpush.weight.mxu1", "w1", f"m{K1}")
                    b.v("vmatmul.mxu1", "acc0", f"m{Q0}", "w0")
                    b.v("vmatmul.acc.mxu1", "acc0", f"m{Q1}", "w1")
                    b.v("vmatpop.bf16.scaled.acc.mxu1", f"m{s}", "acc0", f"e{b.erf(127 + a_q + a_k)}")
                    b.vload2_row(t, Ptr(pm, 0))
                    b.v("vadd.bf16.vr", f"m{s}", f"m{s}", f"m{t}")
                    b.v("vredmax.row.bf16", f"m{t}", f"m{s}")
                    b.v("vmaximum.bf16", f"m{m}", f"m{m}", f"m{t}")
                    b.vstore2(s, Ptr(ps, 0))
                regs.release(K0, K1, Q0, Q1)
                self.ptr(ps, sbuf.tile(0, 0))
                with b.loop(RK, steps=[(ps, TB // 4)]):                                     # exp, row sum
                    b.vload2(s, Ptr(ps, 0))
                    b.v("vsub.bf16", f"m{s}", f"m{s}", f"m{m}")
                    b.v("vexp.bf16", f"m{s}", f"m{s}")
                    b.v("vredsum.row.bf16", f"m{t}", f"m{s}")
                    b.v("vadd.bf16", f"m{l}", f"m{l}", f"m{t}")
                    b.vstore2(s, Ptr(ps, 0))
                b.v("vrecip.bf16", f"m{l}", f"m{l}")
                pv, pp = b.xreg(), b.xreg()
                P, V = regs.one(), regs.one()
                for dd in range(2):                                                         # P @ V, per head-dim tile
                    b.vli(t, 0)
                    b.v("vmatpush.acc.bf16.mxu1", "acc0", f"m{t}")
                    self.ptr(ps, sbuf.tile(0, 0))
                    self.ptr(pv, vtbuf.tile(dd, 0))
                    self.ptr(pp, pbuf.tile(0, 0))
                    with b.loop(RK, steps=[(ps, TB // 4), (pv, FT // 4), (pp, FT // 4)]):
                        if dd == 0:
                            b.vload2(s, Ptr(ps, 0))
                            b.v("vmul.bf16", f"m{s}", f"m{s}", f"m{l}")
                            b.v("vmatpush.acc.bf16.mxu1", "acc1", f"m{s}")
                            b.v("vmatpop.fp8.acc.mxu1", f"m{P}", "acc1", f"e{b.erf(raw_p)}")
                            b.vstore(P, Ptr(pp, 0))
                        else:
                            b.vload(P, Ptr(pp, 0))
                        b.vload_w(V, Ptr(pv, 0))
                        b.v("vmatpush.weight.mxu1", "w0", f"m{V}")
                        b.v("vmatmul.acc.mxu1", "acc0", f"m{P}", "w0")
                    o = regs.one()
                    b.v("vmatpop.fp8.acc.mxu1", f"m{o}", "acc0", f"e{b.erf(raw_o)}")
                    b.vstore(o, Ptr(po, dd * FT))
                    regs.release(o)
                regs.release(P, V, pairs=(m, l, t, s))
                b.xrelease(pk, ps, pm, pv, pp)
            b.xrelease(pq, po)
            b.vsetm(rows)

    def _prep_rows(self, src: Tensor, ct0: int, dst: Tensor, raw: int, rope) -> None:
        """dst row r = FP8 of src tiles (r, ct0), (r, ct0 + 1), rotated by rope when given."""
        b, k, regs = self.b, self.k, self.b.regs
        p_src, p_dst = b.xreg(), b.xreg()
        self.ptr(p_src, src.tile(0, ct0))
        self.ptr(p_dst, dst.tile(0, 0))
        steps = [(p_src, src.ct * src.tile_bytes // 4), (p_dst, 2 * FP8_TILE // 4)]
        if rope is not None:
            p_c, p_s = b.xreg(), b.xreg()
            self.ptr(p_c, rope[0].tile(0, 0))
            self.ptr(p_s, rope[1].tile(0, 0))
            steps += [(p_c, rope[0].tile_bytes // 4), (p_s, rope[1].tile_bytes // 4)]
        with b.loop(src.rt, steps=steps):
            a1, a2 = regs.pair(), regs.pair()
            b.vload2(a1, Ptr(p_src, 0))
            b.vload2(a2, Ptr(p_src, src.tile_bytes))
            if rope is not None:
                c, s, t1, t2 = regs.pair(), regs.pair(), regs.pair(), regs.pair()
                b.vload2(c, Ptr(p_c, 0))
                b.vload2(s, Ptr(p_s, 0))
                b.v("vmul.bf16", f"m{t1}", f"m{a1}", f"m{c}")
                b.v("vmul.bf16", f"m{t2}", f"m{a2}", f"m{s}")
                b.v("vsub.bf16", f"m{t1}", f"m{t1}", f"m{t2}")
                b.v("vmul.bf16", f"m{t2}", f"m{a2}", f"m{c}")
                b.v("vmul.bf16", f"m{a2}", f"m{a1}", f"m{s}")
                b.v("vadd.bf16", f"m{a2}", f"m{t2}", f"m{a2}")
                k.quant_store(t1, 0, Ptr(p_dst, 0), raw)
                k.quant_store(a2, 1, Ptr(p_dst, FP8_TILE), raw)
                regs.release(pairs=(c, s, t1, t2))
            else:
                k.quant_store(a1, 0, Ptr(p_dst, 0), raw)
                k.quant_store(a2, 1, Ptr(p_dst, FP8_TILE), raw)
            regs.release(pairs=(a1, a2))
        b.xrelease(p_src, p_dst, *((p_c, p_s) if rope is not None else ()))

    def _prep_vt(self, src: Tensor, ct0: int, vtbuf: Tensor, raw: int) -> None:
        """vtbuf tile (dd, r) = transpose of FP8 src tile (r, ct0 + dd): rows head dims, columns keys."""
        b, k, regs = self.b, self.k, self.b.regs
        p_src, p0, p1 = b.xreg(), b.xreg(), b.xreg()
        self.ptr(p_src, src.tile(0, ct0))
        self.ptr(p0, vtbuf.tile(0, 0))
        self.ptr(p1, vtbuf.tile(1, 0))
        with b.loop(src.rt, steps=[(p_src, src.ct * src.tile_bytes // 4), (p0, FP8_TILE // 4), (p1, FP8_TILE // 4)]):
            for dd, pdst in ((0, p0), (1, p1)):
                a1 = regs.pair()
                b.vload2(a1, Ptr(p_src, dd * src.tile_bytes))
                vq = k.quant_store(a1, dd, None, raw, keep=True)
                vt = regs.one()
                b.v("vtrpose.xlu", f"m{vt}", f"m{vq}")
                b.vstore(vt, Ptr(pdst, 0))
                regs.release(vq, vt, pairs=(a1,))
        b.xrelease(p_src, p0, p1)

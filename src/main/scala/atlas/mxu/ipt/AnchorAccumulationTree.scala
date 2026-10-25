package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.InnerProductTreeParams

/** An E4M3 x E4M3 product, exact and not normalized. */
class E4M3Product extends Bundle {
  val neg = Bool()
  val sig = UInt(8.W)  // Sa * Sb; 0 for a zero product
  val exp = UInt(5.W)  // max(Ea, 1) + max(Eb, 1) = ea + eb + 14
}

/** One compute lane in the inner-product-tree MXU.
  *
  * Computes d = C + a . w over vecLen E4M3 pairs with an FP32 C, as one NVIDIA Blackwell FP8
  * MMA step (docs/fp8-mxu.md; npu_model configs/fp8_mma.py states the same rule):
  *   - Products are exact and not normalized: P = Sa * Sb at exponent ea + eb, where an E4M3
  *     code is +-S/8 x 2^e with S = (E != 0):m and e = max(E, 1) - 7. Subnormals stay.
  *   - emax is the largest exponent of C and of the nonzero products.
  *   - Each nonzero product and C is cut toward zero at 2^(emax - 25). The terms are summed
  *     exactly, and the sum is cut toward zero to FP32. A zero sum is +0.
  *   - Without nonzero products, d = C, and -0 gives +0. A NaN input or a NaN C gives
  *     0x7FFFFFFF; else an infinite C gives C.
  *
  * Stages: 0 decode and multiply, 1 emax, 2 align, 3 sum, 4 normalize into the output register.
  * A cut c in p.pipelineCuts adds a register after stage c.
  *
  * @param p  IPT geometry and pipeline parameters for this lane.
  */
class AnchorAccumulationTree(p: InnerProductTreeParams) extends Module {

  val io = IO(new Bundle {
    val act        = Input(Vec(p.vecLen, UInt(8.W)))
    val weightBuf0 = Input(Vec(p.vecLen, UInt(8.W)))
    val weightBuf1 = Input(Vec(p.vecLen, UInt(8.W)))
    val psum       = Input(UInt(32.W))  // C
    val bufReadSel = Input(Bool())
    val out        = Output(UInt(32.W))
  })

  private def optPipe[T <: Data](x: T, cut: Int): T = if (p.pipelineCuts.contains(cut)) RegNext(x) else x

  private def treeReduce[T](xs: Seq[T])(op: (T, T) => T): T = {
    require(xs.nonEmpty)
    if (xs.length == 1) xs.head
    else { val next = xs.grouped(2).map {
      case Seq(a, b) => op(a, b)
      case Seq(a) => a }.toSeq
      treeReduce(next)(op)
    }
  }

  private val N = p.vecLen
  private val W = p.sumWidth

  /** C = +-sig x 2^(exp - 150): biased exponent at least 1, 24-bit significand. */
  private def cExp(c: UInt): UInt = c(30, 23) | (c(30, 23) === 0.U)
  private def cSig(c: UInt): UInt = Cat(c(30, 23) =/= 0.U, c(22, 0))

  // ── S0: Decode and multiply ───────────────────────────────────────
  val weight = Mux(io.bufReadSel, io.weightBuf1, io.weightBuf0)

  val prod_s0 = Wire(Vec(N, new E4M3Product))
  for (k <- 0 until N) {
    val (a, b) = (io.act(k), weight(k))
    prod_s0(k).neg := a(7) ^ b(7)
    prod_s0(k).sig := Cat(a(6, 3).orR, a(2, 0)) * Cat(b(6, 3).orR, b(2, 0))
    prod_s0(k).exp := (a(6, 3) | (a(6, 3) === 0.U)) +& (b(6, 3) | (b(6, 3) === 0.U))
  }
  val nonzero_s0 = VecInit((0 until N).map(k => io.act(k)(6, 0).orR && weight(k)(6, 0).orR))
  val nan_s0 = (0 until N).map(k => io.act(k)(6, 0).andR || weight(k)(6, 0).andR).reduce(_ || _) ||
               (io.psum(30, 23).andR && io.psum(22, 0).orR)

  val prod_s1    = optPipe(prod_s0, 0)
  val nonzero_s1 = optPipe(nonzero_s0, 0)
  val nan_s1     = optPipe(nan_s0, 0)
  val psum_s1    = optPipe(io.psum, 0)

  // ── S1: emax (biased); a zero product does not set it ─────────────
  val prodExpMax = treeReduce((0 until N).map(k => Mux(nonzero_s1(k), prod_s1(k).exp, 0.U))) {
    (a: UInt, b: UInt) => Mux(a > b, a, b)
  }
  val prodEmax = prodExpMax +& 113.U  // ea + eb + 127
  val emax     = Mux(prodEmax > cExp(psum_s1), prodEmax, cExp(psum_s1))
  val anyProduct = nonzero_s1.asUInt.orR

  val prod_s2       = optPipe(prod_s1, 1)
  val emax_s2       = optPipe(emax, 1)
  val anyProduct_s2 = optPipe(anyProduct, 1)
  val nan_s2        = optPipe(nan_s1, 1)
  val psum_s2       = optPipe(psum_s1, 1)

  // ── S2: Align: t = sign x floor(|x| / 2^(emax - 25)) ──────────────
  // A product's P sits at bits 26..19 and C's significand at bits 25..2 when its exponent is
  // emax; a distance of 32 or more leaves 0, as does 27 to 31 by itself.
  private def term(neg: Bool, mag: UInt, dist: UInt): SInt = {
    val shifted = Mux(dist >= 32.U, 0.U, mag >> dist(4, 0)).pad(W).asSInt
    Mux(neg, -shifted, shifted)
  }
  val prodInt = VecInit((0 until N).map { k =>
    term(prod_s2(k).neg, Cat(prod_s2(k).sig, 0.U(19.W)), emax_s2 -& (prod_s2(k).exp +& 113.U))
  })
  val psumInt = term(psum_s2(31), Cat(cSig(psum_s2), 0.U(2.W)), emax_s2 - cExp(psum_s2))

  val prodInt_s3    = optPipe(prodInt, 2)
  val psumInt_s3    = optPipe(psumInt, 2)
  val emax_s3       = optPipe(emax_s2, 2)
  val anyProduct_s3 = optPipe(anyProduct_s2, 2)
  val nan_s3        = optPipe(nan_s2, 2)
  val psum_s3       = optPipe(psum_s2, 2)

  // ── S3: Exact sum (W bits hold it, so the tree's wrap-around is exact) ─
  val total = treeReduce(prodInt_s3.toSeq :+ psumInt_s3)(_ + _)

  val total_s4      = optPipe(total, 3)
  val emax_s4       = optPipe(emax_s3, 3)
  val anyProduct_s4 = optPipe(anyProduct_s3, 3)
  val nan_s4        = optPipe(nan_s3, 3)
  val psum_s4       = optPipe(psum_s3, 3)

  // ── S4: Cut toward zero to FP32 ───────────────────────────────────
  // Bit i of |total| is 2^(i + emax - 25 - 127) for biased emax; no rounding, and the
  // contract leaves no subnormal or overflow result here.
  val neg  = total_s4 < 0.S
  val mag  = Mux(neg, -total_s4, total_s4).asUInt(W - 2, 0)
  val lzc  = PriorityEncoder(Reverse(mag))
  val norm = (mag << lzc)(W - 2, 0)
  val exp  = (emax_s4 +& (W - 27).U) - lzc
  val sum  = Mux(total_s4 === 0.S, 0.U, Cat(neg, exp(7, 0), norm(W - 3, W - 25)))

  val c = psum_s4
  val result = Mux(nan_s4, "h7fffffff".U(32.W),
               Mux(c(30, 23).andR, c,
               Mux(!anyProduct_s4, Mux(c === "h80000000".U, 0.U, c), sum)))

  val outReg = RegNext(result, 0.U(32.W))
  io.out := outReg
}

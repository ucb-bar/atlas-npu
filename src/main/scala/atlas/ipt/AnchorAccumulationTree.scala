/*
Anchor-based accumulation lane for E4M3 inputs.

E4M3 x E4M3 -> 13-bit lossless product (E4M3Mul)
followed by a narrow aligned-int converter  (E4M3ProdToAlignedInt)

Pipeline cuts:
  cut 0 – after S0 (multiply + addend exp)
  cut 1 – after S1 (exponent extraction + anchor)
  cut 2 – after S2 (float-to-aligned-int)
  cut 3 – after S3 (integer reduction)
*/

package atlas.ipt

import chisel3._
import chisel3.util._
import sp26FPUnits.hardfloat._
import sp26FPUnits.{AtlasFPType, E4M3ProdFmt, E4M3Mul}
import atlas.common.InnerProductTreeParams

class AnchorAccumulationTree(p: InnerProductTreeParams) extends Module {

  // IO
  // Activations and weights arrive as raw 8-bit IEEE E4M3.
  val io = IO(new Bundle {
    val act        = Input(Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W)))
    val weightBuf0 = Input(Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W)))
    val weightBuf1 = Input(Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W)))
    val bias       = Input(UInt(p.biasFmt.ieeeWidth.W))
    val psum       = Input(UInt(p.psumFmt.ieeeWidth.W))
    val bufReadSel = Input(Bool())
    val addendSel  = Input(AddendSel())
    val out        = Output(UInt(p.outputFmt.ieeeWidth.W))
  })

  // pipeline helpers
  private def optPipe[T <: Data](x: T, cut: Int): T =
    if (p.pipelineCuts.contains(cut)) RegNext(x) else x

  private val sentinelVal = -(1 << (p.expWorkWidth - 2))
  private def sentinel: SInt = sentinelVal.S(p.expWorkWidth.W)

  private def treeReduce[T](xs: Seq[T])(op: (T, T) => T): T = {
    require(xs.nonEmpty)
    if (xs.length == 1) xs.head
    else {
      val next = xs.grouped(2).map {
        case Seq(a, b) => op(a, b)
        case Seq(a)    => a
      }.toSeq
      treeReduce(next)(op)
    }
  }

  // abbreviations
  private val bFmt = p.biasFmt
  private val pFmt = p.psumFmt
  private val oFmt = p.outputFmt
  private val expW = p.expWorkWidth
  private val intW = p.intWidth
  private val N    = p.vecLen

  //  S0:  Weight Select, E4M3 Multiply, Addend Exponent

  // weight buffer mux
  val weightIEEE = Wire(Vec(N, UInt(p.inputFmt.ieeeWidth.W)))
  for (j <- 0 until N)
    weightIEEE(j) := Mux(io.bufReadSel, io.weightBuf1(j), io.weightBuf0(j))

  // lossless E4M3 x E4M3 -> 13-bit product
  val prod_s0 = Wire(Vec(N, UInt(E4M3ProdFmt.width.W)))
  for (j <- 0 until N) {
    val mul = Module(new E4M3Mul)
    mul.io.a    := io.act(j)
    mul.io.b    := weightIEEE(j)
    prod_s0(j)  := mul.io.out
  }

  // addend exponent (combinational, only depends on control inputs)
  val biasExpField = io.bias(bFmt.ieeeWidth - 2, bFmt.mantissaBits)
  val biasIsZero   = io.bias(bFmt.ieeeWidth - 2, 0) === 0.U
  val biasUnbExp   = biasExpField.zext -& bFmt.ieeeBias.S(expW.W)

  val psumExpField = io.psum(pFmt.ieeeWidth - 2, pFmt.mantissaBits)
  val psumFrac     = io.psum(pFmt.mantissaBits - 1, 0)
  val psumIsZero   = psumExpField === 0.U && psumFrac === 0.U
  val psumUnbExp   = psumExpField.zext -& pFmt.ieeeBias.S(expW.W)

  val addendExpS0 = WireDefault(sentinel)
  when(io.addendSel === AddendSel.UseBias) {
    addendExpS0 := Mux(biasIsZero, sentinel, biasUnbExp.pad(expW).asSInt)
  }.elsewhen(io.addendSel === AddendSel.UsePsum) {
    addendExpS0 := Mux(psumIsZero, sentinel, psumUnbExp.pad(expW).asSInt)
  }

  // Cut 0
  val prod_s1      = VecInit((0 until N).map(j => optPipe(prod_s0(j), 0)))
  val addendExp_s1 = optPipe(addendExpS0, 0)
  val bias_s1      = optPipe(io.bias, 0)
  val psum_s1      = optPipe(io.psum, 0)
  val addendSel_s1 = optPipe(io.addendSel, 0)

  //  S1:  Product Exponent Extraction, Anchor Computation

  val prodUnbExp = Wire(Vec(N, SInt(expW.W)))
  for (j <- 0 until N) {
    val pExp  = prod_s1(j)(11, 7)
    val pZero = pExp === 0.U
    prodUnbExp(j) := Mux(pZero,
      sentinel,
      (pExp.zext - E4M3ProdFmt.bias.S).pad(expW).asSInt)
  }

  // anchor = max(all exponents) + headroom
  val allExps: Seq[SInt] = prodUnbExp.toSeq :+ addendExp_s1
  val maxExp  = treeReduce(allExps) { (a: SInt, b: SInt) => Mux(a > b, a, b) }
  val anchor  = (maxExp +& p.anchorHeadroom.S(expW.W)).pad(expW).asSInt

  // Cut 1
  val prod_s2      = VecInit((0 until N).map(j => optPipe(prod_s1(j), 1)))
  val anchor_s2    = optPipe(anchor, 1)
  val bias_s2      = optPipe(bias_s1, 1)
  val psum_s2      = optPipe(psum_s1, 1)
  val addendSel_s2 = optPipe(addendSel_s1, 1)

  //  S2:  Float -> Anchor-Aligned Integer  (products + addend)
  // products: 13-bit custom float -> aligned int
  val prodInt = Wire(Vec(N, SInt(intW.W)))
  for (j <- 0 until N) {
    val conv = Module(new E4M3ProdToAlignedInt(intW, expW))
    conv.io.prod      := prod_s2(j)
    conv.io.anchorExp := anchor_s2
    prodInt(j)        := conv.io.intOut
  }

  // addend: IEEE(biasFmt / psumFmt) -> aligned int
  val biasConv = Module(new IEEEToAlignedInt(bFmt, intW, expW))
  biasConv.io.ieee      := bias_s2
  biasConv.io.anchorExp := anchor_s2

  val psumConv = Module(new IEEEToAlignedInt(pFmt, intW, expW))
  psumConv.io.ieee      := psum_s2
  psumConv.io.anchorExp := anchor_s2

  val addendInt_s2 = WireDefault(0.S(intW.W))
  when(addendSel_s2 === AddendSel.UseBias)        { addendInt_s2 := biasConv.io.intOut }
    .elsewhen(addendSel_s2 === AddendSel.UsePsum)  { addendInt_s2 := psumConv.io.intOut }

  // Cut 2
  val prodInt_s3   = VecInit((0 until N).map(j => optPipe(prodInt(j), 2)))
  val addendInt_s3 = optPipe(addendInt_s2, 2)
  val anchor_s3    = optPipe(anchor_s2, 2)

  //  S3:  Integer Reduction Tree, Addend Addition

  val prodSum: SInt = treeReduce(prodInt_s3.toSeq) { (a: SInt, b: SInt) => a +& b }
  val totalInt = (prodSum + addendInt_s3)(intW - 1, 0).asSInt

  // Cut 3
  val totalInt_s4 = optPipe(totalInt, 3)
  val anchor_s4   = optPipe(anchor_s3, 3)

  //  S4:  Aligned Integer -> IEEE(outputFmt)

  val toFP = Module(new AlignedIntToIEEE(intW, oFmt, expW))
  toFP.io.intIn     := totalInt_s4
  toFP.io.anchorExp := anchor_s4
  val result = toFP.io.ieeeOut

  // output register
  val outReg = RegNext(result, 0.U(oFmt.ieeeWidth.W))
  io.out := outReg
}

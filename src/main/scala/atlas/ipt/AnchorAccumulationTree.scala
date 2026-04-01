package atlas.ipt

import chisel3._
import chisel3.util._
import sp26FPUnits.{E4M3ProdFmt, E4M3Mul}
import atlas.common.InnerProductTreeParams

class AnchorAccumulationTree(p: InnerProductTreeParams) extends Module {

  val io = IO(new Bundle {
    val act        = Input(Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W)))
    val weightBuf0 = Input(Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W)))
    val weightBuf1 = Input(Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W)))
    val psum       = Input(UInt(p.psumFmt.ieeeWidth.W))
    val bufReadSel = Input(Bool())
    val accumulate = Input(Bool())
    val out        = Output(UInt(p.outputFmt.ieeeWidth.W))
  })

  private def optPipe[T <: Data](x: T, cut: Int): T = if (p.pipelineCuts.contains(cut)) RegNext(x) else x

  private val sentinelVal = -(1 << (p.expWorkWidth - 2))

  private def sentinel: SInt = sentinelVal.S(p.expWorkWidth.W)

  private def treeReduce[T](xs: Seq[T])(op: (T, T) => T): T = {
    require(xs.nonEmpty)
    if (xs.length == 1) xs.head
    else { val next = xs.grouped(2).map { 
      case Seq(a, b) => op(a, b)
      case Seq(a) => a }.toSeq
      treeReduce(next)(op) 
    }
  }

  private val pFmt = p.psumFmt
  private val oFmt = p.outputFmt
  private val expW = p.expWorkWidth
  private val intW = p.intWidth
  private val N = p.vecLen

  // S0
  val weightIEEE = Wire(Vec(N, UInt(p.inputFmt.ieeeWidth.W)))

  for (j <- 0 until N) weightIEEE(j) := Mux(io.bufReadSel, io.weightBuf1(j), io.weightBuf0(j))

  val prod_s0 = Wire(Vec(N, UInt(E4M3ProdFmt.width.W)))

  for (j <- 0 until N) { 
    val mul = Module(new E4M3Mul)
    mul.io.a := io.act(j)
    mul.io.b := weightIEEE(j)
    prod_s0(j) := mul.io.out 
  }
  
  val psumExpField = io.psum(pFmt.ieeeWidth - 2, pFmt.mantissaBits)
  val psumFrac = io.psum(pFmt.mantissaBits - 1, 0)
  val psumIsZero = psumExpField === 0.U && psumFrac === 0.U
  val psumUnbExp = psumExpField.zext -& pFmt.ieeeBias.S(expW.W)

  val addendExpS0 = WireDefault(sentinel)
  when(io.accumulate) { addendExpS0 := Mux(psumIsZero, sentinel, psumUnbExp.pad(expW).asSInt) }

  val prod_s1 = VecInit((0 until N).map(j => optPipe(prod_s0(j), 0)))

  val addendExp_s1 = optPipe(addendExpS0, 0)
  val psum_s1 = optPipe(io.psum, 0)
  val accumulate_s1 = optPipe(io.accumulate, 0)

  // S1
  val prodUnbExp = Wire(Vec(N, SInt(expW.W)))
  for (j <- 0 until N) { 
    val pExp = prod_s1(j)(11, 7)
    val pZero = pExp === 0.U
    prodUnbExp(j) := Mux(pZero, sentinel, (pExp.zext - E4M3ProdFmt.bias.S).pad(expW).asSInt) 
  }
  val allExps: Seq[SInt] = prodUnbExp.toSeq :+ addendExp_s1
  val maxExp = treeReduce(allExps) { (a: SInt, b: SInt) => Mux(a > b, a, b) }
  val anchor = (maxExp +& p.anchorHeadroom.S(expW.W)).pad(expW).asSInt
  val prod_s2 = VecInit((0 until N).map(j => optPipe(prod_s1(j), 1)))

  val anchor_s2 = optPipe(anchor, 1)
  val psum_s2 = optPipe(psum_s1, 1)
  val accumulate_s2 = optPipe(accumulate_s1, 1)

  // S2
  val prodInt = Wire(Vec(N, SInt(intW.W)))
  for (j <- 0 until N) { 
    val conv = Module(new E4M3ProdToAlignedInt(intW, expW))
    conv.io.prod := prod_s2(j)
    conv.io.anchorExp := anchor_s2
    prodInt(j) := conv.io.intOut 
  }

  val psumConv = Module(new IEEEToAlignedInt(pFmt, intW, expW))
  psumConv.io.ieee := psum_s2
  psumConv.io.anchorExp := anchor_s2

  val addendInt_s2 = WireDefault(0.S(intW.W))
  when(accumulate_s2) { addendInt_s2 := psumConv.io.intOut }
  val prodInt_s3 = VecInit((0 until N).map(j => optPipe(prodInt(j), 2)))

  val addendInt_s3 = optPipe(addendInt_s2, 2)
  val anchor_s3 = optPipe(anchor_s2, 2)

  // S3
  val prodSum: SInt = treeReduce(prodInt_s3.toSeq) { (a: SInt, b: SInt) => a +& b }
  val totalInt = (prodSum + addendInt_s3)(intW - 1, 0).asSInt

  val totalInt_s4 = optPipe(totalInt, 3)
  val anchor_s4 = optPipe(anchor_s3, 3)

  // S4
  val toFP = Module(new AlignedIntToIEEE(intW, oFmt, expW))
  toFP.io.intIn := totalInt_s4
  toFP.io.anchorExp := anchor_s4

  val rawBF16 = toFP.io.ieeeOut
  val sign = rawBF16(15)
  val expBF16 = rawBF16(14, 7)
  val fracBF16 = rawBF16(6, 0)

  val isSub = (expBF16 === 0.U) && (fracBF16 =/= 0.U)
  val isInf = (expBF16 === "hff".U) && (fracBF16 === 0.U)
  val isNaN = (expBF16 === "hff".U) && (fracBF16 =/= 0.U)

  val sanitized = Wire(UInt(16.W))
  sanitized := rawBF16
  when(isSub || isNaN) { sanitized := 0.U }
  .elsewhen(isInf) { sanitized := Mux(sign.asBool, "hff7f".U(16.W), "h7f7f".U(16.W)) }
  
  val outReg = RegNext(sanitized, 0.U(oFmt.ieeeWidth.W))
  io.out := outReg
}

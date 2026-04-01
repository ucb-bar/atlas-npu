package atlas.common

import sp26FPUnits.AtlasFPType
import sp26FPUnits.E4M3ProdFmt

case class InnerProductTreeParams(
  numLanes: Int = 32, vecLen: Int = 32, tileRows: Int = 32,
  accumIntWidth: Int = 0, pipelineCuts: Set[Int] = Set.empty,
) {
  require(numLanes >= 1)
  require(vecLen >= 1)
  require(tileRows >= 1)
  require(pipelineCuts.forall(c => c >= 0 && c <= 3), s"pipelineCuts must be in {0,...,3}, got $pipelineCuts")

  val inputFmt : AtlasFPType = AtlasFPType.E4M3
  val psumFmt  : AtlasFPType = AtlasFPType.BF16
  val mulFmt                 = E4M3ProdFmt
  val outputFmt: AtlasFPType = AtlasFPType.BF16

  val anchorHeadroom: Int = { 
    val total = vecLen + 1
    BigInt(total).bitLength + 1 
  }

  val intWidth: Int = if (accumIntWidth > 0) accumIntWidth else mulFmt.sigWidth + anchorHeadroom + 17

  val expWorkWidth: Int = Seq(inputFmt.expWidth, mulFmt.expWidth, psumFmt.expWidth, outputFmt.expWidth).max + 4

  val numPipeCuts: Int = pipelineCuts.size

  val latency: Int = numPipeCuts + 1

  val shiftBits: Int = log2Ceil(intWidth + 1)

  val tileRowBits: Int = log2Ceil(tileRows)

  private def log2Ceil(x: Int): Int = { 
    require(x > 0)
    if (x == 1) 1 else BigInt(x - 1).bitLength 
  }

  def isOriginalConfig: Boolean = numLanes == 32 && vecLen == 32 && tileRows == 32

  override def toString: String = {
    val cutsStr = if (pipelineCuts.isEmpty) "none" else pipelineCuts.toSeq.sorted.mkString(",")
    s"InnerProductTreeParams(in=$inputFmt, psum=$psumFmt, out=$outputFmt, ${numLanes}x${vecLen}, tile=${tileRows}, intW=$intWidth, headroom=$anchorHeadroom, cuts={$cutsStr})"
  }
}

object InnerProductTreeParams {
  def withPipelineDepth(depth: Int, base: InnerProductTreeParams = InnerProductTreeParams()): InnerProductTreeParams = {
    require(depth >= 1 && depth <= 5)
    val cuts: Set[Int] = depth match {
      case 1 => Set.empty
      case 2 => Set(1)
      case 3 => Set(0, 2)
      case 4 => Set(0, 1, 2)
      case 5 => Set(0, 1, 2, 3) 
    }
    base.copy(pipelineCuts = cuts)
  }
}

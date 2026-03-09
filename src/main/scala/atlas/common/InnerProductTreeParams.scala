/*
InnerProductTreeParams.scala: floating-point format descriptors and InnerProductTree parameter bundle.

This design is hardcoded for E4M3 inputs.

InnerProductTreeParams: contains every parameter throughout the design:
           input/bias/psum/output formats, lane/vector geometry,
           accumulation integer width, and pipeline cut placement.

Pipeline cuts are specified as a Set[Int] choosing from 4 legal positions between 5 logical stages of the lane datapath:

    S0 WeightSel+Multiply - cut0 - S1 ExpExtract+Anchor - cut1 - S2 Alignment - cut2 - S3 Reduce+Addend - cut3 - S4 Int->FP

The accReg at the lane output is always present (1 additional cycle).
Total latency from compute.fire to out.valid = pipelineCuts + 1.
*/

package atlas.common

import sp26FPUnits.AtlasFPType
import sp26FPUnits.E4M3ProdFmt

case class InnerProductTreeParams(
  // numeric formats (input is always E4M3; product format is E4M3ProdFmt)
  inputFmt : AtlasFPType = AtlasFPType.E4M3,  // activation and weight elements format
  biasFmt  : AtlasFPType = AtlasFPType.E4M3,  // format of per-lane bias input
  psumFmt  : AtlasFPType = AtlasFPType.BF16,  // format of per-lane partial-sum input
  outputFmt: AtlasFPType = AtlasFPType.BF16,  // format of final lane output

  // geometry
  numLanes: Int = 16,                         // number of rows of the weight tile (output width)
  vecLen  : Int = 32,                         // inner-product dimension (columns). number of elements being reduced

  // accumulation
  accumIntWidth: Int = 0,                     // 0 for auto-derive. > 0 to force an exact intWidth

  // pipeline stages
  pipelineCuts: Set[Int] = Set.empty,          // active cut points {0, ... , 3}
) {
  require(numLanes >= 1)
  require(vecLen >= 1)
  require(inputFmt == AtlasFPType.E4M3,
    s"inputFmt must be E4M3, got $inputFmt")
  require(pipelineCuts.forall(c => c >= 0 && c <= 3),
    s"pipelineCuts must be in {0, ... , 3}, got $pipelineCuts")

  // anchor headroom = ceil(log2(vecLen + 1)) + 1
  // the "+1" to vecLen accounts for the addend (bias or psum)
  // the extra final "+1" is a carry guard
  val anchorHeadroom: Int = {
    val total = vecLen + 1 // products + addend
    BigInt(total).bitLength + 1
  }

  // derived accumulation integer width (product significand + headroom + margin)
  val intWidth: Int =
    if (accumIntWidth > 0) accumIntWidth
    else E4M3ProdFmt.sigWidth + anchorHeadroom + 15 // 30 intwidth default

  // exponent working width. wide enough for any format in the design
  val expWorkWidth: Int = {
    val maxEw = Seq(inputFmt.expWidth, E4M3ProdFmt.expWidth,
                    biasFmt.expWidth, psumFmt.expWidth, outputFmt.expWidth).max
    maxEw + 4 // extra bits for signed arithmetic, headroom addition, etc.
  }

  // number of active pipeline register stages inside the lane
  val numPipeCuts: Int = pipelineCuts.size

  // total latency from compute.fire to out.valid (includes output register)
  val latency: Int = numPipeCuts + 1

  // convenience: bits needed to index a shift amount within intWidth.
  val shiftBits: Int = log2Ceil(intWidth + 1)

  private def log2Ceil(x: Int): Int = {
    require(x > 0)
    if (x == 1) 1 else BigInt(x - 1).bitLength
  }

  // helpers

  // default configuration matching the original E4M3 to BF16 design
  def isOriginalConfig: Boolean =
    inputFmt  == AtlasFPType.E4M3 &&
    biasFmt   == AtlasFPType.E4M3 &&
    psumFmt   == AtlasFPType.BF16 &&
    outputFmt == AtlasFPType.BF16 &&
    numLanes == 16 && vecLen == 32

  override def toString: String = {
    val cutsStr = if (pipelineCuts.isEmpty) "none" else pipelineCuts.toSeq.sorted.mkString(",")
    s"InnerProductTreeParams(in=$inputFmt, out=$outputFmt, " +
    s"${numLanes}x${vecLen}, intW=$intWidth, headroom=$anchorHeadroom, cuts={$cutsStr})"
  }
}

object InnerProductTreeParams {
  // create params with pipeline depth to auto-assign cuts
  def withPipelineDepth(depth: Int, base: InnerProductTreeParams = InnerProductTreeParams()): InnerProductTreeParams = {
    require(depth >= 1 && depth <= 5, s"Pipeline depth must be 1..5, got $depth")
    val cuts: Set[Int] = depth match {
      case 1 => Set.empty
      case 2 => Set(1)             // after anchor
      case 3 => Set(0, 2)          // + after alignment
      case 4 => Set(0, 1, 2)       // + after multiply
      case 5 => Set(0, 1, 2, 3)    // all cuts
    }
    base.copy(pipelineCuts = cuts)
  }
}

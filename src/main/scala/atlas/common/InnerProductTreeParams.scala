package atlas.common

import chisel3._
import chisel3.util._
import sp26FPUnits.AtlasFPType
import atlas.mxu.MxuParams

case class InnerProductTreeParams(
  mxu:           MxuParams   = MxuParams(),
  pipelineCuts:  Set[Int]    = Set(1), // default: 2-stage pipelined
) {
  require(pipelineCuts.forall(c => c >= 0 && c <= 3),
    s"pipelineCuts must be in {0,...,3}, got $pipelineCuts")

  require(mxu.arrayRows > 1, s"Inner product trees must have >1 rows, got ${mxu.arrayRows}")
  require(mxu.arrayCols > 1, s"Inner product trees must have >1 cols, got ${mxu.arrayCols}")

  // ── Convenience aliases (match prior API so downstream code compiles) ──
  val numLanes: Int = mxu.arrayCols
  val vecLen:   Int = mxu.arrayRows
  val tileRows: Int = mxu.accumBufferRows

  val inputFmt:  AtlasFPType = mxu.inputFmt
  val accumFmt:  AtlasFPType = mxu.accumFmt

  // ── Fused dot-add (docs/fp8-mxu.md): terms in units of 2^(emax - 25) ──
  /** Two's-complement width of the exact sum of vecLen products (each below 225 x 2^19) and C
    * (below 2^24 x 4); 33 bits for vecLen = 32. */
  val sumWidth: Int = log2Ceil(BigInt(vecLen) * 225 * (1 << 19) + ((1 << 24) - 1) * 4 + 1) + 1

  // ── Pipeline ──
  val numPipeCuts: Int = pipelineCuts.size
  val latency:     Int = numPipeCuts + 1
  val tileRowBits: Int = log2Ceil(tileRows)

  override def toString: String = {
    val cutsStr = if (pipelineCuts.isEmpty) "none"
                  else pipelineCuts.toSeq.sorted.mkString(",")
    s"InnerProductTreeParams(mxu=$mxu, sumW=$sumWidth, cuts={$cutsStr})"
  }
}

object InnerProductTreeParams {
  def withPipelineDepth(
    depth: Int,
    base:  InnerProductTreeParams = InnerProductTreeParams()
  ): InnerProductTreeParams = {
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

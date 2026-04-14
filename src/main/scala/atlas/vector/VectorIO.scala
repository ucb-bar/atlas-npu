// ============================================================================
// VPUData.scala — Shared command/data bundles for the vector engine.
//
// Defines the vector-operation enum plus the small request/response bundles
// used at the top level of the standalone vector engine.
//
// IMPORTANT: VPUOp enum ordering must match ScalarISA VPU_* constants minus 1.
//   ScalarISA is 1-indexed (VPU_NONE=0, VPU_ADD=1, VPU_SUB=2, ...),
//   while ChiselEnum is 0-indexed (add=0, sub=1, mul=2, ...).
//   VectorEngineTop subtracts 1 before casting to bridge the offset.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.VpuParams

object VPUOp extends ChiselEnum {
  // Matches ScalarISA VPU_* values minus 1:
  val add, sub, mul, rcp, sqrt, sin, cos, tanh, log, exp, exp2,
      square, cube, rsum, csum, fp8, fp8pack, fp8unpack, relu,
      rmax, rmin, cmax, cmin, pairmax, pairmin, mov,
      vliOne, vliCol, vliRow, vliAll = Value
}

/** Top-level vector-engine input bundle.
  *
  * @param p  Vector-engine geometry and format parameters.
  */
class VPUInput(val p: VpuParams) extends Bundle {
  val instType = VPUOp()
  val isExpNeg = Bool()
  val vector   = Vec(p.numLanes, UInt(p.wordWidth.W))
}

/** Top-level vector-engine vector result bundle.
  *
  * @param p  Vector-engine geometry and format parameters.
  */
class VPUOutput(val p: VpuParams) extends Bundle {
  val vectorOutputData = Vec(p.numLanes, UInt(p.wordWidth.W))
}

class VectorInput(val p: VpuParams) extends Bundle {
  val instType        = VPUOp()
  val instReadBank1   = UInt(6.W)
  val instReadBank2   = UInt(6.W)
  val instWriteBank   = UInt(6.W)
  val imm             = SInt(16.W)
  val packScaleE8M0   = UInt(8.W)   // E8M0 scale for VFP8PACK
  val unpackScaleE8M0 = UInt(8.W)   // E8M0 scale for VFP8UNPACK
}

class RegData extends Bundle {
  val data = UInt(256.W)
}

class RegFileReadReq extends Bundle {
  val bank = UInt(6.W)
  val row  = UInt(5.W)
}

class RegFileWriteReq extends Bundle {
  val bank = UInt(6.W)
  val row  = UInt(5.W)
  val data = UInt(256.W)
}

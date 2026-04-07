// ============================================================================
// VPUData.scala — Shared command/data bundles for the vector engine.
//
// Defines the vector-operation enum plus the small request/response bundles
// used at the top level of the standalone vector engine.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.VpuParams
import sp26FPUnits.hardfloat._

object VPUOp extends ChiselEnum {
  val add, sub, mul, rcp, sqrt, sin, cos, tanh, log, exp, exp2, square, cube, max, reduSum, fp8, relu = Value
}

/** Top-level vector-engine input bundle.
  *
  * @param p  Vector-engine geometry and format parameters.
  */
class VPUInput(val p: VpuParams) extends Bundle {
    val instType  = VPUOp()
    val isExpNeg = Bool()
    //val vectorInputData = Vec(p.numLanes, UInt(p.wordWidth.W))
    val vector = Vec(p.numLanes, UInt(p.wordWidth.W))
}

/** Top-level vector-engine vector result bundle.
  *
  * @param p  Vector-engine geometry and format parameters.
  */
class VPUOutput(val p: VpuParams) extends Bundle {
    val vectorOutputData = Vec(p.numLanes, UInt(p.wordWidth.W))
}

/** Single-value reduction output bundle.
  *
  * @param p  Vector-engine geometry and format parameters.
  */
class VPUOutputMaxRedu(val p: VpuParams) extends Bundle {
    val maxReduceOutput = UInt(p.wordWidth.W)
}

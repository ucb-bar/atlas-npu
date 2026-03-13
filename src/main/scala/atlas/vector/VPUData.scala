package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.VPUParams
import sp26FPUnits.hardfloat._

object VPUOp extends ChiselEnum {
  val add, sub, mul, rcp, sqrt, sin, cos, tanh, log, exp, exp2, square, cube, max, reduSum, fp8 = Value
}

class VPUInput(val p: VPUParams) extends Bundle {
    val instType  = VPUOp()
    val isExpNeg = Bool()
    //val vectorInputData = Vec(p.numLanes, UInt(p.wordWidth.W))
    val vector = Vec(p.numLanes, UInt(p.wordWidth.W))
}

class VPUOutput(val p: VPUParams) extends Bundle {
    val vectorOutputData = Vec(p.numLanes, UInt(p.wordWidth.W))
}
class VPUOutputMaxRedu(val p: VPUParams) extends Bundle {
    val maxReduceOutput = UInt(p.wordWidth.W)
}


class RegFileReadInput extends Bundle {
    val whichBank = UInt(5.W)
    val rRow= UInt(8.W)
}

class RegFileWriteInput extends Bundle {
    val whichBank = UInt(5.W)
    val wRow= UInt(8.W)
    val wData = UInt(256.W)
}

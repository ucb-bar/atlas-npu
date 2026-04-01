package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.{InnerProductTreeParams, RegFileParams}

object Mxu0Op extends ChiselEnum {
  val PushWeight  = Value(0.U)
  val PushAccFP8  = Value(1.U)
  val PushAccBF16 = Value(2.U)
  val PopAccFP8   = Value(3.U)
  val PopAccBF16  = Value(4.U)
  val Matmul      = Value(5.U)
  val MatmulAcc   = Value(6.U)
}

class Mxu0Cmd(rfP: RegFileParams) extends Bundle {
  val op         = Mxu0Op()
  val trfBank    = UInt(rfP.NUM_BANKS_BITS.W)
  val accSel     = Bool()
  val weightSlot = Bool()
  val scaleE8M0  = UInt(8.W)   // E8M0 scaling factor for PopAccFP8 (passed directly by scheduler)
}

class ComputeReq(p: InnerProductTreeParams) extends Bundle {
  val act          = Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W))
  val psum         = Vec(p.numLanes, UInt(p.psumFmt.ieeeWidth.W))
  val accumulate   = Bool()
  val weightBufSel = Bool()
}

class WeightWriteReq(p: InnerProductTreeParams) extends Bundle {
  val targetBuf = Bool()
  val laneIdx   = UInt(log2Ceil(p.numLanes).W)
  val weights   = Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W))
}

class AccBufMxu0Write(p: InnerProductTreeParams) extends Bundle {
  val accSel = Bool()
  val rowIdx = UInt(p.tileRowBits.W)
  val data   = Vec(p.numLanes, UInt(p.outputFmt.ieeeWidth.W))
}

class AccBufMxu0Read(p: InnerProductTreeParams) extends Bundle {
  val accSel = Bool()
  val rowIdx = UInt(p.tileRowBits.W)
}

class AccBufLoadFP8(p: InnerProductTreeParams) extends Bundle {
  val accSel = Bool()
  val rowIdx = UInt(p.tileRowBits.W)
  val data   = Vec(p.numLanes, UInt(8.W))
}

class AccBufLoadBF16(p: InnerProductTreeParams) extends Bundle {
  val accSel = Bool()
  val rowIdx = UInt(p.tileRowBits.W)
  val data   = Vec(p.numLanes, UInt(16.W))
}

class AccBufStoreReq(p: InnerProductTreeParams) extends Bundle {
  val accSel    = Bool()
  val rowIdx    = UInt(p.tileRowBits.W)
  val scaleE8M0 = UInt(8.W)   // E8M0 value directly (scheduler provides it)
}

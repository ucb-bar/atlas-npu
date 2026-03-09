/*
Top-level inner-product-tree matrix-multiply unit, for
E4M3 x E4M3 inputs.

- 8-bit E4M3 values are broadcast directly to every lane.
- per-lane AnchorAccumulationTree uses E4M3Mul instead of HardFloat's MulRecFN.
*/

package atlas.ipt

import chisel3._
import chisel3.util._
import sp26FPUnits.hardfloat._
import sp26FPUnits.AtlasFPType
import atlas.common.InnerProductTreeParams

object AddendSel extends ChiselEnum {
  val UseAct, UseBias, UsePsum = Value
}

object WeightSel extends ChiselEnum {
  val UseDma, UseTr = Value
}

// IO bundles (external interface stays IEEE)
class ComputeReq(p: InnerProductTreeParams) extends Bundle {
  val act       = Vec(p.vecLen,   UInt(p.inputFmt.ieeeWidth.W))
  val bias      = Vec(p.numLanes, UInt(p.biasFmt.ieeeWidth.W))
  val psum      = Vec(p.numLanes, UInt(p.psumFmt.ieeeWidth.W))
  val addendSel = AddendSel()
}

class WeightLoadReq(p: InnerProductTreeParams) extends Bundle {
  val weightSel  = WeightSel()
  val weightsDma = Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W))
  val weightsTr  = Vec(p.vecLen, UInt(p.inputFmt.ieeeWidth.W))
  val laneIdx    = UInt(log2Ceil(p.numLanes).W)
  val last       = Bool()
}

class InnerProductTrees(p: InnerProductTreeParams = InnerProductTreeParams()) extends Module {

  val io = IO(new Bundle {
    val compute    = Flipped(DecoupledIO(new ComputeReq(p)))
    val weightLoad = Flipped(DecoupledIO(new WeightLoadReq(p)))
    val out        = ValidIO(Vec(p.numLanes, UInt(p.outputFmt.ieeeWidth.W)))
  })

  // always accept, no stall
  io.compute.ready    := true.B
  io.weightLoad.ready := true.B

  // valid pipeline
  val outValid = RegInit(false.B)
  if (p.numPipeCuts == 0) {
    outValid := io.compute.fire
  } else {
    val validSr = RegInit(VecInit(Seq.fill(p.numPipeCuts)(false.B)))
    validSr(0) := io.compute.fire
    for (i <- 1 until p.numPipeCuts) validSr(i) := validSr(i - 1)
    outValid := validSr(p.numPipeCuts - 1)
  }
  io.out.valid := outValid

  //  Activations: pass through IEEE E4M3
  val req = io.compute.bits

  //  Weight double-buffer (stores IEEE)
  val ieeeWidth = p.inputFmt.ieeeWidth

  val wEn = RegInit(false.B)
  val bufReadSel = ~wEn

  val zeroRow  = VecInit(Seq.fill(p.vecLen)(0.U(ieeeWidth.W)))
  val zeroTile = VecInit(Seq.fill(p.numLanes)(zeroRow))

  val wbuf0 = RegInit(zeroTile)
  val wbuf1 = RegInit(zeroTile)

  val wlReq   = io.weightLoad.bits
  val srcIEEE = Wire(Vec(p.vecLen, UInt(ieeeWidth.W)))
  for (c <- 0 until p.vecLen) {
    srcIEEE(c) := Mux(wlReq.weightSel === WeightSel.UseDma,
                       wlReq.weightsDma(c), wlReq.weightsTr(c))
  }

  when(io.weightLoad.fire) {
    val laneIdx = wlReq.laneIdx
    when(wEn)    { wbuf1(laneIdx) := srcIEEE }
      .otherwise { wbuf0(laneIdx) := srcIEEE }
    when(wlReq.last) { wEn := ~wEn }
  }

  //  Lanes: E4M3 in, IEEE(outputFmt) out
  for (r <- 0 until p.numLanes) {
    val lane = Module(new AnchorAccumulationTree(p))
    lane.io.act        := req.act           // 8-bit E4M3
    lane.io.weightBuf0 := wbuf0(r)
    lane.io.weightBuf1 := wbuf1(r)
    lane.io.bias       := req.bias(r)
    lane.io.psum       := req.psum(r)
    lane.io.bufReadSel := bufReadSel
    lane.io.addendSel  := req.addendSel
    io.out.bits(r)     := lane.io.out
    
  }
}

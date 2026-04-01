package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.InnerProductTreeParams

class InnerProductTrees(p: InnerProductTreeParams = InnerProductTreeParams()) extends Module {

  val io = IO(new Bundle {
    val compute     = Flipped(DecoupledIO(new ComputeReq(p)))
    val weightWrite = Flipped(Valid(new WeightWriteReq(p)))
    val out         = ValidIO(Vec(p.numLanes, UInt(p.outputFmt.ieeeWidth.W)))
  })

  io.compute.ready := true.B
  val outValid = RegInit(false.B)

  if (p.numPipeCuts == 0) { outValid := io.compute.fire }
  else {
    val validSr = RegInit(VecInit(Seq.fill(p.numPipeCuts)(false.B)))
    validSr(0) := io.compute.fire
    for (i <- 1 until p.numPipeCuts) validSr(i) := validSr(i - 1)
    outValid := validSr(p.numPipeCuts - 1)
  }

  io.out.valid := outValid

  private val zeroRow = VecInit(Seq.fill(p.vecLen)(0.U(p.inputFmt.ieeeWidth.W)))
  private val zeroTile = VecInit(Seq.fill(p.numLanes)(zeroRow))
  val wbuf0 = RegInit(zeroTile)
  val wbuf1 = RegInit(zeroTile)
  when(io.weightWrite.valid) {
    when(io.weightWrite.bits.targetBuf) { wbuf1(io.weightWrite.bits.laneIdx) := io.weightWrite.bits.weights }
    .otherwise { wbuf0(io.weightWrite.bits.laneIdx) := io.weightWrite.bits.weights }
  }

  val req = io.compute.bits

  for (laneIdx <- 0 until p.numLanes) {
    val lane = Module(new AnchorAccumulationTree(p))
    lane.io.act := req.act
    lane.io.weightBuf0 := wbuf0(laneIdx)
    lane.io.weightBuf1 := wbuf1(laneIdx)
    lane.io.psum := req.psum(laneIdx)
    lane.io.bufReadSel := req.weightBufSel
    lane.io.accumulate := req.accumulate
    io.out.bits(laneIdx) := lane.io.out
  }
  
}

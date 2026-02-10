package atlas.top

import chisel3._
import chisel3.util._
import fpex._
import atlas.common.AtlasParams
import atlas.dma._
import atlas.vector._
import atlas.matrix._


/* 
  This only instantiates an exponential unit for experimental purposes.
  TODO: fix 
 */
class AtlasCluster(p: AtlasParams = AtlasParams()) extends Module {
  val io = IO(new Bundle {
    val fpex_req  = Flipped(Decoupled(new FPEXReq(FPType.FP32T.wordWidth, numLanes = 4, tagWidth = 16)))
    val fpex_resp = Decoupled(new FPEXResp(FPType.FP32T.wordWidth, numLanes = 4, tagWidth = 16))
  })

  val dma    = Module(new DmaEngine(p))
  val vector = Module(new VectorEngine(p))
  val matrix = Module(new MatrixEngine(p))

  // Hard-coded FPEX instance
  val fpex = Module(new FPEX(
    FPType.FP32T,
    numLanes = 4,
    tagWidth = 16
  ))

  // Connect top-level IO straight through
  fpex.io.req  <> io.fpex_req
  io.fpex_resp <> fpex.io.resp
}

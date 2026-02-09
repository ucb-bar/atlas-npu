package atlas.top

import chisel3._
import atlas.common.AtlasParams
import atlas.dma._
import atlas.vector._
import atlas.matrix._

class AtlasCluster(p: AtlasParams = AtlasParams()) extends Module {
  val io = IO(new Bundle {
    // TODO: top-level IO
  })

  val dma    = Module(new DmaEngine(p))
  val vector = Module(new VectorEngine(p))
  val matrix = Module(new MatrixEngine(p))

  // TODO: wire them together
}

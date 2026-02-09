package atlas

import org.scalatest.flatspec.AnyFlatSpec
import circt.stage.ChiselStage
import atlas.top.AtlasCluster

class AtlasClusterSpec extends AnyFlatSpec {
  it should "elaborate" in {
    ChiselStage.emitSystemVerilog(new AtlasCluster)
  }
}

package atlas

import circt.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters
import atlas.config._
import atlas.top.AtlasCluster
import chipyard.AtlasDefaultConfig

object Elaborate extends App {
  val firtoolOptions = Array(
    "--lowering-options=" + List(
      "disallowLocalVariables",
      "disallowPackedArrays",
      "locationInfoStyle=wrapInAtSquareBracket"
    ).mkString(",")
  )

  implicit val p: Parameters = new AtlasDefaultConfig
  ChiselStage.emitSystemVerilogFile(new AtlasCluster, args, firtoolOptions)
}

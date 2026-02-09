package atlas

import circt.stage.ChiselStage
import atlas.top.AtlasCluster

object Elaborate extends App {
  val firtoolOptions = Array(
    "--lowering-options=" + List(
      "disallowLocalVariables",
      "disallowPackedArrays",
      "locationInfoStyle=wrapInAtSquareBracket"
    ).mkString(",")
  )

  // Writes AtlasCluster.sv into --target-dir (or current dir if not provided)
  ChiselStage.emitSystemVerilogFile(new AtlasCluster, args, firtoolOptions)
}

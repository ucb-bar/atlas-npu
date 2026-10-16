// src/main/scala/atlas/top/AtlasCluster.scala
package atlas.top

import chisel3._
import org.chipsalliance.cde.config.Parameters
import atlas.config._
import atlas.ipt._

class AtlasCluster(implicit p: Parameters) extends Module {
  val params = p(InnerProductTreeKey).getOrElse(
    throw new IllegalArgumentException(
      "AtlasCluster requires InnerProductTreeKey to be set in the config. " +
      "Use a config that includes WithInnerProductTrees."
    )
  )
  val matrixUnit = Module(new InnerProductTrees(params))
  val io = IO(matrixUnit.io.cloneType)
  io <> matrixUnit.io
}

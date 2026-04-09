// chipyard/src/main/scala/atlas/top/InnerProductTreesTop.scala
package atlas.top

import chisel3._
import org.chipsalliance.cde.config.Parameters
import atlas.config._
import atlas.ipt._

class InnerProductTreesTop(implicit p: Parameters) extends Module {
  val params = p(InnerProductTreeKey).getOrElse(
    throw new IllegalArgumentException(
      "InnerProductTreesTop requires InnerProductTreeKey to be set in the config. " +
      "Use a config that includes WithInnerProductTrees."
    )
  )

  val inner = Module(new InnerProductTrees(params))

  val io = IO(inner.io.cloneType)
  io <> inner.io
}

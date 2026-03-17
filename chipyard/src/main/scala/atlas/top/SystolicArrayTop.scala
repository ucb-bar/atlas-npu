package atlas.top

import chisel3._
import org.chipsalliance.cde.config.Parameters
import atlas.config._
import atlas.sa._

class SystolicArrayTop(implicit p: Parameters) extends Module {
  val params = p(SystolicArrayKey).getOrElse(
    throw new IllegalArgumentException(
      "SystolicArrayTop requires SystolicArrayKey to be set in the config. " +
      "Use a config that includes WithSystolicArray."
    )
  )

  val inner = Module(new SystolicArray(params))

  val io = IO(inner.io.cloneType)
  io <> inner.io
}
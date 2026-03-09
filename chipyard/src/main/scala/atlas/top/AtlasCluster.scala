// src/main/scala/atlas/top/AtlasCluster.scala
package atlas.top

import chisel3._
import org.chipsalliance.cde.config.Parameters
import atlas.config._
import atlas.ipt._

class AtlasCluster(implicit p: Parameters) extends Module {

  // Matrix unit (InnerProductTrees)
  val matrixUnit = p(InnerProductTreeKey).map { params =>
    val m = Module(new InnerProductTrees(params))
    m
  }

  // Future subunits
  // val vectorUnit = p(VectorUnitKey).map { params =>
  //   Module(new VectorUnit(params))
  // }
  // val controller = p(ControllerKey).map { params =>
  //   Module(new Controller(params))
  // }

  // Top-level IO
  // For now, just expose the matrix unit IO if present.
  // Replace this with a proper unified IO bundle as Atlas grows.
  val io = IO(new Bundle {
    val matrix = matrixUnit.map { m =>
      val port = IO(m.io.cloneType)
      port
    }
  })

  matrixUnit.foreach { m =>
    io.matrix.foreach { port =>
      port <> m.io
    }
  }
}

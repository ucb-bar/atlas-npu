// chipyard/src/main/scala/atlas/config/WithInnerProductTrees.scala
package atlas.config

import org.chipsalliance.cde.config._
import atlas.common.{InnerProductTreeParams}

class WithInnerProductTrees(
  numLanes: Int = 32,
  vecLen: Int = 32,
  tileRows: Int = 32,
  pipelineDepth: Int = 1,
  accumIntWidth: Int = 0
) extends Config((site, here, up) => {
  case InnerProductTreeKey => Some(
    InnerProductTreeParams.withPipelineDepth(
      depth = pipelineDepth,
      base = InnerProductTreeParams(
        numLanes = numLanes,
        vecLen = vecLen,
        tileRows = tileRows,
        accumIntWidth = accumIntWidth
      )
    )
  )
})
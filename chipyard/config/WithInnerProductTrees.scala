// chipyard/src/main/scala/atlas/config/WithInnerProductTrees.scala
package atlas.config

import org.chipsalliance.cde.config._
import atlas.common.InnerProductTreeParams
import atlas.mxu.MxuParams

// Default inner-product-trees config
class WithInnerProductTrees(
  arrayRows: Int = 32,
  arrayCols: Int = 32,
  pipelineDepth: Int = 2,
  accumIntWidth: Int = 0
) extends Config((site, here, up) => {
  case InnerProductTreeKey => Some(
    InnerProductTreeParams.withPipelineDepth(
      depth = pipelineDepth,
      base = InnerProductTreeParams(
        mxu = MxuParams(
          arrayRows = arrayRows,
          arrayCols = arrayCols
        ),
        accumIntWidth = accumIntWidth
      )
    )
  )
})
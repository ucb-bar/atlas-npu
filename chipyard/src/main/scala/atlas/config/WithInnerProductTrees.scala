// chipyard/src/main/scala/atlas/config/WithInnerProductTrees.scala
package atlas.config

import org.chipsalliance.cde.config._
import atlas.common.InnerProductTreeParams
import sp26FPUnits.AtlasFPType

// Default matrix unit config
class WithInnerProductTrees(
  numLanes: Int = 16,
  vecLen: Int = 32,
  pipelineDepth: Int = 2,
  inputFmt: AtlasFPType = AtlasFPType.E4M3,
  mulFmt: AtlasFPType = AtlasFPType.FP16,
  biasFmt: AtlasFPType = AtlasFPType.E4M3,
  psumFmt: AtlasFPType = AtlasFPType.BF16,
  outputFmt: AtlasFPType = AtlasFPType.BF16,
  accumIntWidth: Int = 0
) extends Config((site, here, up) => {
  case InnerProductTreeKey => Some(
    InnerProductTreeParams.withPipelineDepth(
      depth = pipelineDepth,
      base = InnerProductTreeParams(
        inputFmt = inputFmt,
        biasFmt = biasFmt,
        psumFmt = psumFmt,
        outputFmt = outputFmt,
        numLanes = numLanes,
        vecLen = vecLen,
        accumIntWidth = accumIntWidth
      )
    )
  )
})

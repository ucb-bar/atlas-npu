package atlas.config

import org.chipsalliance.cde.config._
import atlas.common.SystolicArrayParams
import sp26FPUnits.AtlasFPType

// Default matrix unit config
class WithSystolicArray(
  inT: AtlasFPType = AtlasFPType.E4M3,
  outT: AtlasFPType = AtlasFPType.BF16,
  rows: Int = 32,
  cols: Int = 16,
  useFP32Accumulation: Boolean = false,
  useE4M3FMA: Boolean = false
) extends Config((site, here, up) => {
  case SystolicArrayKey => Some(
    SystolicArrayParams(
      inT = inT,
      outT = outT,
      rows = rows,
      cols = cols,
      useFP32Accumulation = useFP32Accumulation,
      useE4M3FMA = useE4M3FMA
    )
  )
})

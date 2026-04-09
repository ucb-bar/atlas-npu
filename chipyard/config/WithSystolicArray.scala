package atlas.config

import org.chipsalliance.cde.config._
import atlas.common.{SystolicArrayParams, PEArchitecture}
import atlas.mxu.MxuParams

// Default systolic-array config
class WithSystolicArray(
  rows: Int = 32,
  cols: Int = 32,
  peArch: PEArchitecture = PEArchitecture.HardFloatFMA
) extends Config((site, here, up) => {
  case SystolicArrayKey => Some(
    SystolicArrayParams(
      mxu = MxuParams(
        arrayRows = rows,
        arrayCols = cols
      ),
      peArch = peArch
    )
  )
})

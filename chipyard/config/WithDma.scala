package atlas.config

import org.chipsalliance.cde.config.{Config, Parameters}
import atlas.common.DmaParams

class WithDma(
    beatBytes:        Int    = 32,
    numChannels:      Int    = 8,
    maxInFlight:      Int    = 64,
    maxTransferBytes: Int    = 4096,
    name:             String = "dma"
) extends Config((site, here, up) => {
  case DmaKey => Some(
    DmaParams(
      beatBytes        = beatBytes,
      numChannels      = numChannels,
      maxInFlight      = maxInFlight,
      maxTransferBytes = maxTransferBytes,
      name             = name
    )
  )
})

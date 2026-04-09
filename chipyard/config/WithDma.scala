package atlas.config

import org.chipsalliance.cde.config.{Config, Parameters}
import atlas.common.DmaParams

class WithDma(
    beatBytes:        Int    = 32,
    tagBits:          Int    = 6,
    numChannels:      Int    = 8,
    channelIdBits:    Int    = 3,
    fenceRespBits:    Int    = 4,
    maxInFlight:      Int    = 64,
    maxTransferBytes: Int    = 4096,
    name:             String = "dma"
) extends Config((site, here, up) => {
  case DmaKey => Some(
    DmaParams(
      beatBytes        = beatBytes,
      tagBits          = tagBits,
      numChannels      = numChannels,
      channelIdBits    = channelIdBits,
      fenceRespBits    = fenceRespBits,
      maxInFlight      = maxInFlight,
      maxTransferBytes = maxTransferBytes,
      name             = name
    )
  )
})

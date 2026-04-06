package atlas.config

import org.chipsalliance.cde.config.{Config, Parameters}
import atlas.common.DMAParams

class WithDMA(
    widthBytes: Int = 32,
    tagBits: Int = 6,
    numInterfaces: Int = 8,
    numInterfaceBits: Int = 3,
    fenceRespBits: Int = 4,
    maxInFlight: Int = 64,
    maxSize: Int = 4096,
    maxSizeBits: Int = 13,
    name: String = "dma"
) extends Config((site, here, up) => {
    case DMAKey => Some(
        DMAParams(
            widthBytes = widthBytes,
            tagBits = tagBits,
            numInterfaces = numInterfaces,
            numInterfaceBits = numInterfaceBits,
            fenceRespBits = fenceRespBits,
            maxInFlight = maxInFlight,
            maxSize = maxSize,
            maxSizeBits = maxSizeBits,
            name = name
        )
    )
})

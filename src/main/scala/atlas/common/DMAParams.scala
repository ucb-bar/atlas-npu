package atlas.common

case class DMAParams(
  widthBytes:      Int = 32,    // matches system bus / scratchpad line (256 bits)
  tagBits:         Int = 6,
  numInterfaces:   Int = 8,
  numInterfaceBits:Int = 3,
  fenceRespBits:   Int = 4,
  maxInFlight:     Int = 64,
  maxSize:         Int = 4096,
  maxSizeBits:     Int = 13,
  name:            String = "dma"
)

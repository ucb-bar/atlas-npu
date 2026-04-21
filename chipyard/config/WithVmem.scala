package atlas.config

import chisel3._
import org.chipsalliance.cde.config.{Parameters, Config}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tilelink.TLBusWrapperTopology
import freechips.rocketchip.diplomacy.{BufferParams, AddressSet}
import atlas.common.VmemParams
import atlas.scalar.AtlasMemMap

class WithVmem(
    lineWidthBits: Int    = 256,
    capacityBytes: Int    = AtlasMemMap.VMEM_SIZE,
    base:          BigInt = BigInt(AtlasMemMap.VMEM_BASE),
    beatBytes:     Int    = 32,
    busWhere:      TLBusWrapperLocation = SBUS
) extends Config((site, here, up) => {
  case VmemKey => up(VmemKey) :+ VmemParams(
    lineWidthBits = lineWidthBits,
    capacityBytes = capacityBytes,
    base          = base,
    beatBytes     = beatBytes
  )
})

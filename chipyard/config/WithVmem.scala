package atlas.config

import chisel3._
import org.chipsalliance.cde.config.{Parameters, Config}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tilelink.TLBusWrapperTopology
import freechips.rocketchip.diplomacy.{BufferParams, AddressSet}
import atlas.common.VmemParams

class WithVmem(
    lineWidthBits: Int    = 256,
    capacityBytes: Int    = 256 * 1024,
    base:          BigInt = 0x2000_0000L,
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

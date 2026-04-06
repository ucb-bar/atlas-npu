package atlas.config

import chisel3._
import org.chipsalliance.cde.config.{Parameters, Config}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tilelink.{TLBusWrapperTopology}
import freechips.rocketchip.diplomacy.{BufferParams, AddressSet}
import atlas.common.{ScratchpadParams, SubsystemInjectorKey, SubsystemInjector}

class WithScratchpad(
  base:      BigInt = 0x2000_0000L,
  sizeBytes: Int    = 1024 * 1024,
  banks:     Int    = 1,
  subBanks:  Int    = 1,
  busWhere:  TLBusWrapperLocation = SBUS
) extends Config((site, here, up) => {
  case ScratchpadKey => up(ScratchpadKey) :+ ScratchpadParams(
    base      = base,
    sizeBytes = sizeBytes,
    banks     = banks,
    subBanks  = subBanks,
    busWhere  = busWhere,
    name      = "atlas-scratchpad"
  )
  // case SubsystemInjectorKey => up(SubsystemInjectorKey) + ScratchpadInjector
})


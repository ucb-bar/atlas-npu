package atlas.config

import chisel3._
import org.chipsalliance.cde.config.{Parameters, Config}
import freechips.rocketchip.subsystem._
import freechips.rocketchip.tilelink.{TLBusWrapperTopology}
import freechips.rocketchip.diplomacy.{BufferParams, AddressSet}
import atlas.common.{AtlasConfigs, AtlasParams}
import testchipip.soc.SubsystemInjectorKey
import atlas.tile.{AtlasTileKey, AtlasTileParams,AtlasTileInjector}

class WithAtlasTile(
  base:            BigInt               = AtlasParams().vmem.base,
  size:            BigInt               = AtlasParams().vmem.sizeBytes,
  busWhere:        TLBusWrapperLocation = SBUS,
  name:            String               = "atlas-tile",
  disableMonitors: Boolean              = false,
  atlasParams:     AtlasParams          = AtlasParams()
) extends Config((site, here, up) => {
  case AtlasTileKey =>
    Seq(AtlasTileParams(
      base            = base,
      size            = size,
      busWhere        = busWhere,
      name            = name,
      disableMonitors = disableMonitors,
      atlasParams     = atlasParams
    ))
  case SubsystemInjectorKey =>
    up(SubsystemInjectorKey) + AtlasTileInjector
})

/** Atlas with the inner-product-tree MXU only: the systolic array (MXU0) is removed and
  * its instructions are illegal. */
class WithAtlasTileIptOnly extends WithAtlasTile(atlasParams = AtlasConfigs.iptOnly)

/** Atlas with the inner-product-tree MXU and the NVFP4 MXU (docs/nvfp4-mxu.md), which takes
  * the systolic array's MREG ports. */
class WithAtlasTileNvfp4 extends WithAtlasTile(atlasParams = AtlasConfigs.iptNvfp4)

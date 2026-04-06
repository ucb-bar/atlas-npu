// chipyard/src/main/scala/atlas/config/AtlasConfigs.scala
package chipyard

import org.chipsalliance.cde.config.Config
import atlas.config._
import saturn.common.VectorParams

// Subunit-only configs (for isolated synthesis)

// Matrix unit with default params (16 lanes, vecLen=32, 1-cycle)
class InnerProductTreesDefaultConfig extends Config(
  new WithInnerProductTrees
)

class SystolicArrayDefaultConfig extends Config(
  new WithSystolicArray
)

// // Full Atlas configs
// class AtlasRocketConfig extends Config(
//   new freechips.rocketchip.rocket.WithNHugeCores(1) ++
//   new chipyard.config.AbstractConfig ++
//   new WithAtlasTile()
// )

class AtlasDefaultConfig extends Config(
  new WithInnerProductTrees ++
  new WithDMA() ++
  new WithScratchpad(base=0x20000000L, sizeBytes = (1 << 20)) ++ 
  new WithAtlasTile()
  // ++ new WithSystolicArray
  // ++ new WithVectorUnit
  // ++ new WithController
)

class AtlasPlainRocketConfig extends Config(
  new WithAtlasTile() ++
  new freechips.rocketchip.rocket.WithNBigCores(1) ++
  new EE290BaseConfig
)

class AtlasShuttleVectorConfig extends Config(
  new WithAtlasTile() ++
  new saturn.shuttle.WithShuttleVectorUnit(256, 128, VectorParams.genParams) ++
  new chipyard.config.WithSystemBusWidth(256) ++
  new shuttle.common.WithShuttleTileBeatBytes(16) ++
  new shuttle.common.WithNShuttleCores(1) ++
  new freechips.rocketchip.subsystem.WithCacheBlockBytes(64) ++
  new EE290BaseConfig
)

// chipyard/src/main/scala/atlas/config/AtlasConfigs.scala
package chipyard

import org.chipsalliance.cde.config.Config 
import atlas.config._

// Subunit-only configs (for isolated synthesis)

// Matrix unit with default params (16 lanes, vecLen=32, 1-cycle)
class InnerProductTreesDefaultConfig extends Config(
  new WithInnerProductTrees
)

class SystolicArrayDefaultConfig extends Config(
  new WithSystolicArray
)

// Full Atlas configs

class AtlasDefaultConfig extends Config(
  new WithInnerProductTrees
  // ++ new WithSystolicArray
  // ++ new WithVectorUnit
  // ++ new WithController
)

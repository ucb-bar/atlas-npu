// chipyard/src/main/scala/atlas/config/AtlasConfigs.scala
package chipyard

import org.chipsalliance.cde.config.Config 
import atlas.config._

// Subunit-only configs (for isolated synthesis)

// Matrix unit with default params (16 lanes, vecLen=32, 2-cycle)
class InnerProductTreesDefaultConfig extends Config(
  new WithInnerProductTrees
)

// Full Atlas configs

class AtlasDefaultConfig extends Config(
  new WithInnerProductTrees
  // ++ new WithVectorUnit
  // ++ new WithController
)

// chipyard/src/main/scala/atlas/config/AtlasKeys.scala
package atlas.config

import org.chipsalliance.cde.config.Field
import atlas.common.InnerProductTreeParams
import atlas.common.SystolicArrayParams
import atlas.common.DMAParams
import atlas.common.ScratchpadParams

// Per-subunit keys
case object InnerProductTreeKey extends Field[Option[InnerProductTreeParams]](None)
case object SystolicArrayKey extends Field[Option[SystolicArrayParams]](None)
case object DMAKey extends Field[Option[DMAParams]](None)
case object ScratchpadKey extends Field[Seq[ScratchpadParams]](Nil)


// Add more subunit keys here as Atlas grows:
// case object VectorUnitKey extends Field[Option[VectorUnitParams]](None)
// case object ControllerKey extends Field[Option[ControllerParams]](None)

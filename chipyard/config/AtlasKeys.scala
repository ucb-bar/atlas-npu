// chipyard/src/main/scala/atlas/config/AtlasKeys.scala
package atlas.config

import org.chipsalliance.cde.config.Field
import atlas.common.{InnerProductTreeParams, SystolicArrayParams, DmaParams, VmemParams}

// Per-subunit keys
case object InnerProductTreeKey extends Field[Option[InnerProductTreeParams]](None)
case object SystolicArrayKey extends Field[Option[SystolicArrayParams]](None)
case object DmaKey extends Field[Option[DmaParams]](None)
case object VmemKey extends Field[Seq[VmemParams]](Nil)


// Add more subunit keys here as Atlas grows:
// case object VectorUnitKey extends Field[Option[VectorUnitParams]](None)
// case object ControllerKey extends Field[Option[ControllerParams]](None)

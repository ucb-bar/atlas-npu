// ============================================================================
// HasPipelineParams.scala — helpers shared across the vector lane-box pipelines.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._

/** Small helpers shared across the vector lane-box pipelines. */
trait HasPipelineParams {
    def numFP16Lanes = 16
    def tagWidth = 16

    // Combiantional logic mask: select next value if (lane enabled and not back pressured), otherwise hold current value
    def maskLane[T <: Data](currVec: Vec[T], nxtVec: Vec[T], laneMask: Vec[Bool], bp: Bool): Vec[T] = {
        require(laneMask.length == currVec.length && currVec.length == nxtVec.length)
        VecInit(currVec.zip(nxtVec).zip(laneMask).map{case ((curr, nxt), en) => Mux(en && !bp, nxt, curr)})
    }

    // Sequential logic mask: update register with next value if (lane enabled and not back pressured), otherwise hold current value
    def maskLaneNext[T <: Data](bitsVec: Vec[T], laneMask: Vec[Bool], bp: Bool): Vec[T] = {
        require(laneMask.length == bitsVec.length)
        VecInit(bitsVec.zip(laneMask).map{case (bits, en) => RegEnable(bits, en && !bp)})
    }
}

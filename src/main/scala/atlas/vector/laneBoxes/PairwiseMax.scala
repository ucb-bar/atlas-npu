package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._

// Input bundles
class PairWiseMaxReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val isDoneReadingColMax = Bool() // For column reduction instructions, indicates when done reading all elements for a lane
    val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
    val bVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class PairWiseMaxResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class PairWiseMax(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with VectorParam {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new PairWiseMaxReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Valid(new PairWiseMaxResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    // Compute pairwise max for each lane
    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.valid).asUInt).asBools)
    val maxResults = VecInit(io.req.bits.aVec.zip(io.req.bits.bVec).zip(laneEnable).map { 
        case ((a, b), en) => Mux(en, compareReturnMax(a, b), 0.U) 
    })

    // Delay the valid bit
    val reqValid = RegInit(false.B)
    val outputLaneMask = RegInit(0.U(numLanes.W))
    val outputVec = RegInit(VecInit.fill(numLanes)(0.U(BF16T.wordWidth.W)))
    reqValid := io.req.valid
    outputLaneMask := io.req.bits.laneMask
    when (io.req.valid && !io.req.bits.isDoneReadingColMax) { outputVec := maxResults }

    // Output
    io.resp.valid := reqValid
    io.resp.bits.result := outputVec
    io.resp.bits.laneMask := outputLaneMask
}

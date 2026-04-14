package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._

// Input bundles
class SquareCubeReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val roundingMode = UInt(3.W)
    val isCube = Bool()
    val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class SquareCubeResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}
class SquareCubeVec(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with VectorParam {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new SquareCubeReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Valid(new SquareCubeResp(BF16T.wordWidth, numLanes, tagWidth))
    })
    
    // Setup
    // val isNeg = VecInit(io.req.bits.aVec.map(a => a(w-1)))          // (15)
    // val exp = VecInit(io.req.bits.aVec.map(a => a(w-2, sigW-1)))    // (14,7)
    // val mantissa = VecInit(io.req.bits.aVec.map(a => a(sigW-2, 0))) // (6,0)
    val w = BF16T.wordWidth
    val expW = BF16T.expWidth
    val sigW = BF16T.sigWidth

    // Compute results for square and cube if lane enabled, otherwise 0
    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.valid).asUInt).asBools)
    val result = VecInit(io.req.bits.aVec.zip(laneEnable).map {
        case (a, en) => Mux(en, squareCube(a(w-1), a(w-2, sigW-1), a(sigW-2, 0), io.req.bits.isCube), 0.U(w.W))
    })

    val isValid = RegNext(io.req.valid)
    val resultNext = RegNext(result)
    val laneMaskNext = RegNext(io.req.bits.laneMask)

    io.resp.valid := isValid
    io.resp.bits.laneMask := laneMaskNext
    io.resp.bits.result := resultNext
}

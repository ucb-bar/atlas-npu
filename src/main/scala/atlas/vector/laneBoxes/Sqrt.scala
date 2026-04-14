package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._

// Input bundles
class SqrtReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val roundingMode = UInt(3.W)
    val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class SqrtResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    // val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class Sqrt(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new SqrtReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Valid(new SqrtResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    val addrBits = BF16T.sigWidth - 1             // 7
    val mantissaTopBitIndex = BF16T.sigWidth - 2  // index of the top explicit mantissa bit (0-based from LSB) 6 
    val exponentTopBitIndex = BF16T.wordWidth - 2 // index of the top explicit exponent bit (0-based from LSB) 14
    val exponentLowBitIndex = BF16T.wordWidth - 9 // index of the low explicit exponent bit (0-based from LSB) 7
    val lut = Module(new SqrtLUT(numLanes, addrBits, BF16T.lutValM, BF16T.lutValN)) // 16, 7, 1, 12

    // Setting up the inputs to the LUT
    val lutInputExp = VecInit(io.req.bits.aVec.map(a => a(exponentTopBitIndex, exponentLowBitIndex))) // extract exponent bits for LUT input
    val oddExp = VecInit(io.req.bits.aVec.map(a => !a(exponentLowBitIndex)))
    val lutInputMant = VecInit(io.req.bits.aVec.map(a => a(mantissaTopBitIndex, 0)))                  // extract mantissa bits for LUT input
    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.valid).asUInt).asBools)

    // Connecting LUT inputs
    lut.io.exp := lutInputExp
    lut.io.oddExp := oddExp
    lut.io.raddr := lutInputMant
    lut.io.ren := laneEnable
    
    // Delay the valid bit
    val reqValid = RegInit(false.B)
    reqValid := io.req.valid

    // Output
    io.resp.valid := reqValid
    io.resp.bits.result := lut.io.rdata
}

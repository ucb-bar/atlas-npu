package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._

// Input bundles
class RowMinReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    // val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class RowMinResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    // val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class RowMin(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with VectorParam {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new RowMinReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Valid(new RowMinResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    val min8 = Wire(Vec(8, UInt(BF16T.wordWidth.W)) )
    for (i <- 0 until 8) {
        min8(i) := compareReturnMin(io.req.bits.aVec(2*i), io.req.bits.aVec(2*i + 1))
    }

    val min4 = Wire(Vec(4, UInt(BF16T.wordWidth.W)))
    for (i <- 0 until 4) {
        min4(i) := compareReturnMin(min8(2*i), min8(2*i + 1))
    }   

    val min2 = Wire(Vec(2, UInt(BF16T.wordWidth.W)))
    for (i <- 0 until 2) {
        min2(i) := compareReturnMin(min4(2*i), min4(2*i + 1))
    }

    val min1 = compareReturnMin(min2(0), min2(1))

    // Delay the valid bit
    val reqValid = RegInit(false.B)
    val min1Next = RegInit(0.U(BF16T.wordWidth.W))
    reqValid := io.req.valid
    min1Next := min1

    // Output
    io.resp.valid := reqValid
    io.resp.bits.result := VecInit.fill(numLanes)(min1Next)
}

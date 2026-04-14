package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._

// Input bundles
class RowMaxReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    // val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class RowMaxResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    // val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class RowMax(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with VectorParam {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new RowMaxReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Valid(new RowMaxResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    val max8 = Wire(Vec(8, UInt(BF16T.wordWidth.W)) )
    for (i <- 0 until 8) {
        max8(i) := compareReturnMax(io.req.bits.aVec(2*i), io.req.bits.aVec(2*i + 1))
    }

    val max4 = Wire(Vec(4, UInt(BF16T.wordWidth.W)))
    for (i <- 0 until 4) {
        max4(i) := compareReturnMax(max8(2*i), max8(2*i + 1))
    }   

    val max2 = Wire(Vec(2, UInt(BF16T.wordWidth.W)))
    for (i <- 0 until 2) {
        max2(i) := compareReturnMax(max4(2*i), max4(2*i + 1))
    }

    val max1 = compareReturnMax(max2(0), max2(1))

    // Delay the valid bit
    val reqValid = RegInit(false.B)
    val max1Next = RegInit(0.U(BF16T.wordWidth.W))
    reqValid := io.req.valid
    max1Next := max1

    // Output
    io.resp.valid := reqValid
    io.resp.bits.result := VecInit.fill(numLanes)(max1Next)
}

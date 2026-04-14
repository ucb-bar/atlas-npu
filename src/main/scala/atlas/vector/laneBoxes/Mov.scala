package atlas.vector

import chisel3._
import chisel3.util._

class MovReq(wordWidth: Int, numLanes: Int) extends Bundle {
    val laneMask = UInt(numLanes.W)
    val aVec     = Vec(numLanes, UInt(wordWidth.W))
}

class MovResp(wordWidth: Int, numLanes: Int) extends Bundle {
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class Mov(wordWidth: Int, numLanes: Int = 16) extends Module {
    val io = IO(new Bundle {
        val req  = Flipped(Valid(new MovReq(wordWidth, numLanes)))
        val resp = Valid(new MovResp(wordWidth, numLanes))
    })

    val resultReg = Reg(Vec(numLanes, UInt(wordWidth.W)))
    val validReg  = RegInit(false.B)

    when (io.req.valid) {
        resultReg := io.req.bits.aVec
    }
    validReg := io.req.valid

    io.resp.valid       := validReg
    io.resp.bits.result := resultReg
}

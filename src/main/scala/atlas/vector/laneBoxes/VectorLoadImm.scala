// ============================================================================
// VectorLoadImm.scala — Handles vli.All, vliRow, vli.Col, vli.One
// ============================================================================


package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._

class VLIReq(wordWidth: Int, numLanes: Int, rowIdxWidth: Int) extends Bundle {
    val op     = VPUOp()
    val imm    = SInt(16.W)
    val rowIdx = UInt(rowIdxWidth.W)
}

class VLIResp(wordWidth: Int, numLanes: Int) extends Bundle {
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class VectorLoadImm(BF16T: AtlasFPType, numLanes: Int = 16, rowIdxWidth: Int = 5)
    extends Module
    with HasPipelineParams {

    val io = IO(new Bundle {
        val req  = Flipped(Valid(new VLIReq(BF16T.wordWidth, numLanes, rowIdxWidth)))
        val resp = Valid(new VLIResp(BF16T.wordWidth, numLanes))
    })

    val outVec = Wire(Vec(numLanes, UInt(BF16T.wordWidth.W)))
    val immU   = io.req.bits.imm.asUInt
    val rowIdx =
        if (rowIdxWidth >= 5) io.req.bits.rowIdx(4, 0)
        else io.req.bits.rowIdx
    val isRow0 = (rowIdx === 0.U)

    for (i <- 0 until numLanes) {
        outVec(i) := 0.U
    }

    switch(io.req.bits.op) {
        is(VPUOp.vliAll) {
            for (i <- 0 until numLanes) {
                outVec(i) := immU
            }
        }

        is(VPUOp.vliRow) {
            when(isRow0) {
                for (i <- 0 until numLanes) {
                    outVec(i) := immU
                }
            }
        }

        is(VPUOp.vliCol) {
            outVec(0) := immU
        }

        is(VPUOp.vliOne) {
            when(isRow0) {
                outVec(0) := immU
            }
        }
    }

    io.resp.valid := io.req.valid
    io.resp.bits.result := outVec
}

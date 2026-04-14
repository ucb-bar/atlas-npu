package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._
import sp26FPUnits.hardfloat.consts._

// Input bundles
class ColAddReq(wordWidth: Int, numLanes: Int, tagWidth: Int, recWidth: Int) extends Bundle {
    val isDoneReadingColSum = Bool() // For column reduction instructions, indicates when done reading all elements for a lane
    val aVec = Vec(numLanes, UInt(wordWidth.W))
    val bVec = Vec(numLanes, UInt(recWidth.W))
}

// Output bundles
class ColAddResp(recWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val result = Vec(numLanes, UInt(recWidth.W))
}

class ColAddVec(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
    val computeSigWidth = BF16T.sigWidth + 16
    val recWidth = BF16T.expWidth + computeSigWidth + 1


    val io = IO(new Bundle {
        val req = Flipped(Valid(new ColAddReq(BF16T.wordWidth, numLanes, tagWidth, recWidth)))
        val resp = Valid(new ColAddResp(recWidth, numLanes, tagWidth))
    })

    // Widen BF16 precision by 16 extra significand bits.
    // Example: if BF16 sigWidth = 8, widened sigWidth = 24 (FP32-like precision).
    
    // ----------------------------------------------------------------------------
    // Input conversion:
    // raw BF16 bits -> BF16 recFN -> widened recFN
    // ----------------------------------------------------------------------------

    // Raw BF16 bits to BF16 recFN
    val aRecBF16 = VecInit(io.req.bits.aVec.map(a =>
        recFNFromFN(BF16T.expWidth, BF16T.sigWidth, a)
    ))

    // Widen BF16 recFN -> widened recFN
    val widenA = Seq.fill(numLanes) {
        Module(new RecFNToRecFN(
        BF16T.expWidth, BF16T.sigWidth,
        BF16T.expWidth, computeSigWidth
        ))
    }

    for (i <- 0 until numLanes) {
        widenA(i).io.in := aRecBF16(i)
        widenA(i).io.roundingMode := consts.round_near_even
        widenA(i).io.detectTininess := consts.tininess_afterRounding
    }

    val aRecVec = VecInit(widenA.map(_.io.out))
    val bRecVec = io.req.bits.bVec

    // ----------------------------------------------------------------------------
    // Arithmetic in widened recFN
    // ----------------------------------------------------------------------------

    
    val adders = Seq.fill(numLanes) {
        Module(new AddRecFN(BF16T.expWidth, computeSigWidth))
    }

    val addResult   = Wire(Vec(numLanes, UInt(recWidth.W)))

    // Stage 0
    for (i <- 0 until 16) {
        adders(i).io.subOp := false.B
        adders(i).io.a := aRecVec(i)
        adders(i).io.b := bRecVec(i)
        adders(i).io.roundingMode := consts.round_near_even
        adders(i).io.detectTininess := consts.tininess_afterRounding

        addResult(i) := adders(i).io.out
    }

    // ----------------------------------------------------------------------------
    // Stage 1 bookkeeping
    // ----------------------------------------------------------------------------
    val isValidReg = RegNext(io.req.valid, false.B)

    val addResultNext = RegInit(0.U.asTypeOf(chiselTypeOf(addResult)))
    when (io.req.valid && !io.req.bits.isDoneReadingColSum) {
        addResultNext := addResult
    }

    io.resp.valid := isValidReg
    io.resp.bits.result := addResultNext
}

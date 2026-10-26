package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._


// Input bundles
class AddSubSumReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val isSub = Bool()  // 1 = sub, 0 = add
    val aVec = Vec(numLanes, UInt(wordWidth.W))
    val bVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class AddSubSumResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class AddSubSumVec(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new AddSubSumReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Valid(new AddSubSumResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    // Adders
    val computeSigWidth = BF16T.sigWidth + 16
    val adders = Seq.fill(numLanes) {Module(new VectorAddRecFN(BF16T.expWidth, computeSigWidth))}
    
    // Stage 0: FN -> RecFN -> Rawfloat -> Add/Sub
    
    val aRecVecWiden = VecInit(io.req.bits.aVec.map(a => Cat(a, 0.U(16.W))))
    val bRecVecWiden = VecInit(io.req.bits.bVec.map(b => Cat(b, 0.U(16.W))))
    val aRecVec = VecInit(aRecVecWiden.map(a => recFNFromFN(BF16T.expWidth, computeSigWidth, a)))
    val bRecVec = VecInit(bRecVecWiden.map(b => recFNFromFN(BF16T.expWidth, computeSigWidth, b)))
    val addSubResult = Wire(Vec(numLanes, UInt((BF16T.expWidth + computeSigWidth + 1).W)))
    for (i <- 0 until numLanes) {
        adders(i).io.subOp := io.req.bits.isSub
        adders(i).io.a := aRecVec(i)
        adders(i).io.b := bRecVec(i)
        adders(i).io.roundingMode := 0.U(3.W)             // May change this later
        adders(i).io.detectTininess := false.B
        addSubResult(i) := adders(i).io.out
    }

    // Stage 1: Output and Conversion
    val isValidStage1 = RegNext(io.req.valid, false.B)
    val addSubResultNext = RegInit(0.U.asTypeOf(chiselTypeOf(addSubResult)))
    when (io.req.valid) {addSubResultNext := addSubResult}

    // Round the FP32 sum to BF16, nearest even. FP32 has more than 2 x 8 + 1 significand bits, so
    // the two roundings give BF16_RNE of the exact sum (Figueroa, "When is double rounding
    // innocuous?", 1995).
    val addSubOut = VecInit(addSubResultNext.map { v =>
        val toBF16 = Module(new RecFNToRecFN(BF16T.expWidth, computeSigWidth, BF16T.expWidth, BF16T.sigWidth))
        toBF16.io.in := v
        toBF16.io.roundingMode := round_near_even
        toBF16.io.detectTininess := tininess_afterRounding
        fNFromRecFN(BF16T.expWidth, BF16T.sigWidth, toBF16.io.out)
    })

    // Assigning outputs
    io.resp.valid := isValidStage1
    io.resp.bits.result := addSubOut
}

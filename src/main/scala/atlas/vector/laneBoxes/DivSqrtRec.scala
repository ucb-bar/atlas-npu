package atlas.vector

import chisel3._
import chisel3.util._
import fpex._
import sp26FPUnits._
import sp26FPUnits.hardfloat._

// Input bundles
class DivSqrtReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val roundingMode = UInt(3.W)
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val isSqrt = Bool() 
    val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
    val bVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class DivSqrtResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
    val outValid_div   = Vec(numLanes, Bool())
    val outValid_sqrt  = Vec(numLanes, Bool())
    val exceptionFlags = Vec(numLanes, UInt(5.W))
}

class DivSqrtRec(BF16T: FPType, numLanes: Int = 16, tagWidth: Int = 8) extends Module with HasPipelineParams {
    val io = IO(new Bundle {
        val req = Flipped(Decoupled(new DivSqrtReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Decoupled(new DivSqrtResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    // Initializing the modules
    val divSqrtModules = Seq.fill(numLanes)(Module(new DivSqrtRecFN_small(BF16T.expWidth, BF16T.sigWidth, 0))) 
    val ones = VecInit.fill(numLanes)("h3F80".U(BF16T.wordWidth.W))
    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.fire).asUInt).asBools)

    // Stage 0: FN -> RecFN
    val onesRec = VecInit(ones.map(i => recFNFromFN(BF16T.expWidth, BF16T.sigWidth, i)))
    val aRecVec = VecInit(io.req.bits.aVec.map(a => recFNFromFN(BF16T.expWidth, BF16T.sigWidth, a)))
    val bRecVec = VecInit(io.req.bits.bVec.map(b => recFNFromFN(BF16T.expWidth, BF16T.sigWidth, b)))

    // Stage 1: Computing
    for (i <- 0 until numLanes) {
        divSqrtModules(i).io.inValid := io.req.fire
        divSqrtModules(i).io.sqrtOp := io.req.bits.isSqrt
        divSqrtModules(i).io.a := Mux(io.req.bits.isSqrt, aRecVec(i), onesRec(i))
        divSqrtModules(i).io.b := Mux(io.req.bits.isSqrt, bRecVec(i), aRecVec(i))
        divSqrtModules(i).io.roundingMode := 0.U
        divSqrtModules(i).io.detectTininess := 0.U(1.W)
    }

    // Stage 2: Output and RecFN -> FN
    val outValid_div = Wire(Vec(numLanes, Bool()))
    val outValid_sqrt = Wire(Vec(numLanes, Bool()))
    val out = Wire(Vec(numLanes, UInt((BF16T.expWidth + BF16T.sigWidth + 1).W)))
    val exceptionsFlags = Wire(Vec(numLanes, UInt(5.W)))
    val allDivValid = outValid_div.reduce(_ && _)
    val allSqrtValid = outValid_sqrt.reduce(_ && _)
    val allReady = (0 until numLanes).map(i => divSqrtModules(i).io.inReady).reduce(_ && _)
    for (i <- 0 until numLanes) {
        outValid_div(i) := divSqrtModules(i).io.outValid_div
        outValid_sqrt(i) := divSqrtModules(i).io.outValid_sqrt
        out(i) := divSqrtModules(i).io.out
        exceptionsFlags(i) := divSqrtModules(i).io.exceptionFlags
    }
    val outRecoded = VecInit(out.map(v => fNFromRecFN(BF16T.expWidth, BF16T.sigWidth, v)))

    // Output
    io.req.ready := allReady
    io.resp.valid := allReady           // This should work, when all ready then all computation should have been done
    io.resp.bits.tag := 0.U             // Need to find a way to store the value since the module is variable in length
    io.resp.bits.whichBank := 0.U       // Need to fix later
    io.resp.bits.wRow := 0.U            // Need to fix later
    io.resp.bits.laneMask := 0xFFFF.U   // Need to find a way to store the value since the module is variable in length
    io.resp.bits.result := outRecoded
    io.resp.bits.outValid_div := outValid_div       // This is wrong, need to work on
    io.resp.bits.outValid_sqrt := outValid_sqrt     // This is worng, need to work on
    io.resp.bits.exceptionFlags := exceptionsFlags
}
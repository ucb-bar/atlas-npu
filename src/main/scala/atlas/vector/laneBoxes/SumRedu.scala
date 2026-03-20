package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._
import sp26FPUnits.hardfloat.consts._

// Input bundle for Reduction Sum
class SumReduReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val roundingMode = UInt(3.W)
    val laneMask = UInt(numLanes.W)
    val whichBank = UInt(4.W) 
    val wRow = UInt(8.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundle for Reduction Sum
class SumReduResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val laneMask = UInt(numLanes.W)
    val whichBank = UInt(4.W) 
    val wRow = UInt(8.W)      
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class ReduSumRec(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 8) extends Module with HasPipelineParams {
    require(isPow2(numLanes), "numLanes must be a power of 2")

    val io = IO(new Bundle {
        val req = Flipped(Decoupled(new SumReduReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Decoupled(new SumReduResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    val treeDepth = log2Ceil(numLanes)
    val numStages = treeDepth + 1 

    // --- TEXTBOOK DECOUPLED PIPELINE METADATA ---
    val valids    = RegInit(VecInit(Seq.fill(numStages)(false.B)))
    val readys    = Wire(Vec(numStages, Bool()))
    val tagRegs   = Seq.fill(numStages)(RegInit(0.U(tagWidth.W)))
    val rmRegs    = Seq.fill(numStages)(RegInit(0.U(3.W)))
    val maskRegs  = Seq.fill(numStages)(RegInit(0.U(numLanes.W)))
    val bankRegs  = Seq.fill(numStages)(RegInit(0.U(4.W)))
    val rowRegs   = Seq.fill(numStages)(RegInit(0.U(8.W)))

    // Skid Buffer logic
    readys(numStages - 1) := !valids(numStages - 1) || io.resp.ready
    for (i <- 0 until numStages - 1) {
        readys(i) := !valids(i) || readys(i+1)
    }

    // Stage 0: Entry
    when(readys(0)) {
        valids(0)   := io.req.valid
        tagRegs(0)  := io.req.bits.tag
        rmRegs(0)   := io.req.bits.roundingMode
        maskRegs(0) := io.req.bits.laneMask
        bankRegs(0) := io.req.bits.whichBank
        rowRegs(0)  := io.req.bits.wRow
    }

    // Stages 1 to N-1: Metadata Propagation
    for (i <- 1 until numStages) {
        when(readys(i)) {
            valids(i)   := valids(i-1)
            tagRegs(i)  := tagRegs(i-1)
            rmRegs(i)   := rmRegs(i-1)
            maskRegs(i) := maskRegs(i-1)
            bankRegs(i) := bankRegs(i-1)
            rowRegs(i)  := rowRegs(i-1)
        }
    }

// --- DATAPATH: PIPELINED BINARY TREE ---
    val intSigWidth = 24 // FP32 precision for intermediate sums

    val initialFN = Wire(Vec(numLanes, UInt(BF16T.wordWidth.W)))
    val initialFN_Widen = Wire(Vec(numLanes, UInt(32.W))) // 32-bit wide container

    for (i <- 0 until numLanes) {
        initialFN(i) := Mux(io.req.bits.laneMask(i), io.req.bits.aVec(i), 0.U(BF16T.wordWidth.W))
        // Pad BF16 (16 bits) to FP32 (32 bits) by appending 16 zeros to the LSBs
        initialFN_Widen(i) := Cat(initialFN(i), 0.U(16.W))
    }

    // Convert into a wider RecFN format (expWidth = 8, sigWidth = 24)
    val stg0RecVec = Wire(Vec(numLanes, UInt((BF16T.expWidth + intSigWidth + 1).W)))
    for (i <- 0 until numLanes) {
        stg0RecVec(i) := RegEnable(recFNFromFN(BF16T.expWidth, intSigWidth, initialFN_Widen(i)), readys(0))
    }

    var currentRecVec = stg0RecVec

    for (level <- 0 until treeDepth) {
        val nextVecSize = currentRecVec.length / 2
        val nextRecVec = Wire(Vec(nextVecSize, UInt((BF16T.expWidth + intSigWidth + 1).W)))
        val stageIdx = level + 1 

        for (i <- 0 until nextVecSize) {
            val leftRec  = currentRecVec(2 * i)
            val rightRec = currentRecVec(2 * i + 1)

            val leftRaw  = rawFloatFromRecFN(BF16T.expWidth, intSigWidth, leftRec)
            val rightRaw = rawFloatFromRecFN(BF16T.expWidth, intSigWidth, rightRec)

            // Instantiate Adder and Rounder with the wider internal precision
            val adder = Module(new AddRawFN1(BF16T.expWidth, intSigWidth))
            adder.io.subOp        := false.B
            adder.io.a            := leftRaw
            adder.io.b            := rightRaw
            adder.io.roundingMode := rmRegs(stageIdx - 1)

            val round = Module(new RoundRawFNToRecFN(BF16T.expWidth, intSigWidth, 0))
            round.io.invalidExc   := false.B 
            round.io.infiniteExc  := false.B
            round.io.in           := adder.io.rawOut
            round.io.roundingMode := rmRegs(stageIdx - 1)
            round.io.detectTininess := false.B

            nextRecVec(i) := RegEnable(round.io.out, readys(stageIdx))
        }
        currentRecVec = nextRecVec
    }

    val lastIdx = numStages - 1
    
    io.req.ready := readys(0)
    io.resp.valid := valids(lastIdx)
    io.resp.bits.tag := tagRegs(lastIdx)
    io.resp.bits.laneMask := maskRegs(lastIdx)
    io.resp.bits.whichBank := bankRegs(lastIdx)
    io.resp.bits.wRow := rowRegs(lastIdx)
    
    // Final Step: Round the 24-bit internal sum back down to BF16 (8-bit sigWidth)
    val finalRawWide = rawFloatFromRecFN(BF16T.expWidth, intSigWidth, currentRecVec(0))
    
    // Use RoundAnyRawFNToRecFN to handle the precision downcast safely
    val finalRounder = Module(new RoundAnyRawFNToRecFN(
        inExpWidth = BF16T.expWidth, 
        inSigWidth = intSigWidth, 
        outExpWidth = BF16T.expWidth, 
        outSigWidth = BF16T.sigWidth, 
        options = 0
    ))
    finalRounder.io.invalidExc   := false.B
    finalRounder.io.infiniteExc  := false.B
    finalRounder.io.in           := finalRawWide
    finalRounder.io.roundingMode := rmRegs(lastIdx)
    finalRounder.io.detectTininess := false.B

    // Formatting: Sum in Lane 0, Zeros in Lanes 1-15
    val finalSum = fNFromRecFN(BF16T.expWidth, BF16T.sigWidth, finalRounder.io.out)
    val zeroVal  = 0.U(BF16T.wordWidth.W)
    io.resp.bits.result := VecInit(Seq(finalSum) ++ Seq.fill(numLanes - 1)(zeroVal)) 
}
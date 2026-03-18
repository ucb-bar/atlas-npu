package atlas.vector

import chisel3._
import chisel3.util._
import fpex._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._


// Input bundles
class MulReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val roundingMode = UInt(3.W)
    val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
    val bVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class MulResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class MulRec(BF16T: FPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
    val io = IO(new Bundle {
        val req = Flipped(Decoupled(new MulReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Decoupled(new MulResp(BF16T.wordWidth, numLanes, tagWidth))
    })
  
    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.fire).asUInt).asBools)

    // States for pipelining
    class CommonStageState extends Bundle {
        val valid = Bool()
        val req = chiselTypeOf(io.req.bits)
        val laneEn = chiselTypeOf(laneEnable)
    }

    // Initializing the states
    def numIntermediateStages = 2
    val commonState = RegInit(VecInit(Seq.fill(numIntermediateStages)(0.U.asTypeOf(new CommonStageState))))
    val backPressure = Wire(Vec(numIntermediateStages, Bool()))
    val stateWithBp = commonState.zip(backPressure)
    def st(i: Int) = commonState(i-1)
    def bp(i: Int) = backPressure(i-1)

    // Assigning values to the 0th stage of the pipeline: select between input and holding value based on backpressure.
    stateWithBp.take(1).foreach { case (state, back) =>
        state.valid := Mux(!back, io.req.fire, state.valid)
        state.req := Mux(io.req.fire, io.req.bits, state.req)
        state.laneEn := Mux(io.req.fire, laneEnable, state.laneEn)
    }

    // Assigning values to the rest of the stages: if not back pressured, take value from previous stage; if back pressured, hold current value.
    stateWithBp.zipWithIndex.takeRight(numIntermediateStages - 1).foreach {
        case ((state, back), i) =>
        state.valid := Mux(!back, st(i).valid, state.valid)
        state.req := Mux(st(i).valid && !back, st(i).req, state.req)
        state.laneEn := Mux(st(i).valid && !back, st(i).laneEn, state.laneEn)
    }

    /* Assigning backpressure: 
     * Last stage is back pressured if response is not ready.
     * Current stage is back pressured if it is valid and the next stage is back pressured. 
     */ 
    backPressure(numIntermediateStages-1) := !io.resp.ready
    backPressure.zip(commonState).zipWithIndex.take(numIntermediateStages-1).foreach { case ((bp, st), i) =>
        bp := st.valid && backPressure(i+1)
    }

    // Stage 0: FN -> RecFN -> Rawfloat
    val aRecVec = VecInit(io.req.bits.aVec.map(a => recFNFromFN(BF16T.expWidth, BF16T.sigWidth, a)))
    val bRecVec = VecInit(io.req.bits.bVec.map(b => recFNFromFN(BF16T.expWidth, BF16T.sigWidth, b)))
    val aRecVecRaw = VecInit(aRecVec.map(a => rawFloatFromRecFN(BF16T.expWidth, BF16T.sigWidth, a)))
    val bRecVecRaw = VecInit(bRecVec.map(b => rawFloatFromRecFN(BF16T.expWidth, BF16T.sigWidth, b)))

    // Stage 1: Rawfloat mul
    val aRecVecNext = maskLaneNext(aRecVecRaw, laneEnable, bp = bp(1)) // No backpressure for now
    val bRecVecNext = maskLaneNext(bRecVecRaw, laneEnable, bp = bp(1)) // No backpressure for now
    // val multipliers = Seq.fill(numLanes) {Module(new MulRecFN(BF16T.expWidth, BF16T.sigWidth))}
    // multipliers.zipWithIndex.foreach { case (mul, i) =>
    //     mul.io.a := aRecVecNext(i)
    //     mul.io.b := bRecVecNext(i)
    //     mul.io.roundingMode := 0.U(3.W)
    //     mul.io.detectTininess := false.B
    //     productRec(i) := mul.io.out
    // }
    val multipliers = Seq.fill(numLanes) {Module(new MulRawFN(BF16T.expWidth, BF16T.sigWidth))}
    val invalidExc = Wire(Vec(numLanes, Output(Bool())))
    val productRec = Wire(Vec(numLanes, new RawFloat(BF16T.expWidth, BF16T.sigWidth + 2)))
    multipliers.zipWithIndex.foreach { case (mul, i) =>
        mul.io.a := aRecVecNext(i)
        mul.io.b := bRecVecNext(i)
        invalidExc(i) := mul.io.invalidExc
        productRec(i) := mul.io.rawOut
    }

    // Stage 2: Raw -> RecFN -> FN
    val productRecNext = maskLaneNext(productRec, st(1).laneEn, bp = bp(2))
    val invalidExcNext = maskLaneNext(invalidExc, st(1).laneEn, bp = bp(2))
    val productRecNextRec = Wire(Vec(numLanes, UInt((BF16T.expWidth + BF16T.sigWidth + 1).W)))
    val roundRawFNToRecFN =Seq.fill(numLanes) {Module(new RoundRawFNToRecFN(BF16T.expWidth, BF16T.sigWidth, 0))}
    roundRawFNToRecFN.zipWithIndex.foreach {case (round, i) => 
        round.io.invalidExc     := invalidExcNext(i)
        round.io.infiniteExc    := false.B
        round.io.in             := productRecNext(i)
        round.io.roundingMode   := 0.U(3.W) // Change this later
        round.io.detectTininess := false.B
        productRecNextRec(i)        := round.io.out
        // io.exceptionFlags := round.io.exceptionFlags.    Don't need this right now
    }
    val resValid = st(2).valid
    val resReq = st(2).req
    val outVec = VecInit(productRecNextRec.map(v => fNFromRecFN(BF16T.expWidth, BF16T.sigWidth, v)))

    // Output
    io.req.ready := !backPressure(0)
    io.resp.valid := resValid
    io.resp.bits.tag := resReq.tag
    io.resp.bits.whichBank := resReq.whichBank
    io.resp.bits.wRow := resReq.wRow
    io.resp.bits.result := outVec
    io.resp.bits.laneMask := resReq.laneMask
}

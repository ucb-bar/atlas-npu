package atlas.vector

import chisel3._
import chisel3.util._
import fpex._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._

trait HasPipelineParams {
    def numFP16Lanes = 16
    def tagWidth = 16

    // Combiantional logic mask: select next value if (lane enabled and not back pressured), otherwise hold current value
    def maskLane[T <: Data](currVec: Vec[T], nxtVec: Vec[T], laneMask: Vec[Bool], bp: Bool): Vec[T] = {
        require(laneMask.length == currVec.length && currVec.length == nxtVec.length)
        VecInit(currVec.zip(nxtVec).zip(laneMask).map{case ((curr, nxt), en) => Mux(en && !bp, nxt, curr)})
    }

    // Sequential logic mask: update register with next value if (lane enabled and not back pressured), otherwise hold current value
    def maskLaneNext[T <: Data](bitsVec: Vec[T], laneMask: Vec[Bool], bp: Bool): Vec[T] = {
        require(laneMask.length == bitsVec.length)
        VecInit(bitsVec.zip(laneMask).map{case (bits, en) => RegEnable(bits, en && !bp)})
    }
}

// Input bundles
class AddSubReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val roundingMode = UInt(3.W)
    val sub = Bool() // 1 = sub, 0 = add
    val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
    val bVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class AddSubResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class AddSubRec(BF16T: FPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
    val io = IO(new Bundle {
        val req = Flipped(Decoupled(new AddSubReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Decoupled(new AddSubResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.fire).asUInt).asBools)

    // States for pipelining
    class CommonStageState extends Bundle {
        val valid = Bool()
        val req = chiselTypeOf(io.req.bits)
        val laneEn = chiselTypeOf(laneEnable)
        val isSub = Bool()
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
        state.isSub := Mux(io.req.fire, io.req.bits.sub, state.isSub)
    }

    // Assigning values to the rest of the stages: if not back pressured, take value from previous stage; if back pressured, hold current value.
    stateWithBp.zipWithIndex.takeRight(numIntermediateStages - 1).foreach {
        case ((state, back), i) =>
        state.valid := Mux(!back, st(i).valid, state.valid)
        state.req := Mux(st(i).valid && !back, st(i).req, state.req)
        state.laneEn := Mux(st(i).valid && !back, st(i).laneEn, state.laneEn)
        state.isSub := Mux(st(i).valid && !back, st(i).isSub, state.isSub)
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

    // Stage 1: Rawfloat add/sub
    val aRecVecNext = maskLaneNext(aRecVecRaw, laneEnable, bp = bp(1)) // No backpressure for now
    val bRecVecNext = maskLaneNext(bRecVecRaw, laneEnable, bp = bp(1)) // No backpressure for now
    // val adders = Seq.fill(numLanes) {Module(new VectorAddRecFN(BF16T.expWidth, BF16T.sigWidth))}
    // adders.zipWithIndex.foreach { case (add, i) =>
    //     add.io.subOp := st(1).isSub
    //     add.io.a := aRecVecNext(i)
    //     add.io.b := bRecVecNext(i)
    //     add.io.roundingMode := 0.U(3.W)
    //     add.io.detectTininess := false.B
    //     sumRec(i) := add.io.out
    // }
    val adders = Seq.fill(numLanes) {Module(new AddRawFN1(BF16T.expWidth, BF16T.sigWidth))}
    val invalidExc = Wire(Vec(numLanes, Output(Bool())))
    val sumRec = Wire(Vec(numLanes, new RawFloat(BF16T.expWidth, BF16T.sigWidth + 2)))
    adders.zipWithIndex.foreach { case (add, i) =>
        add.io.subOp := st(1).isSub
        add.io.a := aRecVecNext(i)
        add.io.b := bRecVecNext(i)
        add.io.roundingMode := 0.U(3.W) // Change this later
        invalidExc(i) := add.io.invalidExc
        sumRec(i) := add.io.rawOut
    }

    // Stage 2: Raw -> RecFN -> FN
    val sumRecNext = maskLaneNext(sumRec, st(1).laneEn, bp = bp(2))
    val invalidExcNext = maskLaneNext(invalidExc, st(1).laneEn, bp = bp(2))
    val sumRecNextRec = Wire(Vec(numLanes, UInt((BF16T.expWidth + BF16T.sigWidth + 1).W)))
    val roundRawFNToRecFN =Seq.fill(numLanes) {Module(new RoundRawFNToRecFN(BF16T.expWidth, BF16T.sigWidth, 0))}
    roundRawFNToRecFN.zipWithIndex.foreach {case (round, i) => 
        round.io.invalidExc     := invalidExcNext(i)
        round.io.infiniteExc    := false.B
        round.io.in             := sumRecNext(i)
        round.io.roundingMode   := 0.U(3.W) // Change this later
        round.io.detectTininess := false.B
        sumRecNextRec(i)        := round.io.out
        // io.exceptionFlags := round.io.exceptionFlags.    Don't need this right now
    }
    val resValid = st(2).valid
    val resReq = st(2).req
    val outVec = VecInit(sumRecNextRec.map(v => fNFromRecFN(BF16T.expWidth, BF16T.sigWidth, v)))


    // Output
    io.req.ready := !backPressure(0)
    io.resp.valid := resValid
    io.resp.bits.tag := resReq.tag
    io.resp.bits.whichBank := resReq.whichBank
    io.resp.bits.wRow := resReq.wRow
    io.resp.bits.result := outVec
    io.resp.bits.laneMask := resReq.laneMask
}

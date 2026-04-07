// ============================================================================
// SquareCubeRec.scala — Vector square / cube lane box.
//
// Reuses a small multiply pipeline to compute either x^2 or x^3 lane-wise and
// packages the resulting vector back into a Decoupled response.
// ============================================================================

package atlas.vector


import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._      
import sp26FPUnits.hardfloat.consts._

// Input bundles
class SquareCubeReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val roundingMode = UInt(3.W)
    val laneMask = UInt(numLanes.W)
    val aVec = Vec(numLanes, UInt(wordWidth.W))
    val isCube = Bool()
}

// Output bundles
class SquareCubeResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}

/** Vector square / cube pipeline.
  *
  * @param BF16T     BF16 format descriptor.
  * @param numLanes  Number of parallel vector lanes.
  * @param tagWidth  Width of the forwarded metadata tag.
  */
class SquareCubeRec(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
    val w = BF16T.wordWidth
    val expW = BF16T.expWidth
    val sigW = BF16T.sigWidth
    val io = IO(new Bundle {
        val req = Flipped(Decoupled(new SquareCubeReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Decoupled(new SquareCubeResp(BF16T.wordWidth, numLanes, tagWidth))
    })
    
    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.fire).asUInt).asBools)
    
    // We repeat RUN for cube
    object SCState extends ChiselEnum { 
        val IDLE, RUN, DONE = Value 
    }

    val state = RegInit(SCState.IDLE)

    // phase = 0 => doing a*a, phase = 1 => doing tmp*a
    val phase  = RegInit(0.U(2.W))

    // States for pipelining
    class CommonStageState extends Bundle {
        val valid = Bool()
        val req = chiselTypeOf(io.req.bits)
        val laneEn = chiselTypeOf(laneEnable)
    }

    // Initializing the states
    //def numIntermediateStages = 1
    //val commonState = RegInit(VecInit(Seq.fill(numIntermediateStages)(0.U.asTypeOf(new CommonStageState))))
    val commonState = RegInit(0.U.asTypeOf(new CommonStageState))
    
    // our original input in raw format so we don't have to recode multiple times
    val aRawReg = Reg(Vec(numLanes, new RawFloat(expW, sigW)))

    // holds current raw multiply result
    val squareRawReg = Reg(Vec(numLanes, new RawFloat(expW, sigW + 2)))

    // final output (FN)
    val outReg = Reg(Vec(numLanes, UInt(w.W)))

    // Stage 0: FN -> RecFN -> Rawfloat
    val aRecVec = VecInit(io.req.bits.aVec.map(a => recFNFromFN(BF16T.expWidth, BF16T.sigWidth, a)))
    val aRecVecRaw = VecInit(aRecVec.map(a => rawFloatFromRecFN(BF16T.expWidth, BF16T.sigWidth, a))) 

    // Stage 1: a * a (like squared)
    val squareMul = Seq.fill(numLanes) { Module(new MulRawFN(expW, sigW)) }
    
    val squareRawOut = Wire(Vec(numLanes, new RawFloat(expW, sigW + 2)))

    squareMul.zipWithIndex.foreach { case (mul, i) =>
        mul.io.a := aRawReg(i)
        mul.io.b := aRawReg(i)
        squareRawOut(i) := mul.io.rawOut
    }

    // for cube: must actually round 
    // raw (a * a) -> Recfn -> raw (inspo from sumredu??)
    val squareRound = Seq.fill(numLanes) { Module(new RoundRawFNToRecFN(expW, sigW, 0)) }
    val squareRec = Wire(Vec(numLanes, UInt((expW + sigW + 1).W)))
    val squareRaw = Wire(Vec(numLanes, new RawFloat(expW, sigW)))

    squareRound.zipWithIndex.foreach { case (round, i) =>
        round.io.invalidExc     := false.B
        round.io.infiniteExc    := false.B
        round.io.in             := squareRawReg(i)
        round.io.roundingMode   := commonState.req.roundingMode
        round.io.detectTininess := false.B
        squareRec(i) := round.io.out
        squareRaw(i) := rawFloatFromRecFN(expW, sigW, squareRec(i))
    }


    // only for cube: a^2 * a 
    val cubeMul = Seq.fill(numLanes) { Module(new MulRawFN(expW, sigW)) }
    val cubeOut = Wire(Vec(numLanes, new RawFloat(expW, sigW + 2)))

    cubeMul.zipWithIndex.foreach { case (mul, i) =>
        mul.io.a := squareRaw(i)
        mul.io.b := aRawReg(i)
        cubeOut(i) := mul.io.rawOut
    }


    // convert: raw -> recfn->fn
    val finalRound = Seq.fill(numLanes) { Module(new RoundRawFNToRecFN(expW, sigW, 0)) }
    val finalRec = Wire(Vec(numLanes, UInt((expW + sigW + 1).W)))
    val finalFn = Wire(Vec(numLanes, UInt(w.W)))

    finalRound.zipWithIndex.foreach { case (round, i) =>
        round.io.invalidExc     := false.B
        round.io.infiniteExc    := false.B
        round.io.in             := squareRawReg(i)
        round.io.roundingMode   := commonState.req.roundingMode
        round.io.detectTininess := false.B
        finalRec(i) := round.io.out
        finalFn(i) := fNFromRecFN(expW, sigW, finalRec(i))
    }


    // default
    io.req.ready := (state === SCState.IDLE)
    io.resp.valid := (state === SCState.DONE)

    io.resp.bits.tag := commonState.req.tag
    io.resp.bits.whichBank := commonState.req.whichBank
    io.resp.bits.wRow := commonState.req.wRow
    io.resp.bits.laneMask := commonState.req.laneMask
    io.resp.bits.result := outReg


    // FSM LOGIC:
    // getting actual req 
    when (state === SCState.IDLE) {
        commonState.valid := false.B

        when (io.req.fire) {
            commonState.valid := true.B
            commonState.req := io.req.bits
            commonState.laneEn := laneEnable
            phase := 0.U

            for (i <- 0 until numLanes) {
                aRawReg(i) := aRecVecRaw(i)
            }

            state := SCState.RUN
        }
    }

    // Output when done -> we have this state instead of going straight back to idle since we don't want to accept a new inst if we are still holding old value
    when (state === SCState.DONE) {
        when (io.resp.fire) {
            state := SCState.IDLE
        }
    }

    // RUN: issue request (once) then wait for response
     when (state === SCState.RUN) {
        when (phase === 0.U) {
            // phase 0: compute and register a*a
            for (i <- 0 until numLanes) {
                when (commonState.req.laneMask(i).asBool) {
                    squareRawReg(i) := squareRawOut(i)
                }
            }
            phase := 1.U

        }.elsewhen (phase === 1.U) {
            when (!commonState.req.isCube) {
                // square: finalize
                for (i <- 0 until numLanes) {
                    when (commonState.req.laneMask(i).asBool) {
                        outReg(i) := finalFn(i)
                    }.otherwise {
                        outReg(i) := 0.U
                    }
                }
                state := SCState.DONE

            }.otherwise {
                // cube: compute and register (rounded a*a) * a
                for (i <- 0 until numLanes) {
                    when (commonState.req.laneMask(i).asBool) {
                        squareRawReg(i) := cubeOut(i)
                    }
                }
                phase := 2.U
            }

        }.otherwise {
            // phase 2: finalize cube
            for (i <- 0 until numLanes) {
                when (commonState.req.laneMask(i).asBool) {
                    outReg(i) := finalFn(i)
                }.otherwise {
                    outReg(i) := 0.U
                }
            }
            state := SCState.DONE
        }
    }
}

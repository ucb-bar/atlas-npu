// ============================================================================
// ExpLane.scala — Vector exp / exp2 lane box.
//
// Uses lookup-table-assisted per-lane exponentiation and forwards the vector
// result through a valid-only response interface.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._

class FPEXReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
  val isBase2 = Bool() // 0 = exp, 1 = 2^x,  Added in addition to original exp
  val laneMask = UInt(numLanes.W)
  val xVec = Vec(numLanes, UInt(wordWidth.W))
}

class FPEXResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
  // val laneMask = UInt(numLanes.W)
  val result = Vec(numLanes, UInt(wordWidth.W))
}

/** Vector exp / exp2 pipeline.
  *
  * @param fpT       Floating-point format descriptor.
  * @param numLanes  Number of parallel vector lanes.
  * @param tagWidth  Width of the forwarded metadata tag.
  */
class Exp(fpT: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
  val io = IO(new Bundle {
    val req = Flipped(Valid(new FPEXReq(fpT.wordWidth, numLanes, tagWidth)))
    val resp = Valid(new FPEXResp(fpT.wordWidth, numLanes, tagWidth))
  })

  // Setup
  val rLowBits = fpT.qmnN - fpT.lutAddrBits
  val lutTopEndpoint = ((BigInt(1) << (fpT.lutValM + fpT.lutValN)) - 1).U((fpT.lutValM + fpT.lutValN).W)
  val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.valid).asUInt).asBools)
  val lut = Module(new ExLUT(numLanes, fpT.lutAddrBits, fpT.lutValM, fpT.lutValN))
  val roundToRecFn = Seq.fill(numLanes)(Module(new RoundRawFNToRecFN(fpT.expWidth, fpT.sigWidth, 0)))

  // Convert to raw float and check for special cases
  val rawFloatVec = VecInit(io.req.bits.xVec.map(x => rawFloatFromFN(fpT.expWidth, fpT.sigWidth, x)))
  val expFPOverflow = VecInit(io.req.bits.xVec.map(x => fpT.expFPIsInf(x, false.B)))
  val rawFloatOf = rawFloatVec.zip(expFPOverflow)
  val earlyResult = VecInit(rawFloatOf.map { case (x, of) =>
    MuxCase(
      0.U(fpT.wordWidth.W),
      Seq(
        x.isNaN -> Cat(x.sign, fpT.nanExp, isSigNaNRawFloat(x), fpT.nanSig),
        (x.isZero) -> fpT.one,  // .subNorm is not a function
        (x.isInf && x.sign) -> fpT.zero,
        ((x.isInf && !x.sign) || of) -> Cat(0.U(1.W), fpT.infinity)
      )
    )
  })
  val earlyTerminate = VecInit(rawFloatOf.map { case (x, of) =>x.isInf || x.isZero || x.isNaN || of})
  
  // Convert to qmn format and compute LUT addresses
  val qmnVec = VecInit(rawFloatVec.map(x => fpT.qmnFromRawFloat(x)))
  val expKRVec = qmnVec.map(_.mul(fpT.rln2).getKR).unzip          
  val exp2KRVec = qmnVec.map(_.getKR).unzip                             // Added in addition to original exp
  val kVec = VecInit(expKRVec._1.zip(exp2KRVec._1).map{ case (kLn2, kBase2) => Mux(io.req.bits.isBase2, kBase2, kLn2) }) // Added in addition to original exp
  val rVec = VecInit(expKRVec._2.zip(exp2KRVec._2).map{ case (rLn2, rBase2) => Mux(io.req.bits.isBase2, rBase2, rLn2) }) // Added in addition to original exp

  class CommonStageState extends Bundle {
    val valid = Bool()
    // val req = chiselTypeOf(io.req.bits)
    // val laneEn = chiselTypeOf(laneEnable)
    val earlyTerm = chiselTypeOf(earlyTerminate)
    val earlyRes = chiselTypeOf(earlyResult)
  }

  def numIntermediateStages = 1
  val commonState = RegInit(VecInit(Seq.fill(numIntermediateStages)(0.U.asTypeOf(new CommonStageState))))
  val backPressure = WireInit(VecInit(Seq.fill(numIntermediateStages)(false.B)))
  val stateWithBp = commonState.zip(backPressure)
  def st(i: Int) = commonState(i-1)
  def bp(i: Int) = backPressure(i-1)
  stateWithBp.take(1).foreach { case (state, back) =>
      state.valid := Mux(!back, io.req.valid, state.valid)
      // state.req := Mux(io.req.valid, io.req.bits, state.req)
      // state.laneEn := Mux(!back, laneEnable, state.laneEn)
      state.earlyTerm := maskLane(state.earlyTerm, earlyTerminate, laneEnable, back)
      state.earlyRes := maskLane(state.earlyRes, earlyResult, laneEnable, back)
  }

  // stateWithBp.zipWithIndex.takeRight(numIntermediateStages - 1).foreach {
  //   case ((state, back), i) =>
  //     state.valid := Mux(!back, st(i).valid, state.valid)
  //     state.req := Mux(st(i).valid && !back, st(i).req, state.req)
  //     // state.laneEn := Mux(!back, st(i).laneEn, state.laneEn)
  //     state.earlyTerm := maskLane(state.earlyTerm, st(i).earlyTerm, st(i).laneEn, back)
  //     state.earlyRes := maskLane(state.earlyRes, st(i).earlyRes, st(i).laneEn, back)
  // }

  // Interpolation once received LUT output
  val kVecNext = maskLaneNext(kVec, laneEnable, bp(1))
  val rVecNext = maskLaneNext(rVec, laneEnable, bp(1))
  val addrVec = VecInit(rVecNext.map(r => r(fpT.qmnN - 1, rLowBits)))
  val rLowerVec = VecInit(rVecNext.map(r => r(rLowBits - 1, 0)))
  val lutOutput0 = lut.io.rdata(0)
  val lutOutput1 = lut.io.rdata(1)
  val delta = VecInit(lutOutput0.zip(lutOutput1).zip(addrVec).map {
    case ((y0, y1), addr) =>
      val y1Interp = Mux(addr === ((1 << fpT.lutAddrBits) - 1).U, lutTopEndpoint, y1)
      val delta = y1Interp - y0
      delta
  })
  val deltaFrac = VecInit(delta.zip(rLowerVec).map {case (delta, frac) => (delta * frac) >> rLowBits})
  val interpResult = VecInit(lutOutput0.zip(deltaFrac).map {case (y0, deltaFrac) => y0 + deltaFrac})

  // Output Conversion and rounding
  val resRawFloat = interpResult.zip(kVecNext).map{ case (qmn, k) => fpT.rawFloatFromQmnK(qmn, k) }
  roundToRecFn.zip(resRawFloat).foreach {
    case (round, rawFloat) => {
      round.io.invalidExc := false.B
      round.io.infiniteExc := false.B
      round.io.in := rawFloat
      round.io.roundingMode := 0.U
      round.io.detectTininess := 0.U
    }
  }
  val resValid = st(1).valid
  // val resReq = st(1).req
  val resFN = roundToRecFn.map(round => fNFromRecFN(fpT.expWidth, fpT.sigWidth, round.io.out))
  val resFinal = VecInit(resFN.zip(st(1).earlyTerm.zip(st(1).earlyRes)).map {
    case (res, (earlyTerminate, earlyRes)) => Mux(earlyTerminate, earlyRes, res)
  })

  lut.io.raddr := rVec.map(r => r(fpT.qmnN - 1, rLowBits))
  lut.io.ren := VecInit(laneEnable.map(en => en && io.req.valid))

  io.resp.valid := resValid
  io.resp.bits.result := resFinal
  // io.resp.bits.laneMask := resReq.laneMask
}

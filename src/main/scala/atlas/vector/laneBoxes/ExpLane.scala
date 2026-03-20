package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._

class FPEXReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
  val roundingMode = UInt(3.W)
  val tag = UInt(tagWidth.W)
  val whichBank = UInt(5.W)
  val wRow= UInt(7.W)
  val neg = Bool() // 0 = e^x, 1 = e^-x >>> No longer needed
  val isBase2 = Bool() // 0 = exp, 1 = 2^x,  Added in addition to original exp
  val laneMask = UInt(numLanes.W)
  val xVec = Vec(numLanes, UInt(wordWidth.W))
}

class FPEXResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
  val tag = UInt(tagWidth.W)
  val whichBank = UInt(5.W)
  val wRow= UInt(7.W)
  val laneMask = UInt(numLanes.W)
  val result = Vec(numLanes, UInt(wordWidth.W))
}

class Exp(fpT: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new FPEXReq(fpT.wordWidth, numLanes, tagWidth)))
    val resp = Decoupled(new FPEXResp(fpT.wordWidth, numLanes, tagWidth))
  })

  val lut = Module(new ExLUT(numLanes, fpT.lutAddrBits, fpT.lutValM, fpT.lutValN))
  val roundToRecFn = Seq.fill(numLanes)(Module(new RoundRawFNToRecFN(fpT.expWidth, fpT.sigWidth, 0)))
  val rLowBits = fpT.qmnN - fpT.lutAddrBits
  val lutTopEndpoint = ((BigInt(1) << (fpT.lutValM + fpT.lutValN)) - 1).U((fpT.lutValM + fpT.lutValN).W)
  val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.fire).asUInt).asBools)
  val maskedXVec = io.req.bits.xVec.zip(laneEnable).map {
    case (x, en) => Mux (en, x, 0.U(fpT.wordWidth.W))
  }
  val maskedIsBase2 = io.req.bits.isBase2 && io.req.fire      // Added in addition to original exp

  // Stage 0: Rawfloat decomposition and special case check
  val rawFloatVec = VecInit(maskedXVec.map(
    x => rawFloatFromFN(fpT.expWidth, fpT.sigWidth, x)))
  val expFPOverflow = VecInit(maskedXVec.map(x => fpT.expFPIsInf(x, false.B)))
  val rawFloatOf = rawFloatVec.zip(expFPOverflow)
  val earlyResult = VecInit(rawFloatOf.map { case (x, of) =>
    MuxCase(
      0.U(fpT.wordWidth.W),
      Seq(
        x.isNaN -> Cat(x.sign, fpT.nanExp, isSigNaNRawFloat(x), fpT.nanSig),
        //(x.isZero || x.isSubNorm) -> fpT.one,
        (x.isZero) -> fpT.one,
        (x.isInf && x.sign) -> fpT.zero,
        ((x.isInf && !x.sign) || of) -> Cat(0.U(1.W), fpT.infinity)
      )
    )
  })
  val earlyTerminate = VecInit(rawFloatOf.map { case (x, of) =>
    x.isInf || x.isZero || x.isNaN || of
  })

  class CommonStageState extends Bundle {
    val valid = Bool()
    val req = chiselTypeOf(io.req.bits)
    val laneEn = chiselTypeOf(laneEnable)
    val earlyTerm = chiselTypeOf(earlyTerminate)
    val earlyRes = chiselTypeOf(earlyResult)
  }

  def numIntermediateStages = 6
  val commonState = RegInit(VecInit(Seq.fill(numIntermediateStages)(0.U.asTypeOf(new CommonStageState))))
  val backPressure = Wire(Vec(numIntermediateStages, Bool()))
  val stateWithBp = commonState.zip(backPressure)
  def st(i: Int) = commonState(i-1)
  def bp(i: Int) = backPressure(i-1)
  stateWithBp.take(1).foreach { case (state, back) =>
      state.valid := Mux(!back, io.req.fire, state.valid)
      state.req := Mux(io.req.fire, io.req.bits, state.req)
      state.laneEn := Mux(!back, laneEnable, state.laneEn)
      state.earlyTerm := maskLane(state.earlyTerm, earlyTerminate, laneEnable, back)
      state.earlyRes := maskLane(state.earlyRes, earlyResult, laneEnable, back)
  }

  stateWithBp.zipWithIndex.takeRight(numIntermediateStages - 1).foreach {
    case ((state, back), i) =>
      state.valid := Mux(!back, st(i).valid, state.valid)
      state.req := Mux(st(i).valid && !back, st(i).req, state.req)
      state.laneEn := Mux(!back, st(i).laneEn, state.laneEn)
      state.earlyTerm := maskLane(state.earlyTerm, st(i).earlyTerm, st(i).laneEn, back)
      state.earlyRes := maskLane(state.earlyRes, st(i).earlyRes, st(i).laneEn, back)
  }

  backPressure(numIntermediateStages-1) := !io.resp.ready
  backPressure.zip(commonState).zipWithIndex.take(numIntermediateStages-1).foreach { case ((bp, st), i) =>
      bp := st.valid && backPressure(i+1)
  }

  // stage 1: convert to Qmn
  val stage1RawFloatVec = maskLaneNext(rawFloatVec, laneEnable, bp(1))
  val stage1Qmn = VecInit(stage1RawFloatVec.map(x => fpT.qmnFromRawFloat(x)))
  val stage1IsBase2 = RegEnable(maskedIsBase2, io.req.fire)               // Added in addition to original exp

  // Stage 2: multiply x/ln2, extract k and r, init lut read r[top] and r[top] + 1
  val stage2Qmn = maskLaneNext(stage1Qmn, st(1).laneEn, bp(2))
  val stage2IsBase2 = RegEnable(stage1IsBase2, st(1).valid && !bp(2))    // Added in addition to original exp
  val xrln2KRVec = stage2Qmn.map(_.mul(fpT.rln2).getKR).unzip          
  val xrKRVec = stage2Qmn.map(_.getKR).unzip                             // Added in addition to original exp
  val stage2kVec = VecInit(xrln2KRVec._1.zip(xrKRVec._1).map{ case (kLn2, kBase2) => Mux(stage2IsBase2, kBase2, kLn2) }) // Added in addition to original exp
  val stage2rVec = VecInit(xrln2KRVec._2.zip(xrKRVec._2).map{ case (rLn2, rBase2) => Mux(stage2IsBase2, rBase2, rLn2) }) // Added in addition to original exp

  // Stage 3: lut ready r[top] and r[top] + 1, compute delta
  val stage3kVec = maskLaneNext(stage2kVec, st(2).laneEn, bp(3))
  val stage3rVec = maskLaneNext(stage2rVec, st(2).laneEn, bp(3))
  val stage3AddrVec = VecInit(stage3rVec.map(r => r(fpT.qmnN - 1, rLowBits)))
  val stage3rLowerVec = VecInit(stage3rVec.map(r => r(rLowBits - 1, 0)))
  val stage3y0 = lut.io.rdata(0)
  val stage3y1 = lut.io.rdata(1)
  val stage3delta = VecInit(stage3y0.zip(stage3y1).zip(stage3AddrVec).map {
    case ((y0, y1), addr) =>
      val y1Interp = Mux(addr === ((1 << fpT.lutAddrBits) - 1).U, lutTopEndpoint, y1)
      val delta = y1Interp - y0
      delta
  })

  // Stage 4: compute delta * frac
  val stage4kVec = maskLaneNext(stage3kVec, st(3).laneEn, bp(4))
  val stage4y0 = maskLaneNext(stage3y0, st(3).laneEn, bp(4))
  val stage4delta = maskLaneNext(stage3delta, st(3).laneEn, bp(4))
  val stage4frac = maskLaneNext(stage3rLowerVec, st(3).laneEn, bp(4))
  val stage4deltaFrac = VecInit(stage4delta.zip(stage4frac).map {
    case (delta, frac) => (delta * frac) >> rLowBits
  })

  // Stage 5: add delta * frac to y0
  val stage5kVec = maskLaneNext(stage4kVec, st(4).laneEn, bp(5))
  val stage5y0 = maskLaneNext(stage4y0, st(4).laneEn, bp(5))
  val stage5deltaFrac = maskLaneNext(stage4deltaFrac, st(4).laneEn, bp(5))
  val stage5pow2rVec = VecInit(stage5y0.zip(stage5deltaFrac).map {
    case (y0, deltaFrac) => y0 + deltaFrac
  })

  // Stage 6: convert and return result
  val stage6kVec = maskLaneNext(stage5kVec, st(5).laneEn, bp(6))
  val stage6pow2rVec = maskLaneNext(stage5pow2rVec, st(5).laneEn, bp(6))
  val resRawFloat = stage6pow2rVec.zip(stage6kVec).map{ case (qmn, k) => fpT.rawFloatFromQmnK(qmn, k) }
  roundToRecFn.zip(resRawFloat).zip(st(6).laneEn).foreach {
    case ((round, rawFloat), en) => {
      round.io.invalidExc := false.B
      round.io.infiniteExc := false.B
      round.io.in := rawFloat
      round.io.roundingMode := Mux(en, st(6).req.roundingMode, 0.U)
      round.io.detectTininess := 0.U
    }
  }
  val resValid = st(6).valid
  val resReq = st(6).req
  val resFN = roundToRecFn.map(round => fNFromRecFN(fpT.expWidth, fpT.sigWidth, round.io.out))
  val resFinal = VecInit(resFN.zip(st(6).earlyTerm.zip(st(6).earlyRes)).map {
    case (res, (earlyTerminate, earlyRes)) => Mux(earlyTerminate, earlyRes, res)
  })

  lut.io.raddr := stage2rVec.map(r => r(fpT.qmnN - 1, rLowBits))
  lut.io.ren := VecInit(st(2).laneEn.map(en => en && st(2).valid && !bp(3)))

  io.req.ready := !backPressure(0)
  io.resp.valid := resValid
  io.resp.bits.tag := resReq.tag
  io.resp.bits.whichBank := resReq.whichBank
  io.resp.bits.wRow := resReq.wRow
  io.resp.bits.result := resFinal
  io.resp.bits.laneMask := resReq.laneMask
}
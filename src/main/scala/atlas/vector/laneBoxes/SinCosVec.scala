package sp26FPUnits

import chisel3._
import chisel3.util._
import sp26FPUnits.hardfloat._
import sp26FPUnits._

// Input bundles
class SinCosVecReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val cos = Bool() // 1 = cos, 0 = sin
    val laneMask = UInt(numLanes.W)
    val xVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class SinCosVecResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class SinCosVec(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasSinCosParams {
    val io = IO(new Bundle {
        val req = Flipped(Valid(new SinCosVecReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Valid(new SinCosVecResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    // Input restriciton
    val zeroBF16  = 0.U(BF16T.wordWidth.W)
    val twoPiBF16 = "h40C9".U(BF16T.wordWidth.W) // BF16 encoding of ~6.28125
    for (i <- 0 until numLanes) {
        when (io.req.valid && io.req.bits.laneMask(i)) {
            assert(
                io.req.bits.xVec(i) >= zeroBF16,
                s"xVec($i) must be in [0, 2pi]"
            )
        }
    }

    // Initial Setup
    val rLowBits = BF16T.qmnN - BF16T.lutAddrBits
    val lut = Module(new SinCosLUT(numLanes, BF16T.lutAddrBits, BF16T.lutValM, BF16T.lutValN))
    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.valid).asUInt).asBools)
    val maskedCos = io.req.bits.cos && io.req.valid
    val maskedXVec = io.req.bits.xVec.zip(laneEnable).map {
        case (x, en) => Mux (en, x, 0.U(BF16T.wordWidth.W))
    }
    
    /* Stage 0:
     *  1. BF16 -> Q(m,n)
     *  2. Determine sin/cos behavior
     *  3. Determine otuput sign
    */
    val qmnVec = VecInit(maskedXVec.map(x => bf16ToQmnTimesTwoOverPi(x)))
    val nVec   = VecInit(qmnVec.map(x => x(BF16T.qmnN - 1, 0)))
    val quadrantBits = VecInit(qmnVec.map(x => x(BF16T.qmnN + 1, BF16T.qmnN)))
    val isNegVec = VecInit(quadrantBits.map(x => Mux(maskedCos, 
            Mux(x === 1.U || x === 2.U, true.B, false.B), // Cosine is negative in quadrants 2 and 3
            Mux(x === 2.U || x === 3.U, true.B, false.B)  // Sine is negative in quadrants 3 and 4
    ))) 
    val isCosVec = VecInit(quadrantBits.map(x => Mux(maskedCos, 
        Mux(x === 0.U || x === 2.U, true.B, false.B), // If cos based, quadrant 1 and 3 behave the same as cos
        Mux(x === 1.U || x === 3.U, true.B, false.B)  // If sin based, quadrant 2 and 4 behave the same as cos
    )))
    
    // States for pipelining
    class CommonStageState extends Bundle {
        val valid = Bool()
        // val req = chiselTypeOf(io.req.bits)
        // val laneEn = chiselTypeOf(laneEnable)
        val isCos = Bool()
        val qmnNVec = chiselTypeOf(nVec)
    }

    // Initializing the states
    def numIntermediateStages = 1
    val commonState = RegInit(VecInit(Seq.fill(numIntermediateStages)(0.U.asTypeOf(new CommonStageState))))
    val backPressure = WireInit(VecInit(Seq.fill(numIntermediateStages)(false.B)))
    val stateWithBp = commonState.zip(backPressure)
    def st(i: Int) = commonState(i-1)
    def bp(i: Int) = backPressure(i-1)

    // Assigning values to the 0th stage of the pipeline: select between input and holding value based on backpressure.
    stateWithBp.take(1).foreach { case (state, back) =>
        state.valid := Mux(!back, io.req.valid, state.valid)
        // state.req := Mux(io.req.valid, io.req.bits, state.req)
        // state.laneEn := Mux(!back, laneEnable, state.laneEn)
        state.isCos := Mux(io.req.valid, maskedCos, state.isCos)
        state.qmnNVec := maskLane(state.qmnNVec, nVec, laneEnable, back)
    }

    lut.io.raddr := nVec.map(r => r(BF16T.qmnN - 1, rLowBits))
    lut.io.ren := VecInit(laneEnable.map(en => en && io.req.valid))
    lut.io.isCos := isCosVec

    /* Stage 1: Interpolation */
    val stage1MaskedNeg = maskLaneNext(isNegVec, laneEnable, bp(1))
    val stage1IsCosVec = maskLaneNext(isCosVec, laneEnable, bp(1))
    val stage1rLowerVec = VecInit(st(1).qmnNVec.zip(stage1IsCosVec).map{case(r, c) => Mux(c, (1 << rLowBits).U - r(rLowBits-1, 0), r(rLowBits-1, 0))})  
    val y0Vec = lut.io.rdata(0)
    val y1Vec = lut.io.rdata(1)
    val delta = VecInit(y0Vec.zip(y1Vec).zip(stage1IsCosVec).map { 
        case ((y0, y1), c) => Mux(c, y0 - y1, y1 - y0 )})
    val basey = VecInit(y0Vec.zip(y1Vec).zip(stage1IsCosVec).map{case ((y0, y1), c) => Mux(c, y1, y0)})
    val resultLutFixed = VecInit(delta.zip(stage1rLowerVec).zip(basey).map { case ((d, frac), y) =>
        val interp = ((d * frac) >> rLowBits) + y
        interp(BF16T.lutValM + BF16T.lutValN - 1, 0)
    })
    val resultBF16 = VecInit(resultLutFixed.map(x => lutFixedToBf16(x, BF16T.lutValM, BF16T.lutValN)))
    val resultFinal = VecInit(resultBF16.zip(stage1MaskedNeg).map{case (x, c) => Mux(c, Cat(1.U, x(14,0)), x)})
    val resValid = st(1).valid
    // val resReq = st(1).req

    // Output signals
    io.resp.valid := resValid
    io.resp.bits.result := resultFinal
}

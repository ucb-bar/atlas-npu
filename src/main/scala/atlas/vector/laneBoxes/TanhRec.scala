package atlas.vector

import chisel3._
import chisel3.util._
import fpex.hardfloat._
import fpex._
import sp26FPUnits._

// Input bundles
class TanhReq(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val laneMask = UInt(numLanes.W)
    val xVec = Vec(numLanes, UInt(wordWidth.W))
}

// Output bundles
class TanhResp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
    val tag = UInt(tagWidth.W)
    val whichBank = UInt(5.W)
    val wRow= UInt(7.W)
    val laneMask = UInt(numLanes.W)
    val result = Vec(numLanes, UInt(wordWidth.W))
}

class TanhRec(BF16T: FPType, numLanes: Int = 16, tagWidth: Int = 16) extends Module with HasPipelineParams {
    val w    = BF16T.wordWidth
    val expW = BF16T.expWidth
    val sigW = BF16T.sigWidth 
    val n    = BF16T.qmnN     
    val m    = BF16T.qmnM   
    val n1 = BF16T.lutValN 

    val io = IO(new Bundle {
        val req = Flipped(Decoupled(new TanhReq(BF16T.wordWidth, numLanes, tagWidth)))
        val resp = Decoupled(new TanhResp(BF16T.wordWidth, numLanes, tagWidth))
    })

    val laneEnable = VecInit((io.req.bits.laneMask & VecInit.fill(numLanes)(io.req.fire).asUInt).asBools)
    val lut = Module(new TanhLUT(ports = numLanes, addrBits = BF16T.lutAddrBits, m = 1, n = n1))
    val tanhModules = Seq.fill(numLanes) { Module(new Tanh(BF16T)) }

    class CommonStageState extends Bundle {
        val valid = Bool()
        val req = chiselTypeOf(io.req.bits)
        val laneEn = chiselTypeOf(laneEnable)
    }

    def numIntermediateStages = 1
    val commonState = RegInit(VecInit(Seq.fill(numIntermediateStages)(0.U.asTypeOf(new CommonStageState))))
    val backPressure = Wire(Vec(numIntermediateStages, Bool()))
    val stateWithBp = commonState.zip(backPressure)
    def st(i: Int) = commonState(i-1)
    def bp(i: Int) = backPressure(i-1)
    stateWithBp.take(1).foreach { case (state, back) =>
        state.valid := Mux(!back, io.req.fire, state.valid)
        state.req := Mux(io.req.fire, io.req.bits, state.req)
        state.laneEn := Mux(!back, laneEnable, state.laneEn)
    }
    backPressure(numIntermediateStages-1) := !io.resp.ready

    for (i <- 0 until numLanes) {
        tanhModules(i).io.x := io.req.bits.xVec(i)
        tanhModules(i).io.valid := laneEnable(i)

        lut.io.raddr(i) := tanhModules(i).io.lutAddr
        lut.io.ren(i) := tanhModules(i).io.lutEn

        tanhModules(i).io.lutVal1 := lut.io.rdata(0)(i)
        tanhModules(i).io.lutVal2 := lut.io.rdata(1)(i)
    }

    io.req.ready := !backPressure(0)
    io.resp.valid := commonState(0).valid
    io.resp.bits.tag := commonState(0).req.tag
    io.resp.bits.whichBank := commonState(0).req.whichBank
    io.resp.bits.wRow := commonState(0).req.wRow
    io.resp.bits.laneMask := 0xFFFF.U   // Need to fix later
    for (i <-0 until numLanes) {
        io.resp.bits.result(i) := tanhModules(i).io.result
    }
}
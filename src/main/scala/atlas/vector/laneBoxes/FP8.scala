package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._ 
import sp26FPUnits.hardfloat.consts._

// A BF16 -> FP8 (E4M3) converter
class FP8Req(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
  val tag = UInt(tagWidth.W)
  val whichBank = UInt(5.W)
  val wRow = UInt(7.W)
  val laneMask = UInt(numLanes.W)

  val expShift = SInt(8.W)
  val leftAlign = Bool()
  val xVec = Vec(numLanes, UInt(wordWidth.W))
}

class FP8Resp(wordWidth: Int, numLanes: Int, tagWidth: Int) extends Bundle {
  val tag = UInt(tagWidth.W)
  val whichBank = UInt(5.W)
  val wRow = UInt(7.W)
  val laneMask = UInt(numLanes.W)
  val result = Vec(numLanes, UInt(wordWidth.W))
}

class FP8(BF16T: AtlasFPType, numLanes: Int = 16, tagWidth: Int = 16)
    extends Module
    with HasPipelineParams {

  val io = IO(new Bundle {
    val req = Flipped(Decoupled(new FP8Req(BF16T.wordWidth, numLanes, tagWidth)))
    val resp = Decoupled(new FP8Resp(BF16T.wordWidth, numLanes, tagWidth))
  })

  // E4M3 constants
  val fp8ExpBits  = 4
  val fp8MantBits = 3
  val fp8Bias     = 7

  val E4M3_MAX_POS = "h7e".U(8.W)
  val E4M3_MAX_NEG = "hfe".U(8.W)

  // conversion
  val outVec = Wire(Vec(numLanes, UInt(BF16T.wordWidth.W)))

  for (i <- 0 until numLanes) {
    val bf16 = io.req.bits.xVec(i)

    // unpack bf16 numbers
    val sign   = bf16(15)
    val expBF  = bf16(14, 7)
    val mantBF = bf16(6, 0)

    // handling for other numbers
    val isZero = (expBF === 0.U) && (mantBF === 0.U)
    val isSub  = (expBF === 0.U) && (mantBF =/= 0.U)
    val isInf  = (expBF === "hff".U) && (mantBF === 0.U)
    val isNaN  = (expBF === "hff".U) && (mantBF =/= 0.U)

    // BF16 bias = 127, FP8 E4M3 bias = 7, so subtract 120 
    // (exp_bf16 - 127 + 7) + shift = exp_bf16 - 120 + shift
    val unbExp = expBF.zext - 127.S(10.W)
    val scaleExpWide = io.req.bits.expShift.pad(10)
    val expAdjusted = unbExp + scaleExpWide
   // val expAdjusted = expBF.asSInt - 120.S + io.req.bits.expShift

    // full 8-bit significand = 1.frac[6:0]
    val mant8 = Cat(1.U(1.W), mantBF)

    // round right shift by 4 using round-to-nearest-even- same as OutputConv!
    val trunc  = Cat(0.U(1.W), mant8(7, 4))   // 5 bits
    val guard  = mant8(3)                    // bit we may throw away
    val sticky = mant8(2, 0).orR                // are any of the bits we threw away 1?
    val lsb    = trunc(0)                     // our least sig bit- are we keeping it
    val inc    = guard && (sticky || lsb)       // round up... 
    val roundedSig= trunc + inc              // rounded signifciand

    val normCarry = (roundedSig === 16.U(5.W))

    val finalExpAdjusted = Wire(SInt(10.W))
    finalExpAdjusted := expAdjusted
    when (normCarry) {
        finalExpAdjusted := expAdjusted + 1.S
    }
    
    val mantFP8 = Wire(UInt(3.W))
    mantFP8 := 0.U
    when (!normCarry) {
      mantFP8 := (roundedSig - 8.U)(2, 0)
    }
    
    val fp8Pack = Wire(UInt(8.W))
    fp8Pack := 0.U

    when (isZero || isSub || isNaN) {
      fp8Pack := 0.U
    }.elsewhen (isInf) {
      fp8Pack := Mux(sign.asBool, E4M3_MAX_NEG, E4M3_MAX_POS)
    }.otherwise {
      when (finalExpAdjusted > 8.S) {
        // overflow -> max finite
        fp8Pack := Mux(sign.asBool, E4M3_MAX_NEG, E4M3_MAX_POS)
      }.elsewhen (finalExpAdjusted >= (-6).S) {
        // normal E4M3
        val expFP8 = (finalExpAdjusted + 7.S).asUInt(3, 0)
        fp8Pack := Cat(sign, expFP8, mantFP8)
      }.otherwise {
        // no FP8 subnormals
        fp8Pack := 0.U
      }
    }

    outVec(i) := Mux(
        io.req.bits.leftAlign,
        Cat(fp8Pack, 0.U(8.W)),
        Cat(0.U(8.W), fp8Pack)
        )
    }
    // changing from pure combinational to one cycle latency
    val respValidReg = RegInit(false.B)
    val respBitsReg  = RegInit(0.U.asTypeOf(new FP8Resp(BF16T.wordWidth, numLanes, tagWidth)))

    io.req.ready  := !respValidReg || io.resp.ready
    io.resp.valid := respValidReg
    io.resp.bits  := respBitsReg

    when (io.req.fire) {
        respValidReg := true.B
        respBitsReg.tag       := io.req.bits.tag
        respBitsReg.whichBank := io.req.bits.whichBank
        respBitsReg.wRow      := io.req.bits.wRow
        respBitsReg.laneMask  := io.req.bits.laneMask

        for (i <- 0 until numLanes) {
            respBitsReg.result(i) := outVec(i)
        }
            
    }

    when (io.resp.fire) {
        respValidReg := false.B
    }
  
    
}
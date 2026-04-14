// ============================================================================
// FP8Pack.scala — Vector BF16-to-FP8 conversion with 2→1 packing.
//
// Two consecutive BF16 input rows (16 lanes each) are converted to FP8 and
// packed into a single 256-bit output row (32 FP8 bytes).
//
// Timing: accepts 2 req valids, produces 1 resp valid.
//   req valid 0 → convert 16 BF16 → 16 FP8, buffer in low half
//   req valid 1 → convert 16 BF16 → 16 FP8, pack into high half, output
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._
import sp26FPUnits.hardfloat._
import sp26FPUnits.hardfloat.consts._

class FP8Req(wordWidth: Int, numLanes: Int) extends Bundle {
  val expShift = SInt(8.W)
  val xVec     = Vec(numLanes, UInt(wordWidth.W))
}

class FP8Resp(wordWidth: Int, numLanes: Int) extends Bundle {
  val result = Vec(numLanes, UInt(wordWidth.W))
}

class FP8Pack(BF16T: AtlasFPType, numLanes: Int = 16)
    extends Module
    with HasPipelineParams {

  val io = IO(new Bundle {
    val req  = Flipped(Valid(new FP8Req(BF16T.wordWidth, numLanes)))
    val resp = Valid(new FP8Resp(BF16T.wordWidth, numLanes))
  })

  // E4M3 constants
  val E4M3_MAX_POS = "h7e".U(8.W)
  val E4M3_MAX_NEG = "hfe".U(8.W)

  // ── Per-lane BF16 → FP8 conversion (combinational) ──────────────
  val fp8Bytes = Wire(Vec(numLanes, UInt(8.W)))

  for (i <- 0 until numLanes) {
    val bf16 = io.req.bits.xVec(i)

    val sign   = bf16(15)
    val expBF  = bf16(14, 7)
    val mantBF = bf16(6, 0)

    val isZero = (expBF === 0.U) && (mantBF === 0.U)
    val isSub  = (expBF === 0.U) && (mantBF =/= 0.U)
    val isInf  = (expBF === "hff".U) && (mantBF === 0.U)
    val isNaN  = (expBF === "hff".U) && (mantBF =/= 0.U)

    val unbExp       = expBF.zext - 127.S(10.W)
    val scaleExpWide = io.req.bits.expShift.pad(10)
    val expAdjusted  = unbExp - scaleExpWide

    val mant8      = Cat(1.U(1.W), mantBF)
    val trunc      = Cat(0.U(1.W), mant8(7, 4))
    val guard      = mant8(3)
    val sticky     = mant8(2, 0).orR
    val lsb        = trunc(0)
    val inc        = guard && (sticky || lsb)
    val roundedSig = trunc + inc
    val normCarry  = (roundedSig === 16.U(5.W))

    val finalExpAdjusted = Wire(SInt(10.W))
    finalExpAdjusted := Mux(normCarry, expAdjusted + 1.S, expAdjusted)

    val mantFP8 = Wire(UInt(3.W))
    mantFP8 := Mux(normCarry, 0.U, (roundedSig - 8.U)(2, 0))

    val fp8 = Wire(UInt(8.W))
    fp8 := 0.U
    when(isZero || isSub || isNaN) {
      fp8 := 0.U
    }.elsewhen(isInf) {
      fp8 := Mux(sign.asBool, E4M3_MAX_NEG, E4M3_MAX_POS)
    }.otherwise {
      when(finalExpAdjusted > 8.S) {
        fp8 := Mux(sign.asBool, E4M3_MAX_NEG, E4M3_MAX_POS)
      }.elsewhen(finalExpAdjusted >= (-6).S) {
        val expFP8 = (finalExpAdjusted + 7.S).asUInt(3, 0)
        val packed = Cat(sign, expFP8, mantFP8)
        fp8 := Mux(
          (finalExpAdjusted === 8.S) && (mantFP8 === 7.U),
          Mux(sign.asBool, E4M3_MAX_NEG, E4M3_MAX_POS),
          packed
        )
      }.otherwise {
        fp8 := 0.U
      }
    }
    fp8Bytes(i) := fp8
  }

  // ── 2→1 accumulation ───────────────────────────────────────────
  val phase        = RegInit(false.B)              // false = first half, true = second
  val lowHalf      = Reg(Vec(numLanes, UInt(8.W))) // buffered first 16 FP8 bytes
  val respValidReg = RegInit(false.B)
  val respBitsReg  = Reg(Vec(numLanes, UInt(BF16T.wordWidth.W)))

  io.resp.valid := respValidReg
  io.resp.bits.result := respBitsReg

  when(io.req.valid) {
    when(!phase) {
      // First input: buffer 16 FP8 bytes, no output yet
      lowHalf := fp8Bytes
      phase   := true.B
      respValidReg := false.B
    }.otherwise {
      // Second input: pack both halves into one 256-bit row
      // result(i) = bits[16i+15 : 16i]. Each 16-bit slot holds two FP8 bytes.
      // Low half (bytes 0..15) → result(0)..result(7)
      // High half (bytes 16..31) → result(8)..result(15)
      phase        := false.B
      respValidReg := true.B
      for (j <- 0 until numLanes / 2) {
        respBitsReg(j)              := Cat(lowHalf(2 * j + 1),  lowHalf(2 * j))
        respBitsReg(j + numLanes/2) := Cat(fp8Bytes(2 * j + 1), fp8Bytes(2 * j))
      }
    }
  }.otherwise {
    respValidReg := false.B
  }
}

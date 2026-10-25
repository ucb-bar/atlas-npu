// ============================================================================
// FP8Pack.scala — Vector BF16-to-FP8 conversion with 2→1 packing.
//
// Two consecutive BF16 input rows (16 lanes each) are converted to FP8 and
// packed into a single 256-bit output row (32 FP8 bytes). Each byte is
// E4M3_RN_SATFINITE(x * 2^(scale - 127)) (Cvt.e4m3RnSatfinite).
//
// Timing: accepts 2 req valids, produces 1 resp valid.
//   req valid 0 → convert 16 BF16 → 16 FP8, buffer in low half
//   req valid 1 → convert 16 BF16 → 16 FP8, pack into high half, output
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.Cvt
import sp26FPUnits._
import sp26FPUnits.hardfloat._
import sp26FPUnits.hardfloat.consts._

class FP8Req(wordWidth: Int, numLanes: Int) extends Bundle {
  val scale    = UInt(8.W)
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

  // ── Per-lane BF16 → FP8 conversion (combinational) ──────────────
  val fp8Bytes =
    VecInit(io.req.bits.xVec.map(x => Cvt.e4m3RnSatfinite(Cat(x, 0.U(16.W)), io.req.bits.scale)))

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

// ============================================================================
// FP8Unpack.scala — FP8 (E4M3) → BF16 conversion with 1→2 unpacking.
//
// One packed 256-bit input row (32 FP8 bytes) produces two BF16 output rows
// (16 BF16 values each).
//
// Timing: accepts 1 req valid, produces 2 resp valids.
//   req valid   → buffer 32 FP8 bytes
//   resp valid 0 → convert low 16 bytes → 16 BF16 values
//   resp valid 1 → convert high 16 bytes → 16 BF16 values
//
// Each byte converts as BF16_RNE(x * 2^(scale - 127)) (Cvt.bf16Rne).
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.Cvt
import sp26FPUnits._

class FP8UnpackReq(wordWidth: Int, numLanes: Int) extends Bundle {
  val scale    = UInt(8.W)
  val xVec     = Vec(numLanes, UInt(wordWidth.W))
}

class FP8UnpackResp(wordWidth: Int, numLanes: Int) extends Bundle {
  val result = Vec(numLanes, UInt(wordWidth.W))
}

class FP8Unpack(BF16T: AtlasFPType, numLanes: Int = 16)
    extends Module
    with HasPipelineParams {

  val io = IO(new Bundle {
    val req  = Flipped(Valid(new FP8UnpackReq(BF16T.wordWidth, numLanes)))
    val resp = Valid(new FP8UnpackResp(BF16T.wordWidth, numLanes))
  })

  // ── State: idle → lowHalf → highHalf → idle ────────────────────
  val sIdle :: sLow :: sHigh :: Nil = Enum(3)
  val state     = RegInit(sIdle)
  val inputBuf  = Reg(UInt(256.W))
  val scaleBuf  = Reg(UInt(8.W))

  // Buffer incoming packed rows so the VPU can stream mreg reads at line rate
  // without any external backpressure, even though each unpacked row takes
  // two response beats to drain.
  val reqQ = Module(new Queue(new FP8UnpackReq(BF16T.wordWidth, numLanes), 32))
  reqQ.io.enq.valid := io.req.valid
  reqQ.io.enq.bits  := io.req.bits
  reqQ.io.deq.ready := false.B

  assert(!io.req.valid || reqQ.io.enq.ready,
    "FP8Unpack: request queue overflow (software scheduling violated)")

  // ── Extract 16 FP8 bytes based on current half ─────────────────
  val fp8Bytes = Wire(Vec(numLanes, UInt(8.W)))
  for (i <- 0 until numLanes) {
    fp8Bytes(i) := Mux(state === sHigh,
      inputBuf(128 + 8 * i + 7, 128 + 8 * i),   // high 16 bytes
      inputBuf(      8 * i + 7,       8 * i)     // low 16 bytes
    )
  }

  // ── Per-lane FP8 → BF16 conversion (combinational) ─────────────
  val outVec = VecInit(fp8Bytes.map(x => Cvt.bf16Rne(Cvt.e4m3ToF32(x), scaleBuf)))

  // ── Control ────────────────────────────────────────────────────
  io.resp.valid := (state === sLow) || (state === sHigh)
  io.resp.bits.result := outVec

  switch(state) {
    is(sIdle) {
      when(reqQ.io.deq.valid) {
        reqQ.io.deq.ready := true.B
        inputBuf := reqQ.io.deq.bits.xVec.asUInt
        scaleBuf := reqQ.io.deq.bits.scale
        state    := sLow
      }
    }
    is(sLow) {
      state := sHigh
    }
    is(sHigh) {
      when(reqQ.io.deq.valid) {
        reqQ.io.deq.ready := true.B
        inputBuf := reqQ.io.deq.bits.xVec.asUInt
        scaleBuf := reqQ.io.deq.bits.scale
        state    := sLow
      }.otherwise {
        state := sIdle
      }
    }
  }
}

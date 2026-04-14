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
// Subnormals and the reserved NaN encoding are flushed to signed zero,
// matching the MXU dequant policy.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import sp26FPUnits._

class FP8UnpackReq(wordWidth: Int, numLanes: Int) extends Bundle {
  val expShift = SInt(8.W)
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
  val expBuf    = Reg(SInt(8.W))

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
  val outVec = Wire(Vec(numLanes, UInt(BF16T.wordWidth.W)))

  for (i <- 0 until numLanes) {
    val fp8     = fp8Bytes(i)
    val sign    = fp8(7)
    val expFP8  = fp8(6, 3)
    val mantFP8 = fp8(2, 0)

    val isZero = (expFP8 === 0.U) && (mantFP8 === 0.U)
    val isSub  = (expFP8 === 0.U) && (mantFP8 =/= 0.U)
    val isNaN  = (expFP8 === "hf".U) && (mantFP8 === "h7".U)

    val bf16 = Wire(UInt(16.W))
    bf16 := 0.U

    when(isZero || isSub || isNaN) {
      bf16 := Cat(sign, 0.U(15.W))
    }.otherwise {
      val unbExpFP8   = expFP8.zext - 7.S(10.W)
      val scaleWide   = expBuf.pad(10)
      val unbExpBF16  = unbExpFP8 + scaleWide
      val expBF16Wide = unbExpBF16 + 127.S(10.W)

      when(expBF16Wide <= 0.S) {
        bf16 := Cat(sign, 0.U(15.W))
      }.elsewhen(expBF16Wide >= 255.S) {
        bf16 := Mux(sign.asBool, "hff7f".U(16.W), "h7f7f".U(16.W))
      }.otherwise {
        val fracBF16 = Cat(mantFP8, 0.U(4.W))
        bf16 := Cat(sign, expBF16Wide(7, 0), fracBF16)
      }
    }
    outVec(i) := bf16
  }

  // ── Control ────────────────────────────────────────────────────
  io.resp.valid := (state === sLow) || (state === sHigh)
  io.resp.bits.result := outVec

  switch(state) {
    is(sIdle) {
      when(reqQ.io.deq.valid) {
        reqQ.io.deq.ready := true.B
        inputBuf := reqQ.io.deq.bits.xVec.asUInt
        expBuf   := reqQ.io.deq.bits.expShift
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
        expBuf   := reqQ.io.deq.bits.expShift
        state    := sLow
      }.otherwise {
        state := sIdle
      }
    }
  }
}

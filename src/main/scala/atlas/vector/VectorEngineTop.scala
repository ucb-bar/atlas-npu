// ============================================================================
// VectorEngineTop.scala — Top-level VPU wrapper.
//
// Translates ScalarCore VpuCmd into VectorEngine VectorInput, manages mreg
// port wiring, and exposes active bank tracking for the MregBankTracker.
//
// Bank tracking rationale:
//   Most BF16 VPU ops span two consecutive mregs because the FSM toggles
//   the bank at counter(5). The exceptions are:
//     - fp8pack:   reads two BF16 mregs, writes one packed-FP8 mreg
//     - fp8unpack: reads one packed-FP8 mreg, writes two BF16 mregs
//     - vliCol / vliOne: write exactly one 32x16 BF16 tensor register
//     - vliAll / vliRow: write a full BF16 tensor pair
//   We expose enough ports for the MregBankTracker to see every live bank the
//   VPU is reading or reserving for writeback, including overlap between two
//   independent single-input ops.
// ============================================================================

package atlas.vector

import chisel3._
import chisel3.util._
import atlas.common.{VpuParams, MregParams, MregReadReq, MregWriteReq}
import atlas.scalar.VpuCmd

class VectorEngineTop(
  p:     VpuParams,
  mregP: MregParams
) extends Module {

  val io = IO(new Bundle {
    // ── Command input (fire-and-forget, no backpressure) ──
    val cmd = Flipped(Valid(new VpuCmd()))

    // ── Tensor register file (mreg) ports ──
    val mregReadReq0  = Valid(new MregReadReq(mregP))
    val mregReadResp0 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregReadReq1  = Valid(new MregReadReq(mregP))
    val mregReadResp1 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWriteReq0 = Valid(new MregWriteReq(mregP))
    val mregWriteReq1 = Valid(new MregWriteReq(mregP))

    // ── Busy signals ──
    val busy      = Output(Bool())
    val issueBusy = Output(UInt((VPUOp.all.size + 1).W))

    // ── Active mreg bank tracking (for MregBankTracker) ──
    val activeReads  = Output(Vec(4, Valid(UInt(mregP.mregIdBits.W))))
    val activeWrites = Output(Vec(4, Valid(UInt(mregP.mregIdBits.W))))
  })

  // ==========================================================================
  // Inner vector engine
  // ==========================================================================

  val core = Module(new VectorEngine(p))

  // ==========================================================================
  // Instruction Translation (VpuCmd → VectorInput)
  //
  // ScalarISA VPU_* constants are 1-indexed (VPU_NONE=0, VPU_ADD=1, ...),
  // while VPUOp ChiselEnum is 0-indexed (add=0, sub=1, ...).
  // Subtract 1 before casting to align the two numbering schemes.
  // ==========================================================================

  private def opIsVliAllOrRow(op: VPUOp.Type): Bool =
    op === VPUOp.vliAll || op === VPUOp.vliRow

  private def opIsVliColOrOne(op: VPUOp.Type): Bool =
    op === VPUOp.vliCol || op === VPUOp.vliOne

  val vecInput = Wire(new VectorInput(p))
  val cmdOp    = VPUOp.safe(io.cmd.bits.op - 1.U)._1

  val isFp8pack   = cmdOp === VPUOp.fp8pack
  val isFp8unpack = cmdOp === VPUOp.fp8unpack
  val isPackUnpack = isFp8pack || isFp8unpack
  val isVliAllOrRow = opIsVliAllOrRow(cmdOp)
  val isVliColOrOne = opIsVliColOrOne(cmdOp)
  val isVli = isVliAllOrRow || isVliColOrOne

  vecInput.instType      := cmdOp
  vecInput.instReadBank1 := Mux(isPackUnpack, io.cmd.bits.vs2, io.cmd.bits.vs1)
  vecInput.instReadBank2 := io.cmd.bits.vs2
  vecInput.instWriteBank := io.cmd.bits.vd
  vecInput.imm           := io.cmd.bits.imm.asSInt

  vecInput.packScaleE8M0   := io.cmd.bits.scaleE8M0
  vecInput.unpackScaleE8M0 := io.cmd.bits.scaleE8M0

  core.io.inst.valid := io.cmd.valid
  core.io.inst.bits  := vecInput

  val sameReadBank = core.io.read1.valid && core.io.read2.valid &&
    (core.io.read1.bits.bank === core.io.read2.bits.bank)
  val mirroredReadReq = sameReadBank &&
    (core.io.read1.bits.row === core.io.read2.bits.row)
  val mirroredReadReq_d = RegNext(mirroredReadReq, init = false.B)

  // ==========================================================================
  // Mreg read port 0 ↔ VectorEngine read1
  // ==========================================================================

  io.mregReadReq0.valid       := core.io.read1.valid
  io.mregReadReq0.bits.mregId := core.io.read1.bits.bank
  io.mregReadReq0.bits.row    := core.io.read1.bits.row

  core.io.data1.valid     := io.mregReadResp0.valid
  core.io.data1.bits.data := io.mregReadResp0.bits

  // ==========================================================================
  // Mreg read port 1 ↔ VectorEngine read2
  // ==========================================================================

  io.mregReadReq1.valid       := core.io.read2.valid && !mirroredReadReq
  io.mregReadReq1.bits.mregId := core.io.read2.bits.bank
  io.mregReadReq1.bits.row    := core.io.read2.bits.row

  core.io.data2.valid     := Mux(mirroredReadReq_d, io.mregReadResp0.valid, io.mregReadResp1.valid)
  core.io.data2.bits.data := Mux(mirroredReadReq_d, io.mregReadResp0.bits, io.mregReadResp1.bits)

  // ==========================================================================
  // Mreg write port 0 ↔ VectorEngine write1
  // ==========================================================================

  io.mregWriteReq0.valid       := core.io.write1.valid
  io.mregWriteReq0.bits.mregId := core.io.write1.bits.bank
  io.mregWriteReq0.bits.row    := core.io.write1.bits.row
  io.mregWriteReq0.bits.data   := core.io.write1.bits.data

  // ==========================================================================
  // Mreg write port 1 ↔ VectorEngine write2
  // ==========================================================================

  io.mregWriteReq1.valid       := core.io.write2.valid
  io.mregWriteReq1.bits.mregId := core.io.write2.bits.bank
  io.mregWriteReq1.bits.row    := core.io.write2.bits.row
  io.mregWriteReq1.bits.data   := core.io.write2.bits.data

  // ==========================================================================
  // Busy
  // ==========================================================================

  io.busy      := core.io.busy
  io.issueBusy := core.io.issueBusy

  // ==========================================================================
  // Structural hazard assertion (debug only)
  // ==========================================================================

  val cmdIssueBusy = core.io.issueBusy(io.cmd.bits.op)
  assert(!(io.cmd.valid && cmdIssueBusy),
    "VPU: command issued while selected VPU resources are busy (software-scheduling contract violated)")

  assert(!(sameReadBank && !mirroredReadReq),
    "VPU: concurrent reads to the same MREG bank must target the same row")
  assert(!(mirroredReadReq_d && io.mregReadResp1.valid),
    "VPU: mirrored VPU operand reads must not receive a second MREG response")

  val isTwoInput = (cmdOp === VPUOp.add || cmdOp === VPUOp.sub ||
                    cmdOp === VPUOp.mul || cmdOp === VPUOp.pairmax ||
                    cmdOp === VPUOp.pairmin)
  val readsPrimaryPair = !isVli && !isFp8unpack
  val readsSecondaryPair = isTwoInput
  val writesPair = !isFp8pack && !isVliColOrOne

  assert(!(io.cmd.valid && readsPrimaryPair && vecInput.instReadBank1(0)),
    "VPU: pair-read primary bank must be even")
  assert(!(io.cmd.valid && readsSecondaryPair && vecInput.instReadBank2(0)),
    "VPU: pair-read secondary bank must be even")
  assert(!(io.cmd.valid && writesPair && vecInput.instWriteBank(0)),
    "VPU: pair-write destination bank must be even")

  // ==========================================================================
  // Active mreg bank tracking
  // ==========================================================================

  io.activeReads  := core.io.activeReads
  io.activeWrites := core.io.activeWrites
}

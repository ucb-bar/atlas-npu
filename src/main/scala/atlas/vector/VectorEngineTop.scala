// ============================================================================
// VectorEngineTop.scala — Top-level VPU wrapper.
//
// Translates ScalarCore VpuCmd into VectorEngine VectorInput, manages mreg
// port wiring, and exposes active bank tracking for the MregBankTracker.
//
// Bank tracking rationale:
//   Most BF16 VPU ops span two consecutive mregs because the FSM toggles
//   the bank at counter(5). VLI is the exception: it writes exactly one
//   32x16 BF16 tensor register. We expose enough read/write ports for the
//   MregBankTracker to see every bank the VPU accesses.
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

    // ── Busy signal ──
    val busy = Output(Bool())

    // ── Active mreg bank tracking (for MregBankTracker) ──
    val activeReads  = Output(Vec(4, Valid(UInt(mregP.mregIdBits.W))))
    val activeWrites = Output(Vec(2, Valid(UInt(mregP.mregIdBits.W))))
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

  val vecInput = Wire(new VectorInput(p))
  val cmdOp    = VPUOp.safe(io.cmd.bits.op - 1.U)._1

  val isPackUnpack = (cmdOp === VPUOp.fp8pack) ||
                     (cmdOp === VPUOp.fp8unpack)

  vecInput.instType      := cmdOp
  vecInput.instReadBank1 := Mux(isPackUnpack, io.cmd.bits.vs2, io.cmd.bits.vs1)
  vecInput.instReadBank2 := io.cmd.bits.vs2
  vecInput.instWriteBank := io.cmd.bits.vd
  vecInput.imm           := io.cmd.bits.imm.asSInt

  vecInput.packScaleE8M0   := io.cmd.bits.scaleE8M0
  vecInput.unpackScaleE8M0 := io.cmd.bits.scaleE8M0

  core.io.inst.valid := io.cmd.valid
  core.io.inst.bits  := vecInput

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

  io.mregReadReq1.valid       := core.io.read2.valid
  io.mregReadReq1.bits.mregId := core.io.read2.bits.bank
  io.mregReadReq1.bits.row    := core.io.read2.bits.row

  core.io.data2.valid     := io.mregReadResp1.valid
  core.io.data2.bits.data := io.mregReadResp1.bits

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

  io.busy := core.io.busy

  // ==========================================================================
  // Structural hazard assertion (debug only)
  // ==========================================================================

  assert(!(io.cmd.valid && io.busy),
    "VPU: command issued while engine is busy (software-scheduling contract violated)")

  // ==========================================================================
  // Active mreg bank tracking
  // ==========================================================================

  val isVli = (cmdOp === VPUOp.vliAll || cmdOp === VPUOp.vliRow ||
               cmdOp === VPUOp.vliCol || cmdOp === VPUOp.vliOne)
  val isTwoInput = (cmdOp === VPUOp.add || cmdOp === VPUOp.sub ||
                    cmdOp === VPUOp.mul || cmdOp === VPUOp.pairmax ||
                    cmdOp === VPUOp.pairmin)

  val activeRd = Reg(Vec(4, Valid(UInt(mregP.mregIdBits.W))))
  val activeWr = Reg(Vec(2, Valid(UInt(mregP.mregIdBits.W))))

  when(reset.asBool || !core.io.busy) {
    for (i <- 0 until 4) { activeRd(i).valid := false.B }
    for (i <- 0 until 2) { activeWr(i).valid := false.B }
  }

  when(io.cmd.valid) {
    val readsAny = !isVli
    activeRd(0).valid := readsAny
    activeRd(0).bits  := vecInput.instReadBank1
    activeRd(1).valid := readsAny
    activeRd(1).bits  := vecInput.instReadBank1 + 1.U

    activeRd(2).valid := isTwoInput
    activeRd(2).bits  := vecInput.instReadBank2
    activeRd(3).valid := isTwoInput
    activeRd(3).bits  := vecInput.instReadBank2 + 1.U

    activeWr(0).valid := true.B
    activeWr(0).bits  := vecInput.instWriteBank
    activeWr(1).valid := !isVli
    activeWr(1).bits  := vecInput.instWriteBank + 1.U
  }

  io.activeReads  := activeRd
  io.activeWrites := activeWr
}

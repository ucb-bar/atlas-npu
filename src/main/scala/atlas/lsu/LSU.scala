// ============================================================================
// LSU.scala — Load-Store Unit (concurrent scalar + vector paths).
//
// Two independent datapaths operating concurrently:
//
//   Scalar path (pipelined — no FSM)
//     Scalar loads:  VMEM read (cycle 0) → word extraction (cycle 1).
//                    Fixed 2-cycle latency.
//     Scalar stores: Single-cycle masked VMEM write (no RMW).
//                    Fixed 1-cycle latency.
//
//   Vector path (FSM)
//     VLOAD  (VMEM → mreg): Streams mregRows lines.
//                    Fixed mregRows + 1 cycle latency.
//     VSTORE (mreg → VMEM): Streams mregRows rows.
//                    Fixed mregRows + 1 cycle latency.
//
// Software guarantees no VMEM bank conflicts between scalar and vector
// paths.  Vmem asserts on violations.  All LSU operations have
// deterministic latency — no stalls.
//
// Busy signals:
//   scalarBusy — true while a scalar load response is pending.
//   vecBusy    — true while a VLOAD/VSTORE is in progress.
// ============================================================================

package atlas.lsu

import chisel3._
import chisel3.util._

import atlas.common._
import atlas.scalar.ScalarISA

// ============================================================================
// Scalar command bundle
// ============================================================================

class LsuScalarCmd(vmemP: VmemParams) extends Bundle {
  val isStore  = Bool()
  val byteAddr = UInt(vmemP.byteAddrBits.W)
  val wdata    = UInt(32.W)
  val wmask    = UInt(4.W)
}

// ============================================================================
// LSU module
// ============================================================================

class LSU(vmemP: VmemParams, mregP: MregParams) extends Module {
  import ScalarISA._

  private val lineOffBits = log2Ceil(vmemP.lineBytes)  // 5
  private val mregRows    = mregP.mregRows

  val io = IO(new Bundle {
    val scalarCmd  = Flipped(Valid(new LsuScalarCmd(vmemP)))
    val scalarResp = Valid(UInt(32.W))
    val cmd        = Flipped(Valid(new atlas.scalar.LsuCmd))

    // ── VMEM scalar ports (deterministic — always granted) ──
    val vmemScalarRead     = Valid(new VmemLineReadPort(vmemP))
    val vmemScalarReadData = Flipped(Valid(UInt(vmemP.lineWidthBits.W)))
    val vmemScalarWrite    = Valid(new MaskedVmemLineWritePort(vmemP))

    // ── VMEM vector ports (deterministic — always granted) ──
    val vmemVecRead     = Valid(new VmemLineReadPort(vmemP))
    val vmemVecReadData = Flipped(Valid(UInt(vmemP.lineWidthBits.W)))
    val vmemVecWrite    = Valid(new VmemLineWritePort(vmemP))

    // ── Mreg ports ──
    val mregReadReq  = Valid(new MregReadReq(mregP))
    val mregReadResp = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWriteReq = Valid(new MregWriteReq(mregP))

    val scalarBusy = Output(Bool())
    val vecBusy    = Output(Bool())

    val activeMregRead  = Valid(UInt(mregP.mregIdBits.W))
    val activeMregWrite = Valid(UInt(mregP.mregIdBits.W))
  })

  // ==========================================================================
  // Scalar Load Path (pipelined — fixed 2-cycle latency)
  // ==========================================================================

  val scalarLoadPending = RegInit(false.B)
  val scalarWordIdx     = Reg(UInt(log2Ceil(vmemP.wordsPerLine).W))

  val issueScalarLoad = io.scalarCmd.valid && !io.scalarCmd.bits.isStore

  // Decompose byte address into line address and word index.
  val scalarFullByteAddr = io.scalarCmd.bits.byteAddr
  val scalarLineAddr     = scalarFullByteAddr(vmemP.byteAddrBits - 1, lineOffBits)

  io.vmemScalarRead.valid          := issueScalarLoad
  io.vmemScalarRead.bits.bankIdx   := vmemP.getBankIdx(scalarLineAddr)
  io.vmemScalarRead.bits.bankAddr  := vmemP.getBankAddr(scalarLineAddr)

  when(issueScalarLoad) {
    scalarLoadPending := true.B
    scalarWordIdx     := scalarFullByteAddr(lineOffBits - 1, 2)
  }

  val lineWords = Wire(Vec(vmemP.wordsPerLine, UInt(32.W)))
  for (i <- 0 until vmemP.wordsPerLine) {
    lineWords(i) := io.vmemScalarReadData.bits(32 * i + 31, 32 * i)
  }

  io.scalarResp.valid := io.vmemScalarReadData.valid && scalarLoadPending
  io.scalarResp.bits  := lineWords(scalarWordIdx)

  when(io.scalarResp.valid) {
    scalarLoadPending := false.B
  }

  io.scalarBusy := scalarLoadPending

  // ==========================================================================
  // Scalar Store Path (single-cycle masked write)
  // ==========================================================================

  val issueScalarStore = io.scalarCmd.valid && io.scalarCmd.bits.isStore
  val storeLineAddr    = scalarFullByteAddr(vmemP.byteAddrBits - 1, lineOffBits)
  val storeWordIdx     = scalarFullByteAddr(lineOffBits - 1, 2)

  io.vmemScalarWrite.valid         := issueScalarStore
  io.vmemScalarWrite.bits.bankIdx  := vmemP.getBankIdx(storeLineAddr)
  io.vmemScalarWrite.bits.bankAddr := vmemP.getBankAddr(storeLineAddr)
  io.vmemScalarWrite.bits.data     := io.scalarCmd.bits.wdata.pad(vmemP.lineWidthBits) << (storeWordIdx ## 0.U(5.W))

  // Position the 4-bit byte-mask within the 32-byte line mask.
  val shiftedMask = io.scalarCmd.bits.wmask.pad(vmemP.lineBytes) << (storeWordIdx ## 0.U(2.W))
  io.vmemScalarWrite.bits.mask := VecInit(shiftedMask(vmemP.lineBytes - 1, 0).asBools)

  // ==========================================================================
  // Vector Path FSM (deterministic — no stalls)
  // ==========================================================================

  val (sVecIdle :: sVecRun :: sVecDrain :: Nil) = Enum(3)

  val vecState = RegInit(sVecIdle)

  val vecOp       = Reg(UInt(2.W))
  val vecMregId   = Reg(UInt(mregP.mregIdBits.W))
  val vecVmemBase = Reg(UInt(vmemP.lineAddrBits.W))
  val vecCounter  = RegInit(0.U(log2Ceil(mregRows + 1).W))
  val vecIsLoad   = vecOp === LSU_VLOAD

  val prevCounter = RegNext(vecCounter)
  val prevValid   = RegNext(vecState === sVecRun, false.B)

  io.vecBusy := vecState =/= sVecIdle

  val vecActive = (vecState === sVecRun || vecState === sVecDrain)
  io.activeMregRead.valid  := vecActive && !vecIsLoad
  io.activeMregRead.bits   := vecMregId
  io.activeMregWrite.valid := vecActive && vecIsLoad
  io.activeMregWrite.bits  := vecMregId

  // ── Vector port defaults ──
  val vecLineAddr = vecVmemBase + vecCounter

  io.vmemVecRead.valid          := false.B
  io.vmemVecRead.bits.bankIdx   := vmemP.getBankIdx(vecLineAddr)
  io.vmemVecRead.bits.bankAddr  := vmemP.getBankAddr(vecLineAddr)

  io.vmemVecWrite.valid         := false.B
  io.vmemVecWrite.bits.bankIdx  := 0.U
  io.vmemVecWrite.bits.bankAddr := 0.U
  io.vmemVecWrite.bits.data     := 0.U

  io.mregReadReq.valid       := false.B
  io.mregReadReq.bits.mregId := vecMregId
  io.mregReadReq.bits.row    := vecCounter
  io.mregWriteReq.valid      := false.B
  io.mregWriteReq.bits.mregId := vecMregId
  io.mregWriteReq.bits.row   := 0.U
  io.mregWriteReq.bits.data  := 0.U

  switch(vecState) {
    is(sVecIdle) {
      when(io.cmd.valid) {
        vecState    := sVecRun
        vecOp       := io.cmd.bits.op
        vecMregId   := io.cmd.bits.mregBank
        vecVmemBase := io.cmd.bits.vmemLineAddr
        vecCounter  := 0.U
      }
    }

    is(sVecRun) {
      when(vecIsLoad) {
        io.vmemVecRead.valid          := true.B
        io.vmemVecRead.bits.bankIdx   := vmemP.getBankIdx(vecLineAddr)
        io.vmemVecRead.bits.bankAddr  := vmemP.getBankAddr(vecLineAddr)
      }.otherwise {
        io.mregReadReq.valid       := true.B
        io.mregReadReq.bits.mregId := vecMregId
        io.mregReadReq.bits.row    := vecCounter
      }
      vecCounter := vecCounter + 1.U
      when(vecCounter === (mregRows - 1).U) {
        vecState := sVecDrain
      }
    }

    is(sVecDrain) {
      vecState := sVecIdle
    }
  }

  // ==========================================================================
  // Matrix write pipeline (one-cycle delayed from the read)
  // ==========================================================================

  val prevLineAddr = vecVmemBase + prevCounter

  when(prevValid || (vecState === sVecDrain)) {
    when(vecIsLoad) {
      io.mregWriteReq.valid       := io.vmemVecReadData.valid
      io.mregWriteReq.bits.mregId := vecMregId
      io.mregWriteReq.bits.row    := prevCounter
      io.mregWriteReq.bits.data   := io.vmemVecReadData.bits
    }.otherwise {
      io.vmemVecWrite.valid          := io.mregReadResp.valid
      io.vmemVecWrite.bits.bankIdx   := vmemP.getBankIdx(prevLineAddr)
      io.vmemVecWrite.bits.bankAddr  := vmemP.getBankAddr(prevLineAddr)
      io.vmemVecWrite.bits.data      := io.mregReadResp.bits
    }
  }
}

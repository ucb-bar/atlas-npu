// ============================================================================
// LSU.scala — Load-Store Unit (concurrent scalar + VLOAD + VSTORE paths).
//
// Three independent datapaths operating concurrently:
//
//   Scalar path (pipelined — no FSM)
//     Scalar loads:  VMEM read (cycle 0) → word extraction (cycle 1).
//                    Fixed 2-cycle latency.
//     Scalar stores: Single-cycle masked VMEM write (no RMW).
//                    Fixed 1-cycle latency.
//
//   VLOAD path (FSM)
//     VLOAD  (VMEM → mreg): Streams mregRows lines.
//                    Fixed mregRows + 1 cycle latency.
//
//   VSTORE path (FSM)
//     VSTORE (mreg → VMEM): Streams mregRows rows.
//                    Fixed mregRows + 1 cycle latency.
//
// Software guarantees no same-port VMEM bank conflicts between scalar and
// vector paths. Under block banking, aligned VLOAD/VSTORE operations stay in
// one bank for their entire 32-line transfer. Vmem asserts on violations. All
// LSU operations have deterministic latency — no stalls.
//
// Busy signals:
//   scalarBusy — true while a scalar load response is pending.
//   vloadBusy  — true while a VLOAD is in progress.
//   vstoreBusy — true while a VSTORE is in progress.
//   vecBusy    — true while either vector FSM is active.
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
  private val tensorLineBits = log2Ceil(mregRows)

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
    val vloadBusy  = Output(Bool())
    val vstoreBusy = Output(Bool())
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
  // Vector Path FSMs (deterministic — no stalls)
  // ==========================================================================

  val (sVecIdle :: sVecRun :: sVecDrain :: Nil) = Enum(3)

  val issueVloadCmd  = io.cmd.valid && io.cmd.bits.op === LSU_VLOAD
  val issueVstoreCmd = io.cmd.valid && io.cmd.bits.op === LSU_VSTORE

  val vloadState    = RegInit(sVecIdle)
  val vloadMregId   = Reg(UInt(mregP.mregIdBits.W))
  val vloadVmemBase = Reg(UInt(vmemP.lineAddrBits.W))
  val vloadCounter  = RegInit(0.U(log2Ceil(mregRows + 1).W))

  val vstoreState    = RegInit(sVecIdle)
  val vstoreMregId   = Reg(UInt(mregP.mregIdBits.W))
  val vstoreVmemBase = Reg(UInt(vmemP.lineAddrBits.W))
  val vstoreCounter  = RegInit(0.U(log2Ceil(mregRows + 1).W))

  io.vloadBusy  := vloadState =/= sVecIdle
  io.vstoreBusy := vstoreState =/= sVecIdle
  io.vecBusy    := io.vloadBusy || io.vstoreBusy

  assert(!(issueVloadCmd && io.vloadBusy),
    "ASSERT FAIL: VLOAD issued while VLOAD path is busy")
  assert(!(issueVstoreCmd && io.vstoreBusy),
    "ASSERT FAIL: VSTORE issued while VSTORE path is busy")
  assert(!(issueVloadCmd &&
           io.cmd.bits.vmemLineAddr(tensorLineBits - 1, 0) =/= 0.U),
    "ASSERT FAIL: VLOAD base must be 1 KiB aligned for block banking")
  assert(!(issueVstoreCmd &&
           io.cmd.bits.vmemLineAddr(tensorLineBits - 1, 0) =/= 0.U),
    "ASSERT FAIL: VSTORE base must be 1 KiB aligned for block banking")

  io.activeMregRead.valid  := io.vstoreBusy
  io.activeMregRead.bits   := vstoreMregId
  io.activeMregWrite.valid := io.vloadBusy
  io.activeMregWrite.bits  := vloadMregId

  // ── Vector port defaults ──
  val vloadLineAddr  = vloadVmemBase + vloadCounter
  val vstoreLineAddr = vstoreVmemBase + vstoreCounter

  io.vmemVecRead.valid         := false.B
  io.vmemVecRead.bits.bankIdx  := vmemP.getBankIdx(vloadLineAddr)
  io.vmemVecRead.bits.bankAddr := vmemP.getBankAddr(vloadLineAddr)

  io.vmemVecWrite.valid         := false.B
  io.vmemVecWrite.bits.bankIdx  := 0.U
  io.vmemVecWrite.bits.bankAddr := 0.U
  io.vmemVecWrite.bits.data     := 0.U

  io.mregReadReq.valid        := false.B
  io.mregReadReq.bits.mregId  := vstoreMregId
  io.mregReadReq.bits.row     := vstoreCounter(log2Ceil(mregRows) - 1, 0)
  io.mregWriteReq.valid       := false.B
  io.mregWriteReq.bits.mregId := vloadMregId
  io.mregWriteReq.bits.row    := 0.U
  io.mregWriteReq.bits.data   := 0.U

  val vloadIssueRead = WireDefault(false.B)
  switch(vloadState) {
    is(sVecIdle) {
      when(issueVloadCmd) {
        vloadState    := sVecRun
        vloadMregId   := io.cmd.bits.mregBank
        vloadVmemBase := io.cmd.bits.vmemLineAddr
        vloadCounter  := 0.U
      }
    }

    is(sVecRun) {
      vloadIssueRead := true.B
      io.vmemVecRead.valid         := true.B
      io.vmemVecRead.bits.bankIdx  := vmemP.getBankIdx(vloadLineAddr)
      io.vmemVecRead.bits.bankAddr := vmemP.getBankAddr(vloadLineAddr)
      vloadCounter := vloadCounter + 1.U
      when(vloadCounter === (mregRows - 1).U) {
        vloadState := sVecDrain
      }
    }

    is(sVecDrain) {
      vloadState := sVecIdle
    }
  }

  val vloadIssuedRow    = Wire(UInt(log2Ceil(mregRows).W))
  val vloadRespPending  = RegInit(false.B)
  val vloadRespRow      = RegInit(0.U(log2Ceil(mregRows).W))
  vloadIssuedRow := vloadCounter(log2Ceil(mregRows) - 1, 0)

  vloadRespPending := vloadIssueRead
  when(vloadIssueRead) {
    vloadRespRow := vloadIssuedRow
  }

  io.mregWriteReq.valid       := io.vmemVecReadData.valid && vloadRespPending
  io.mregWriteReq.bits.mregId := vloadMregId
  io.mregWriteReq.bits.row    := vloadRespRow
  io.mregWriteReq.bits.data   := io.vmemVecReadData.bits

  val vstoreIssueRead = WireDefault(false.B)
  switch(vstoreState) {
    is(sVecIdle) {
      when(issueVstoreCmd) {
        vstoreState    := sVecRun
        vstoreMregId   := io.cmd.bits.mregBank
        vstoreVmemBase := io.cmd.bits.vmemLineAddr
        vstoreCounter  := 0.U
      }
    }

    is(sVecRun) {
      vstoreIssueRead := true.B
      io.mregReadReq.valid       := true.B
      io.mregReadReq.bits.mregId := vstoreMregId
      io.mregReadReq.bits.row    := vstoreCounter(log2Ceil(mregRows) - 1, 0)
      vstoreCounter := vstoreCounter + 1.U
      when(vstoreCounter === (mregRows - 1).U) {
        vstoreState := sVecDrain
      }
    }

    is(sVecDrain) {
      vstoreState := sVecIdle
    }
  }

  val vstoreRespPending  = RegInit(false.B)
  val vstoreRespLineAddr = RegInit(0.U(vmemP.lineAddrBits.W))

  vstoreRespPending := vstoreIssueRead
  when(vstoreIssueRead) {
    vstoreRespLineAddr := vstoreLineAddr
  }

  io.vmemVecWrite.valid         := io.mregReadResp.valid && vstoreRespPending
  io.vmemVecWrite.bits.bankIdx  := vmemP.getBankIdx(vstoreRespLineAddr)
  io.vmemVecWrite.bits.bankAddr := vmemP.getBankAddr(vstoreRespLineAddr)
  io.vmemVecWrite.bits.data     := io.mregReadResp.bits
}

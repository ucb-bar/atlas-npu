// ============================================================================
// SystolicArraySequencer.scala — 3-slot concurrent sequencer for the SA MXU.
//
// Contains control logic, FSMs, and (de)quantization units.
// Does NOT instantiate the datapath (SystolicArray), weight buffer,
// or accumulation buffer — those live in the top-level wrapper.
//
// Slot architecture:
//   Slot A — mreg read port 0: Compute OR single-port Push.
//   Slot B — mreg read port 1: Push only (concurrent with Slot A compute).
//   Slot C — mreg write ports: Pop only (concurrent with Slots A and B).
//
// Conflict detection prevents hazards on shared resources (weight buffer,
// accum buffer, mreg banks) across concurrently active slots.
//
// Mreg port usage:
//   read port 0 : weight push, accum push FP8, accum push BF16 (lo half),
//                 activation stream (compute)
//   read port 1 : accum push BF16 (hi half), OR push-only via Slot B
//   write port 0: pop FP8 (packed), pop BF16 (lo half)
//   write port 1: pop BF16 (hi half)
//
// Bank reporting:
//   activeReads / activeWrites expose which mreg banks are currently being
//   accessed, so the scalar core can perform direction-aware hazard detection
//   across engines without coarse engine-level stalls.
// ============================================================================

package atlas.sa

import chisel3._
import chisel3.util._
import atlas.common.{SystolicArrayParams, MregParams, MregReadReq, MregWriteReq}
import atlas.mxu.{MxuOp, MxuCmd, ComputeReq, WeightWriteReq,
                   AccBufWriteReq, AccBufReadAddr, AccBufLoadReq, AccBufStoreAddr,
                   FPUtils}

/** Three-slot concurrent sequencer for the systolic-array MXU.
  *
  * @param p      Systolic-array geometry parameters.
  * @param mregP  Tensor register file parameters.
  */
class SystolicArraySequencer(
    p:     SystolicArrayParams,
    mregP: MregParams
) extends Module {

  val io = IO(new Bundle {
    // ── Command input ──
    val cmd = Flipped(Decoupled(new MxuCmd(mregP.mregIdBits)))

    // ── Tensor register file (mreg) ports ──
    val mregReadReq0  = Valid(new MregReadReq(mregP))
    val mregReadResp0 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregReadReq1  = Valid(new MregReadReq(mregP))
    val mregReadResp1 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWriteReq0 = Valid(new MregWriteReq(mregP))
    val mregWriteReq1 = Valid(new MregWriteReq(mregP))

    // ── Weight buffer ──
    val weightWriteReq = Valid(new WeightWriteReq(p.mxu))

    // ── Core (SystolicArray) ──
    val compute = Valid(new ComputeReq(p.mxu))
    val coreOut = Flipped(Valid(Vec(p.cols, UInt(p.outT.ieeeWidth.W))))

    // ── Accumulation buffer ──
    val accComputeWrite    = Valid(new AccBufWriteReq(p.mxu))
    val accComputeReadAddr = Output(new AccBufReadAddr(p.mxu))
    val accComputeReadData = Input(Vec(p.cols, UInt(p.outT.ieeeWidth.W)))
    val accLoadReq         = Valid(new AccBufLoadReq(p.mxu))
    val accStoreAddr       = Output(new AccBufStoreAddr(p.mxu))
    val accStoreData       = Input(Vec(p.cols, UInt(p.outT.ieeeWidth.W)))

    // ── Granular busy signals for scalar-core hazard logic ──
    val compBusy    = Output(Bool())
    val pushBusy    = Output(Bool())
    val popBusy     = Output(Bool())
    val dataBusy    = Output(Bool())
    val computeBusy = Output(Bool())

    // ── Active mreg bank reports for direction-aware TRF tracking ──
    val activeReads  = Output(Vec(2, Valid(UInt(mregP.mregIdBits.W))))
    val activeWrites = Output(Vec(2, Valid(UInt(mregP.mregIdBits.W))))
  })

  // ── Helpers ────────────────────────────────────────────────────────
  private def unpackRow(x: UInt, elemWidth: Int, n: Int): Vec[UInt] =
    VecInit((0 until n).map(i => x((i + 1) * elemWidth - 1, i * elemWidth)))
  private def packMregRow(x: Seq[UInt]): UInt =
    (if (x.nonEmpty) Cat(x.reverse) else 0.U(1.W)).pad(mregP.mregRowBits)

  private val tileRows  = p.accSize
  private val rowBits   = log2Ceil(tileRows)
  private val maxCount  = Seq(p.rows, p.cols, tileRows).max
  private val rowCountW = log2Ceil(maxCount + 1)

  // ==========================================================================
  // (De)quantization — FPUtils converter banks
  // ==========================================================================

  val deqBank   = FPUtils.E4M3ToBF16Bank(p.cols)
  val quantBank = FPUtils.BF16ToE4M3Bank(p.cols)

  // ==========================================================================
  // Port defaults
  // ==========================================================================

  io.weightWriteReq.valid    := false.B
  io.weightWriteReq.bits     := 0.U.asTypeOf(new WeightWriteReq(p.mxu))
  io.compute.valid           := false.B
  io.compute.bits            := 0.U.asTypeOf(new ComputeReq(p.mxu))
  io.accComputeWrite.valid   := false.B
  io.accComputeWrite.bits    := 0.U.asTypeOf(new AccBufWriteReq(p.mxu))
  io.accComputeReadAddr      := 0.U.asTypeOf(new AccBufReadAddr(p.mxu))
  io.accLoadReq.valid        := false.B
  io.accLoadReq.bits         := 0.U.asTypeOf(new AccBufLoadReq(p.mxu))
  io.accStoreAddr            := 0.U.asTypeOf(new AccBufStoreAddr(p.mxu))
  io.mregReadReq0.valid      := false.B
  io.mregReadReq0.bits       := 0.U.asTypeOf(new MregReadReq(mregP))
  io.mregReadReq1.valid      := false.B
  io.mregReadReq1.bits       := 0.U.asTypeOf(new MregReadReq(mregP))
  io.mregWriteReq0.valid     := false.B
  io.mregWriteReq0.bits      := 0.U.asTypeOf(new MregWriteReq(mregP))
  io.mregWriteReq1.valid     := false.B
  io.mregWriteReq1.bits      := 0.U.asTypeOf(new MregWriteReq(mregP))

  // ==========================================================================
  // Command classification
  // ==========================================================================

  val cmdOp = io.cmd.bits.op

  val opIsCompute  = (cmdOp === MxuOp.Matmul || cmdOp === MxuOp.MatmulAcc)
  val opIsPush     = (cmdOp === MxuOp.PushWeight || cmdOp === MxuOp.PushAccFP8)
  val opIsPushBF16 = (cmdOp === MxuOp.PushAccBF16)
  val opIsPop      = (cmdOp === MxuOp.PopAccFP8 || cmdOp === MxuOp.PopAccBF16)

  val hasCompute  = io.cmd.valid && opIsCompute
  val hasPush     = io.cmd.valid && opIsPush
  val hasPushBF16 = io.cmd.valid && opIsPushBF16
  val hasPop      = io.cmd.valid && opIsPop

  // ==========================================================================
  // Slot A — mreg read port 0: Compute OR Push
  // ==========================================================================

  val (aIdle :: aPushSetup :: aPushSteady :: aBF16Setup :: aBF16Steady ::
       aCompSetup :: aCompActive :: aCompDrain :: Nil) = Enum(8)

  val slotAState   = RegInit(aIdle)
  val slotACmd     = Reg(new MxuCmd(mregP.mregIdBits))
  val slotARow     = Reg(UInt(rowCountW.W))
  val slotANextRow = Reg(UInt(rowCountW.W))

  val slotAIdle   = (slotAState === aIdle)
  val slotAIsComp = (slotAState === aCompSetup || slotAState === aCompActive ||
                     slotAState === aCompDrain)
  val slotAIsPush = (slotAState === aPushSetup || slotAState === aPushSteady ||
                     slotAState === aBF16Setup || slotAState === aBF16Steady)

  // ==========================================================================
  // Slot B — mreg read port 1: Push only
  // ==========================================================================

  val bIdle :: bPushSetup :: bPushSteady :: Nil = Enum(3)

  val slotBState   = RegInit(bIdle)
  val slotBCmd     = Reg(new MxuCmd(mregP.mregIdBits))
  val slotBRow     = Reg(UInt(rowCountW.W))
  val slotBNextRow = Reg(UInt(rowCountW.W))

  val slotBIdle = (slotBState === bIdle)

  // ==========================================================================
  // Slot C — mreg write ports: Pop only
  // ==========================================================================

  val cIdle :: cRunning :: Nil = Enum(2)

  val slotCState = RegInit(cIdle)
  val slotCCmd   = Reg(new MxuCmd(mregP.mregIdBits))
  val slotCRow   = Reg(UInt(rowCountW.W))

  val slotCIdle = (slotCState === cIdle)

  // ==========================================================================
  // Busy signals
  // ==========================================================================

  io.compBusy    := !slotAIdle && slotAIsComp
  io.pushBusy    := (!slotAIdle && slotAIsPush) || !slotBIdle
  io.popBusy     := !slotCIdle
  io.dataBusy    := io.pushBusy || io.popBusy
  io.computeBusy := io.compBusy

  // ==========================================================================
  // Active mreg bank reports
  // ==========================================================================

  // Slot A reads mreg during push and compute issue (not during drain).
  val slotAReadingMreg = (slotAState === aPushSetup || slotAState === aPushSteady ||
                          slotAState === aBF16Setup || slotAState === aBF16Steady ||
                          slotAState === aCompSetup || slotAState === aCompActive)

  // BF16 push reads two adjacent banks; during BF16, slot B is guaranteed idle.
  val slotABF16Push = (slotAState === aBF16Setup || slotAState === aBF16Steady)

  io.activeReads(0).valid := slotAReadingMreg
  io.activeReads(0).bits  := slotACmd.mregId
  io.activeReads(1).valid := !slotBIdle || slotABF16Push
  io.activeReads(1).bits  := Mux(slotABF16Push, slotACmd.mregId + 1.U, slotBCmd.mregId)

  io.activeWrites(0).valid := !slotCIdle
  io.activeWrites(0).bits  := slotCCmd.mregId
  io.activeWrites(1).valid := !slotCIdle && slotCCmd.op === MxuOp.PopAccBF16
  io.activeWrites(1).bits  := slotCCmd.mregId + 1.U

  // ==========================================================================
  // Conflict detection
  // ==========================================================================

  // Weight-buffer conflict: push targets same slot as active compute.
  val wbufConflict = slotAIsComp &&
    io.cmd.valid && (cmdOp === MxuOp.PushWeight) &&
    (io.cmd.bits.weightSlot === slotACmd.weightSlot)

  // Accum-buffer conflict: push/pop targets same buffer as active compute.
  val compAccSel = slotACmd.accSel
  val accConflictWithComp = slotAIsComp && io.cmd.valid && (
    ((cmdOp === MxuOp.PushAccFP8 || cmdOp === MxuOp.PushAccBF16) &&
      io.cmd.bits.accSel === compAccSel) ||
    ((cmdOp === MxuOp.PopAccFP8 || cmdOp === MxuOp.PopAccBF16) &&
      io.cmd.bits.accSel === compAccSel)
  )

  // Accum-buffer conflict: Slot A push vs incoming pop on same buffer.
  val slotAPushAcc = slotAIsPush &&
    (slotACmd.op === MxuOp.PushAccFP8 || slotACmd.op === MxuOp.PushAccBF16)
  val accConflictPushPop = slotAPushAcc && hasPop &&
    (io.cmd.bits.accSel === slotACmd.accSel)

  // Accum-buffer conflict: Slot B push vs incoming pop on same buffer.
  val slotBPushAcc = !slotBIdle && (slotBCmd.op === MxuOp.PushAccFP8)
  val accConflictBPop = slotBPushAcc && hasPop &&
    (io.cmd.bits.accSel === slotBCmd.accSel)

  // Mreg bank conflict: pop writes to a bank currently being read.
  val popMregId  = io.cmd.bits.mregId
  val popMregId2 = io.cmd.bits.mregId + 1.U

  val mregBankConflictPop = hasPop && (
    (!slotAIdle && (popMregId === slotACmd.mregId)) ||
    (!slotBIdle && (popMregId === slotBCmd.mregId)) ||
    (cmdOp === MxuOp.PopAccBF16 && (
      (!slotAIdle && (popMregId2 === slotACmd.mregId)) ||
      (!slotBIdle && (popMregId2 === slotBCmd.mregId))
    ))
  )

  // Mreg bank conflict: push reads from a bank currently being written by pop.
  val mregBankConflictPush = (hasPush || hasPushBF16) && !slotCIdle && (
    (io.cmd.bits.mregId === slotCCmd.mregId) ||
    (slotCCmd.op === MxuOp.PopAccBF16 &&
      io.cmd.bits.mregId === (slotCCmd.mregId + 1.U))
  )

  val anyConflict = wbufConflict || accConflictWithComp || accConflictPushPop ||
                    accConflictBPop || mregBankConflictPop || mregBankConflictPush

  // ==========================================================================
  // Command acceptance & routing
  // ==========================================================================

  val canAcceptCompute = opIsCompute && slotAIdle && !anyConflict
  val pushToB          = opIsPush && slotBIdle && !anyConflict
  val pushToA          = opIsPush && !slotBIdle && slotAIdle && !anyConflict
  val canAcceptPush    = pushToB || pushToA
  val canAcceptBF16    = opIsPushBF16 && slotAIdle && slotBIdle && !anyConflict
  val canAcceptPop     = opIsPop && slotCIdle && !anyConflict

  io.cmd.ready := canAcceptCompute || canAcceptPush || canAcceptBF16 || canAcceptPop

  // ==========================================================================
  // Compute pipeline (Slot A)
  // ==========================================================================

  val compNextRow     = Reg(UInt(rowCountW.W))
  val compRowsIssued  = Reg(UInt(rowCountW.W))
  val compRowsSent    = Reg(UInt(rowCountW.W))
  val compRowsWritten = Reg(UInt(rowCountW.W))

  val isAccumulate = (slotACmd.op === MxuOp.MatmulAcc)

  private def driveCoreBeat(rowIdx: UInt): Unit = {
    io.accComputeReadAddr.accSel := slotACmd.accSel
    io.accComputeReadAddr.rowIdx := rowIdx
    io.compute.valid             := true.B
    io.compute.bits.act          := unpackRow(io.mregReadResp0.bits, p.inT.ieeeWidth, p.rows)
    io.compute.bits.psum         := io.accComputeReadData
    io.compute.bits.accumulate   := isAccumulate
    io.compute.bits.weightBufSel := slotACmd.weightSlot
  }

  // ==========================================================================
  // Push helpers (shared by Slot A and Slot B)
  // ==========================================================================

  /** Row limit for a push operation: p.cols for weights, tileRows for accum. */
  private def pushRowLimit(op: MxuOp.Type): UInt =
    Mux(op === MxuOp.PushWeight, p.cols.U, tileRows.U)

  private def drivePushWeight(cmd: MxuCmd, row: UInt, portOut: UInt): Unit = {
    val rowData = unpackRow(portOut, p.inT.ieeeWidth, p.rows)
    io.weightWriteReq.valid           := true.B
    io.weightWriteReq.bits.weightSlot := cmd.weightSlot
    io.weightWriteReq.bits.laneIdx    := row(p.mxu.colIdxBits - 1, 0)
    io.weightWriteReq.bits.data       := rowData
  }

  private def drivePushAccFP8(cmd: MxuCmd, row: UInt, portOut: UInt): Unit = {
    val fp8Row = unpackRow(portOut, 8, p.cols)
    io.accLoadReq.valid       := true.B
    io.accLoadReq.bits.accSel := cmd.accSel
    io.accLoadReq.bits.rowIdx := row(rowBits - 1, 0)
    io.accLoadReq.bits.data   := deqBank(fp8Row)
  }

  private def drivePushAccBF16(cmd: MxuCmd, row: UInt): Unit = {
    val lo = io.mregReadResp0.bits
    val hi = io.mregReadResp1.bits
    val bf16Row = Wire(Vec(p.cols, UInt(16.W)))
    for (i <- 0 until p.cols) {
      if (i < 16) bf16Row(i) := lo((i + 1) * 16 - 1, i * 16)
      else        bf16Row(i) := hi((i - 16 + 1) * 16 - 1, (i - 16) * 16)
    }
    io.accLoadReq.valid       := true.B
    io.accLoadReq.bits.accSel := cmd.accSel
    io.accLoadReq.bits.rowIdx := row(rowBits - 1, 0)
    io.accLoadReq.bits.data   := bf16Row
  }

  private def processPushRow(cmd: MxuCmd, row: UInt, portOut: UInt): Unit = {
    when(cmd.op === MxuOp.PushWeight) {
      drivePushWeight(cmd, row, portOut)
    }.elsewhen(cmd.op === MxuOp.PushAccFP8) {
      drivePushAccFP8(cmd, row, portOut)
    }
  }

  private def issueMregRead(
      port: Valid[MregReadReq], cmd: MxuCmd, row: UInt
  ): Unit = {
    port.valid       := true.B
    port.bits.mregId := cmd.mregId
    port.bits.row    := row
  }

  // ==========================================================================
  // Slot A FSM
  // ==========================================================================

  switch(slotAState) {
    is(aIdle) {
      when(io.cmd.fire) {
        when(canAcceptCompute) {
          slotACmd := io.cmd.bits
          io.mregReadReq0.valid       := true.B
          io.mregReadReq0.bits.mregId := io.cmd.bits.mregId
          io.mregReadReq0.bits.row    := 0.U
          compNextRow      := 1.U
          compRowsIssued   := 1.U
          compRowsSent     := 0.U
          compRowsWritten  := 0.U
          slotAState       := aCompSetup

        }.elsewhen(pushToA) {
          slotACmd     := io.cmd.bits
          slotARow     := 0.U
          slotANextRow := 1.U
          issueMregRead(io.mregReadReq0, io.cmd.bits, 0.U)
          slotAState := aPushSetup

        }.elsewhen(canAcceptBF16) {
          slotACmd     := io.cmd.bits
          slotARow     := 0.U
          slotANextRow := 1.U
          io.mregReadReq0.valid       := true.B
          io.mregReadReq0.bits.mregId := io.cmd.bits.mregId
          io.mregReadReq0.bits.row    := 0.U
          io.mregReadReq1.valid       := true.B
          io.mregReadReq1.bits.mregId := io.cmd.bits.mregId + 1.U
          io.mregReadReq1.bits.row    := 0.U
          slotAState := aBF16Setup
        }
      }
    }

    // ── Push (single-port) via Slot A ──
    is(aPushSetup) {
      processPushRow(slotACmd, slotARow, io.mregReadResp0.bits)
      when(slotANextRow >= pushRowLimit(slotACmd.op)) {
        slotAState := aIdle
      }.otherwise {
        issueMregRead(io.mregReadReq0, slotACmd, slotANextRow)
        slotARow     := slotARow + 1.U
        slotANextRow := slotANextRow + 1.U
        slotAState   := aPushSteady
      }
    }
    is(aPushSteady) {
      processPushRow(slotACmd, slotARow, io.mregReadResp0.bits)
      when(slotANextRow >= pushRowLimit(slotACmd.op)) {
        slotAState := aIdle
      }.otherwise {
        issueMregRead(io.mregReadReq0, slotACmd, slotANextRow)
        slotARow     := slotARow + 1.U
        slotANextRow := slotANextRow + 1.U
      }
    }

    // ── Push BF16 (dual-port) via Slot A ──
    is(aBF16Setup) {
      drivePushAccBF16(slotACmd, slotARow)
      when(slotANextRow >= tileRows.U) {
        slotAState := aIdle
      }.otherwise {
        io.mregReadReq0.valid       := true.B
        io.mregReadReq0.bits.mregId := slotACmd.mregId
        io.mregReadReq0.bits.row    := slotANextRow
        io.mregReadReq1.valid       := true.B
        io.mregReadReq1.bits.mregId := slotACmd.mregId + 1.U
        io.mregReadReq1.bits.row    := slotANextRow
        slotARow     := slotARow + 1.U
        slotANextRow := slotANextRow + 1.U
        slotAState   := aBF16Steady
      }
    }
    is(aBF16Steady) {
      drivePushAccBF16(slotACmd, slotARow)
      when(slotANextRow >= tileRows.U) {
        slotAState := aIdle
      }.otherwise {
        io.mregReadReq0.valid       := true.B
        io.mregReadReq0.bits.mregId := slotACmd.mregId
        io.mregReadReq0.bits.row    := slotANextRow
        io.mregReadReq1.valid       := true.B
        io.mregReadReq1.bits.mregId := slotACmd.mregId + 1.U
        io.mregReadReq1.bits.row    := slotANextRow
        slotARow     := slotARow + 1.U
        slotANextRow := slotANextRow + 1.U
      }
    }

    // ── Compute via Slot A ──
    is(aCompSetup) {
      driveCoreBeat(0.U(rowBits.W))
      compRowsSent := 1.U
      when(tileRows.U > 1.U) {
        io.mregReadReq0.valid       := true.B
        io.mregReadReq0.bits.mregId := slotACmd.mregId
        io.mregReadReq0.bits.row    := compNextRow
        compNextRow    := compNextRow + 1.U
        compRowsIssued := compRowsIssued + 1.U
        slotAState     := aCompActive
      }.otherwise {
        slotAState := aCompDrain
      }
    }
    is(aCompActive) {
      driveCoreBeat(compRowsSent(rowBits - 1, 0))
      compRowsSent := compRowsSent + 1.U
      when(compRowsIssued < tileRows.U) {
        io.mregReadReq0.valid       := true.B
        io.mregReadReq0.bits.mregId := slotACmd.mregId
        io.mregReadReq0.bits.row    := compNextRow
        compNextRow    := compNextRow + 1.U
        compRowsIssued := compRowsIssued + 1.U
      }.otherwise {
        slotAState := aCompDrain
      }
    }
    is(aCompDrain) {
      when(compRowsWritten >= tileRows.U) {
        slotAState := aIdle
      }
    }
  }

  // Compute pipeline writeback — SA outputs arrive in-order.
  when(io.coreOut.valid) {
    io.accComputeWrite.valid       := true.B
    io.accComputeWrite.bits.accSel := slotACmd.accSel
    io.accComputeWrite.bits.rowIdx := compRowsWritten(rowBits - 1, 0)
    io.accComputeWrite.bits.data   := io.coreOut.bits
    compRowsWritten := compRowsWritten + 1.U
  }

  // ==========================================================================
  // Slot B FSM — Push via mreg read port 1
  // ==========================================================================

  switch(slotBState) {
    is(bIdle) {
      when(io.cmd.fire && pushToB) {
        slotBCmd     := io.cmd.bits
        slotBRow     := 0.U
        slotBNextRow := 1.U
        issueMregRead(io.mregReadReq1, io.cmd.bits, 0.U)
        slotBState := bPushSetup
      }
    }

    is(bPushSetup) {
      processPushRow(slotBCmd, slotBRow, io.mregReadResp1.bits)
      when(slotBNextRow >= pushRowLimit(slotBCmd.op)) {
        slotBState := bIdle
      }.otherwise {
        issueMregRead(io.mregReadReq1, slotBCmd, slotBNextRow)
        slotBRow     := slotBRow + 1.U
        slotBNextRow := slotBNextRow + 1.U
        slotBState   := bPushSteady
      }
    }

    is(bPushSteady) {
      processPushRow(slotBCmd, slotBRow, io.mregReadResp1.bits)
      when(slotBNextRow >= pushRowLimit(slotBCmd.op)) {
        slotBState := bIdle
      }.otherwise {
        issueMregRead(io.mregReadReq1, slotBCmd, slotBNextRow)
        slotBRow     := slotBRow + 1.U
        slotBNextRow := slotBNextRow + 1.U
      }
    }
  }

  // ==========================================================================
  // Slot C FSM — Pop via mreg write ports
  // ==========================================================================

  switch(slotCState) {
    is(cIdle) {
      when(io.cmd.fire && canAcceptPop) {
        slotCCmd   := io.cmd.bits
        slotCRow   := 0.U
        slotCState := cRunning
      }
    }

    is(cRunning) {
      io.accStoreAddr.accSel := slotCCmd.accSel
      io.accStoreAddr.rowIdx := slotCRow(rowBits - 1, 0)

      when(slotCCmd.op === MxuOp.PopAccFP8) {
        val fp8Row = quantBank(io.accStoreData, slotCCmd.scaleE8M0)
        io.mregWriteReq0.valid       := true.B
        io.mregWriteReq0.bits.mregId := slotCCmd.mregId
        io.mregWriteReq0.bits.row    := slotCRow
        io.mregWriteReq0.bits.data   := packMregRow(fp8Row.map(_(7, 0)))
      }.elsewhen(slotCCmd.op === MxuOp.PopAccBF16) {
        io.mregWriteReq0.valid       := true.B
        io.mregWriteReq0.bits.mregId := slotCCmd.mregId
        io.mregWriteReq0.bits.row    := slotCRow
        io.mregWriteReq0.bits.data   := packMregRow(io.accStoreData.take(16).map(_(15, 0)))
        io.mregWriteReq1.valid       := true.B
        io.mregWriteReq1.bits.mregId := slotCCmd.mregId + 1.U
        io.mregWriteReq1.bits.row    := slotCRow
        io.mregWriteReq1.bits.data   := packMregRow(io.accStoreData.drop(16).map(_(15, 0)))
      }

      when(slotCRow + 1.U >= tileRows.U) {
        slotCState := cIdle
      }.otherwise {
        slotCRow := slotCRow + 1.U
      }
    }
  }
}

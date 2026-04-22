// ============================================================================
// InnerProductTreesSequencer.scala — 3-slot concurrent sequencer for the IPT MXU.
//
// Contains control logic, FSMs, and (de)quantization units.
// Does NOT instantiate the datapath (InnerProductTrees), weight buffer,
// or accumulation buffer — those live in InnerProductTreesTop.
//
// Slot architecture:
//   Slot A — mreg read port 0: Compute OR single-port Push.
//   Slot B — mreg read port 1: Push only (concurrent with Slot A compute).
//   Slot C — mreg write ports: Pop only (concurrent with Slots A and B).
//
// Software-scheduled model:
//   The scalar core / compiler guarantees that commands are only issued
//   when the target slot is idle and no resource conflicts exist.
//   No hardware conflict detection or backpressure is performed.
//   Assertions fire on violations to aid debugging.
//
// Compute pipeline overlap:
//   When the compute datapath is pipelined (numPipeCuts > 0), the sequencer
//   allows a new compute to be issued while the previous compute's tail
//   results are still draining out of the pipeline.  A drain counter tracks
//   in-flight writebacks from the old command, and a tagged accSel pipeline
//   ensures each writeback targets the correct accumulation buffer even
//   after slotACmd has been overwritten by the new command.
//
// Bank reporting:
//   activeReads / activeWrites expose which mreg banks are currently being
//   accessed, so the scalar core can perform direction-aware hazard detection
//   across engines without coarse engine-level stalls.
// ============================================================================

package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.{InnerProductTreeParams, MregParams, MregReadReq, MregWriteReq}
import atlas.mxu.{MxuOp, MxuCmd, ComputeReq, WeightWriteReq,
                  AccBufWriteReq, AccBufReadAddr, AccBufLoadReq, AccBufStoreAddr,
                  FPUtils}

/** Three-slot concurrent sequencer for the inner-product-tree MXU.
  *
  * @param p      IPT geometry parameters.
  * @param mregP  Tensor register file parameters.
  */
class InnerProductTreesSequencer(
    p:     InnerProductTreeParams = InnerProductTreeParams(),
    mregP: MregParams             = MregParams()
) extends Module {

  require(p.vecLen * p.inputFmt.ieeeWidth == mregP.mregRowBits)
  require(p.tileRows <= mregP.mregRows)

  val io = IO(new Bundle {
    // ── Command input (fire-and-forget, no backpressure) ──
    val cmd = Flipped(Valid(new MxuCmd(mregP.mregIdBits)))

    // ── Tensor register file (mreg) ports ──
    val mregReadReq0  = Valid(new MregReadReq(mregP))
    val mregReadResp0 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregReadReq1  = Valid(new MregReadReq(mregP))
    val mregReadResp1 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWriteReq0 = Valid(new MregWriteReq(mregP))
    val mregWriteReq1 = Valid(new MregWriteReq(mregP))

    // ── Core (InnerProductTrees) interface ──
    val weightWriteReq = Valid(new WeightWriteReq(p.mxu))
    val compute        = Valid(new ComputeReq(p.mxu))
    val coreOut        = Flipped(Valid(Vec(p.numLanes, UInt(p.outputFmt.ieeeWidth.W))))

    // ── Accumulation buffer interface ──
    val accComputeWrite    = Valid(new AccBufWriteReq(p.mxu))
    val accComputeReadAddr = Output(new AccBufReadAddr(p.mxu))
    val accComputeReadEn   = Output(Bool())
    val accComputeReadData = Input(Vec(p.numLanes, UInt(p.accumFmt.ieeeWidth.W)))
    val accLoadReq         = Valid(new AccBufLoadReq(p.mxu))
    val accStoreAddr       = Output(new AccBufStoreAddr(p.mxu))
    val accStoreReadEn     = Output(Bool())
    val accStoreData       = Input(Vec(p.numLanes, UInt(16.W)))

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

  private val tileRows  = p.tileRows
  private val rowBits   = p.tileRowBits
  private val maxCount  = Seq(p.numLanes, tileRows).max
  private val rowCountW = log2Ceil(maxCount + 1)

  // ==========================================================================
  // (De)quantization — FPUtils converter banks
  // ==========================================================================

  val deqBank   = FPUtils.E4M3ToBF16Bank(p.numLanes)
  val quantBank = FPUtils.BF16ToE4M3Bank(p.numLanes)

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
  io.accComputeReadEn        := false.B
  io.accLoadReq.valid        := false.B
  io.accLoadReq.bits         := 0.U.asTypeOf(new AccBufLoadReq(p.mxu))
  io.accStoreAddr            := 0.U.asTypeOf(new AccBufStoreAddr(p.mxu))
  io.accStoreReadEn          := false.B
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

  val drainPending = RegInit(0.U(rowCountW.W))
  val drainAccSel  = Reg(Bool())
  val isDraining   = drainPending > 0.U

  io.compBusy    := (!slotAIdle && slotAIsComp) || isDraining
  io.pushBusy    := (!slotAIdle && slotAIsPush) || !slotBIdle
  io.popBusy     := !slotCIdle
  io.dataBusy    := io.pushBusy || io.popBusy
  io.computeBusy := io.compBusy

  // ==========================================================================
  // Active mreg bank reports
  // ==========================================================================

  val slotAReadingMreg = (slotAState === aPushSetup || slotAState === aPushSteady ||
                          slotAState === aBF16Setup || slotAState === aBF16Steady ||
                          slotAState === aCompSetup || slotAState === aCompActive)

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
  // Structural hazard assertions (debug only — do not gate execution)
  // ==========================================================================

  // ── Slot availability ──
  assert(!(io.cmd.valid && opIsCompute &&
           !(slotAIdle || slotAState === aCompDrain)),
    "IPT: Compute issued while Slot A is not idle/draining")

  assert(!(io.cmd.valid && opIsPush && !slotAIdle && !slotBIdle),
    "IPT: Push issued while both Slot A and Slot B are busy")

  assert(!(io.cmd.valid && opIsPushBF16 && (!slotAIdle || !slotBIdle)),
    "IPT: PushAccBF16 issued while Slot A or Slot B is busy")

  assert(!(io.cmd.valid && opIsPop && !slotCIdle),
    "IPT: Pop issued while Slot C is busy")

  // ── Weight-buffer conflicts ──
  val wbufConflict = slotAIsComp &&
    io.cmd.valid && (cmdOp === MxuOp.PushWeight) &&
    (io.cmd.bits.weightSlot === slotACmd.weightSlot)

  val wbufConflictComputeVsPush = io.cmd.valid && opIsCompute && (
    (!slotBIdle && slotBCmd.op === MxuOp.PushWeight &&
      io.cmd.bits.weightSlot === slotBCmd.weightSlot) ||
    (slotAIsPush && slotACmd.op === MxuOp.PushWeight &&
      io.cmd.bits.weightSlot === slotACmd.weightSlot)
  )

  assert(!wbufConflict,
    "IPT: PushWeight targets same weight slot as active compute")
  assert(!wbufConflictComputeVsPush,
    "IPT: Compute targets same weight slot as active push")

  // ── Accum-buffer conflicts ──
  val compAccSel = slotACmd.accSel
  val accConflictWithComp = slotAIsComp && io.cmd.valid && (
    ((cmdOp === MxuOp.PushAccFP8 || cmdOp === MxuOp.PushAccBF16) &&
      io.cmd.bits.accSel === compAccSel) ||
    ((cmdOp === MxuOp.PopAccFP8 || cmdOp === MxuOp.PopAccBF16) &&
      io.cmd.bits.accSel === compAccSel)
  )

  val slotAPushAcc = slotAIsPush &&
    (slotACmd.op === MxuOp.PushAccFP8 || slotACmd.op === MxuOp.PushAccBF16)
  val accConflictPushPop = slotAPushAcc && hasPop &&
    (io.cmd.bits.accSel === slotACmd.accSel)

  val slotBPushAcc = !slotBIdle && (slotBCmd.op === MxuOp.PushAccFP8)
  val accConflictBPop = slotBPushAcc && hasPop &&
    (io.cmd.bits.accSel === slotBCmd.accSel)

  assert(!accConflictWithComp,
    "IPT: Push/Pop targets same accum buffer as active compute")
  assert(!accConflictPushPop,
    "IPT: Pop targets same accum buffer as active Slot A push")
  assert(!accConflictBPop,
    "IPT: Pop targets same accum buffer as active Slot B push")

  // ── Mreg bank conflicts ──
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

  val mregBankConflictPush = (hasPush || hasPushBF16) && !slotCIdle && (
    (io.cmd.bits.mregId === slotCCmd.mregId) ||
    (slotCCmd.op === MxuOp.PopAccBF16 &&
      io.cmd.bits.mregId === (slotCCmd.mregId + 1.U))
  )

  assert(!mregBankConflictPop,
    "IPT: Pop writes to mreg bank currently being read")
  assert(!mregBankConflictPush,
    "IPT: Push reads from mreg bank currently being written by pop")

  // ── Drain conflict ──
  val drainAccConflict = isDraining && io.cmd.valid && (
    (opIsCompute && io.cmd.bits.accSel === drainAccSel) ||
    ((cmdOp === MxuOp.PushAccFP8 || cmdOp === MxuOp.PushAccBF16) &&
      io.cmd.bits.accSel === drainAccSel) ||
    ((cmdOp === MxuOp.PopAccFP8 || cmdOp === MxuOp.PopAccBF16) &&
      io.cmd.bits.accSel === drainAccSel)
  )

  assert(!drainAccConflict,
    "IPT: Command targets accum buffer still receiving drain writebacks")

  // ==========================================================================
  // Command routing (software guarantees no conflicts)
  // ==========================================================================

  val acceptCompute = io.cmd.valid && opIsCompute
  val pushToB       = io.cmd.valid && opIsPush && slotBIdle
  val pushToA       = io.cmd.valid && opIsPush && !slotBIdle
  val acceptBF16    = io.cmd.valid && opIsPushBF16
  val acceptPop     = io.cmd.valid && opIsPop

  // ==========================================================================
  // Compute pipeline (Slot A)
  // ==========================================================================

  private val tagDepth = p.numPipeCuts + 1
  val rowTagData = Reg(Vec(tagDepth, UInt(rowBits.W)))
  for (i <- (tagDepth - 1) to 1 by -1) { rowTagData(i) := rowTagData(i - 1) }
  rowTagData(0) := 0.U

  val rowTagAccSel = Reg(Vec(tagDepth, Bool()))
  for (i <- (tagDepth - 1) to 1 by -1) { rowTagAccSel(i) := rowTagAccSel(i - 1) }
  rowTagAccSel(0) := false.B

  val compNextRow     = Reg(UInt(rowBits.W))
  val compRowsIssued  = Reg(UInt(rowCountW.W))
  val compRowsSent    = Reg(UInt(rowCountW.W))
  val compRowsWritten = Reg(UInt(rowCountW.W))
  val overlapStartCompute =
    (slotAState === aCompDrain) && acceptCompute && !isDraining

  val isAccumulate = (slotACmd.op === MxuOp.MatmulAcc)

  private def driveCoreBeat(rowIdx: UInt): Unit = {
    io.accComputeReadAddr.accSel := slotACmd.accSel
    io.compute.valid             := true.B
    io.compute.bits.act          := unpackRow(io.mregReadResp0.bits, p.inputFmt.ieeeWidth, p.vecLen)
    io.compute.bits.psum         := io.accComputeReadData
    io.compute.bits.accumulate   := isAccumulate
    io.compute.bits.weightBufSel := slotACmd.weightSlot
    rowTagData(0)   := rowIdx
    rowTagAccSel(0) := slotACmd.accSel
  }

  private def issueAccComputeRead(cmd: MxuCmd, row: UInt): Unit = {
    io.accComputeReadAddr.accSel := cmd.accSel
    io.accComputeReadAddr.rowIdx := row(rowBits - 1, 0)
    io.accComputeReadEn          := true.B
  }

  private def issueAccStoreRead(cmd: MxuCmd, row: UInt): Unit = {
    io.accStoreAddr.accSel := cmd.accSel
    io.accStoreAddr.rowIdx := row(rowBits - 1, 0)
    io.accStoreReadEn      := true.B
  }

  // ==========================================================================
  // Push helpers (shared by Slot A and Slot B)
  // ==========================================================================

  private def pushRowLimit(op: MxuOp.Type): UInt =
    Mux(op === MxuOp.PushWeight, p.numLanes.U, tileRows.U)

  private def drivePushWeight(cmd: MxuCmd, row: UInt, portOut: UInt): Unit = {
    val rowData = unpackRow(portOut, p.inputFmt.ieeeWidth, p.vecLen)
    io.weightWriteReq.valid           := true.B
    io.weightWriteReq.bits.weightSlot := cmd.weightSlot
    io.weightWriteReq.bits.laneIdx    := row(p.mxu.colIdxBits - 1, 0)
    io.weightWriteReq.bits.data       := rowData
  }

  private def drivePushAccFP8(cmd: MxuCmd, row: UInt, portOut: UInt): Unit = {
    val rowData = unpackRow(portOut, 8, p.numLanes)
    io.accLoadReq.valid       := true.B
    io.accLoadReq.bits.accSel := cmd.accSel
    io.accLoadReq.bits.rowIdx := row(rowBits - 1, 0)
    io.accLoadReq.bits.data   := deqBank(rowData)
  }

  private def drivePushAccBF16(cmd: MxuCmd, row: UInt): Unit = {
    val lo = io.mregReadResp0.bits
    val hi = io.mregReadResp1.bits
    val bf16Row = Wire(Vec(p.numLanes, UInt(16.W)))
    for (i <- 0 until p.numLanes) {
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

  private def startNewCompute(cmd: MxuCmd): Unit = {
    slotACmd := cmd
    io.mregReadReq0.valid       := true.B
    io.mregReadReq0.bits.mregId := cmd.mregId
    io.mregReadReq0.bits.row    := 0.U
    issueAccComputeRead(cmd, 0.U(rowBits.W))
    compNextRow      := 1.U
    compRowsIssued   := 1.U
    compRowsSent     := 0.U
    compRowsWritten  := 0.U
    slotAState       := aCompSetup
  }

  // ==========================================================================
  // Slot A FSM
  // ==========================================================================

  switch(slotAState) {
    is(aIdle) {
      when(acceptCompute) {
        startNewCompute(io.cmd.bits)

      }.elsewhen(pushToA) {
        slotACmd     := io.cmd.bits
        slotARow     := 0.U
        slotANextRow := 1.U
        issueMregRead(io.mregReadReq0, io.cmd.bits, 0.U)
        slotAState := aPushSetup

      }.elsewhen(acceptBF16) {
        slotACmd     := io.cmd.bits
        slotARow     := 0.U
        slotANextRow := 1.U
        io.mregReadReq0.valid     := true.B
        io.mregReadReq0.bits.mregId := io.cmd.bits.mregId
        io.mregReadReq0.bits.row  := 0.U
        io.mregReadReq1.valid     := true.B
        io.mregReadReq1.bits.mregId := io.cmd.bits.mregId + 1.U
        io.mregReadReq1.bits.row  := 0.U
        slotAState := aBF16Setup
      }
    }

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

    is(aBF16Setup) {
      drivePushAccBF16(slotACmd, slotARow)
      when(slotANextRow >= tileRows.U) {
        slotAState := aIdle
      }.otherwise {
        io.mregReadReq0.valid     := true.B
        io.mregReadReq0.bits.mregId := slotACmd.mregId
        io.mregReadReq0.bits.row  := slotANextRow
        io.mregReadReq1.valid     := true.B
        io.mregReadReq1.bits.mregId := slotACmd.mregId + 1.U
        io.mregReadReq1.bits.row  := slotANextRow
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
        io.mregReadReq0.valid     := true.B
        io.mregReadReq0.bits.mregId := slotACmd.mregId
        io.mregReadReq0.bits.row  := slotANextRow
        io.mregReadReq1.valid     := true.B
        io.mregReadReq1.bits.mregId := slotACmd.mregId + 1.U
        io.mregReadReq1.bits.row  := slotANextRow
        slotARow     := slotARow + 1.U
        slotANextRow := slotANextRow + 1.U
      }
    }

    is(aCompSetup) {
      driveCoreBeat(0.U(rowBits.W))
      compRowsSent := 1.U
      when(tileRows.U > 1.U) {
        io.mregReadReq0.valid     := true.B
        io.mregReadReq0.bits.mregId := slotACmd.mregId
        io.mregReadReq0.bits.row  := compNextRow
        issueAccComputeRead(slotACmd, compNextRow)
        compNextRow    := compNextRow + 1.U
        compRowsIssued := compRowsIssued + 1.U
        slotAState     := aCompActive
      }.otherwise {
        drainAccSel := slotACmd.accSel
        slotAState  := aCompDrain
      }
    }
    is(aCompActive) {
      driveCoreBeat(compRowsSent(rowBits - 1, 0))
      compRowsSent := compRowsSent + 1.U
      when(compRowsIssued < tileRows.U) {
        io.mregReadReq0.valid     := true.B
        io.mregReadReq0.bits.mregId := slotACmd.mregId
        io.mregReadReq0.bits.row  := compNextRow
        issueAccComputeRead(slotACmd, compNextRow)
        compNextRow    := compNextRow + 1.U
        compRowsIssued := compRowsIssued + 1.U
      }.otherwise {
        drainAccSel := slotACmd.accSel
        slotAState  := aCompDrain
      }
    }
    is(aCompDrain) {
      when(acceptCompute && !isDraining) {
        // Accept a new compute before falling back to idle. This keeps a
        // command landing on the exact drain-complete boundary from being
        // silently dropped.
        drainPending := tileRows.U - compRowsWritten
        drainAccSel  := slotACmd.accSel
        startNewCompute(io.cmd.bits)
      }.elsewhen(compRowsWritten >= tileRows.U && !isDraining) {
        slotAState := aIdle
      }
    }
  }

  when(io.coreOut.valid) {
    val writeRow    = rowTagData(tagDepth - 1)
    val writeAccSel = rowTagAccSel(tagDepth - 1)

    io.accComputeWrite.valid       := true.B
    io.accComputeWrite.bits.accSel := writeAccSel
    io.accComputeWrite.bits.rowIdx := writeRow
    io.accComputeWrite.bits.data   := io.coreOut.bits

    when(overlapStartCompute) {
      drainPending := tileRows.U - compRowsWritten - 1.U
    }.elsewhen(drainPending > 0.U) {
      drainPending := drainPending - 1.U
    }.otherwise {
      compRowsWritten := compRowsWritten + 1.U
    }
  }

  // ==========================================================================
  // Slot B FSM — Push via mreg read port 1
  // ==========================================================================

  switch(slotBState) {
    is(bIdle) {
      when(pushToB) {
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
      when(acceptPop) {
        slotCCmd   := io.cmd.bits
        slotCRow   := 0.U
        issueAccStoreRead(io.cmd.bits, 0.U(rowBits.W))
        slotCState := cRunning
      }
    }

    is(cRunning) {
      when(slotCCmd.op === MxuOp.PopAccFP8) {
        val fp8Row = quantBank(io.accStoreData, slotCCmd.scaleE8M0)
        io.mregWriteReq0.valid      := true.B
        io.mregWriteReq0.bits.mregId := slotCCmd.mregId
        io.mregWriteReq0.bits.row   := slotCRow
        io.mregWriteReq0.bits.data  := packMregRow(fp8Row.map(_.pad(8)))

      }.elsewhen(slotCCmd.op === MxuOp.PopAccBF16) {
        io.mregWriteReq0.valid      := true.B
        io.mregWriteReq0.bits.mregId := slotCCmd.mregId
        io.mregWriteReq0.bits.row   := slotCRow
        io.mregWriteReq0.bits.data  := packMregRow(io.accStoreData.take(16))
        io.mregWriteReq1.valid      := true.B
        io.mregWriteReq1.bits.mregId := slotCCmd.mregId + 1.U
        io.mregWriteReq1.bits.row   := slotCRow
        io.mregWriteReq1.bits.data  := packMregRow(io.accStoreData.drop(16))
      }

      when(slotCRow + 1.U >= tileRows.U) {
        slotCState := cIdle
      }.otherwise {
        issueAccStoreRead(slotCCmd, slotCRow + 1.U)
        slotCRow := slotCRow + 1.U
      }
    }
  }
}

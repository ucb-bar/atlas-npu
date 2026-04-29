// ============================================================================
// InnerProductTreesSequencer.scala — Port-aligned sequencer for the IPT MXU.
//
// Architecture: one FSM per matrix-register-file port —
//   ReadP0  (mreg read port 0):  compute feed, single-port push, BF16 lo half
//   ReadP1  (mreg read port 1):  single-port push (preferred), BF16 hi half
//   WriteP0 (mreg write port 0): pop FP8, pop BF16 lo half
//   WriteP1 (mreg write port 1): pop BF16 hi half
//
// Plus an in-flight FIFO of {accSel, rowsWritten} entries: the head advances
// on every coreOut.valid; the entry pops when all `tileRows` writebacks have
// landed. This same FIFO drives writeback routing (which buffer/row receives
// the row emerging this cycle?) and the per-buffer hazard signal
// `accBufRow0Ready` (any in-flight entry with `rowsWritten === 0` blocks new
// same-buffer ops).
//
// Software-scheduled execution model: the compiler / scalar core guarantees
// no resource conflicts. Every reject condition has a named assertion to make
// any contract violation a loud sim-time failure (silicon strips assertions —
// behavior under software contract violation is undefined). The cmd interface
// is fire-and-forget (`Flipped(Valid)`); a violated cmd is blocked AND fires
// a loud assertion.
//
// Back-to-back issue: a new compute can be accepted on the last feed cycle
// of the current compute (no intermediate idle / drain state). This collapses
// the 1-cycle issue bubble and lifts datapath utilization from 32/33 to 32/32
// on back-to-back matmul chains targeting alternating accum buffers.
// ============================================================================

package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.{InnerProductTreeParams, MregParams, MregReadReq, MregWriteReq}
import atlas.mxu.{MxuOp, MxuCmd, ComputeReq, WeightWriteReq,
                  AccBufWriteReq, AccBufReadAddr, AccBufLoadReq, AccBufStoreAddr,
                  FPUtils}

/** Port-aligned concurrent sequencer for the inner-product-tree MXU.
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
  require(p.numPipeCuts + 1 < p.tileRows,
    s"IPT: pipeline latency (${p.numPipeCuts + 1}) must be < tileRows (${p.tileRows}) " +
    s"for inflightDepth=2 to suffice")

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

  val cmdOp        = io.cmd.bits.op
  val cmdAccBuf    = io.cmd.bits.accSel
  val cmdWslot     = io.cmd.bits.weightSlot

  val isCompute    = (cmdOp === MxuOp.Matmul) || (cmdOp === MxuOp.MatmulAcc)
  val isPushW      = (cmdOp === MxuOp.PushWeight)
  val isPushAccFP8 = (cmdOp === MxuOp.PushAccFP8)
  val isPushBF16   = (cmdOp === MxuOp.PushAccBF16)
  val isPush       = isPushW || isPushAccFP8                         // single-port push
  val isPopFP8     = (cmdOp === MxuOp.PopAccFP8)
  val isPopBF16    = (cmdOp === MxuOp.PopAccBF16)
  val isPop        = isPopFP8 || isPopBF16

  // ==========================================================================
  // FSM state declarations
  // ==========================================================================

  // ReadP0 — mreg read port 0
  val (p0Idle :: p0Push :: p0BF16Lo :: p0Comp :: Nil) = Enum(4)
  val p0State   = RegInit(p0Idle)
  val p0Cmd     = Reg(new MxuCmd(mregP.mregIdBits))
  val p0Row     = Reg(UInt(rowCountW.W))
  val p0NextRow = Reg(UInt(rowCountW.W))

  // ReadP1 — mreg read port 1
  val (p1Idle :: p1Push :: p1BF16Hi :: Nil) = Enum(3)
  val p1State   = RegInit(p1Idle)
  val p1Cmd     = Reg(new MxuCmd(mregP.mregIdBits))
  val p1Row     = Reg(UInt(rowCountW.W))
  val p1NextRow = Reg(UInt(rowCountW.W))

  // WriteP0 — mreg write port 0
  val (w0Idle :: w0PopFP8 :: w0PopBF16Lo :: Nil) = Enum(3)
  val w0State = RegInit(w0Idle)
  val w0Cmd   = Reg(new MxuCmd(mregP.mregIdBits))
  val w0Row   = Reg(UInt(rowCountW.W))

  // WriteP1 — mreg write port 1
  val (w1Idle :: w1PopBF16Hi :: Nil) = Enum(2)
  val w1State = RegInit(w1Idle)
  val w1Cmd   = Reg(new MxuCmd(mregP.mregIdBits))
  val w1Row   = Reg(UInt(rowCountW.W))

  val p0Idl = p0State === p0Idle
  val p1Idl = p1State === p1Idle
  val w0Idl = w0State === w0Idle
  val w1Idl = w1State === w1Idle

  // ==========================================================================
  // In-flight FIFO — {accSel, rowsWritten} per in-flight matmul
  //
  //   Entries are added on compute accept and removed when all `tileRows`
  //   row-writebacks have landed. The head is the oldest entry (the matmul
  //   currently being written back). Depth = 2: with back-to-back issue, the
  //   maximum simultaneous in-flight is `ceil(latency / tileRows) + 1` = 2 at
  //   the default config (latency = 2, tileRows = 32). The require above
  //   guards against future configs that would push this past 2.
  // ==========================================================================

  private val inflightDepth = 2
  private val inflightPtrW  = log2Ceil(inflightDepth)

  class InFlightEntry extends Bundle {
    val accSel      = Bool()
    val rowsWritten = UInt((rowBits + 1).W)
  }

  val inflight      = Reg(Vec(inflightDepth, new InFlightEntry))
  val inflightValid = RegInit(VecInit(Seq.fill(inflightDepth)(false.B)))
  val ifHead        = RegInit(0.U(inflightPtrW.W))
  val ifTail        = RegInit(0.U(inflightPtrW.W))

  val fifoFull           = inflightValid.asUInt.andR
  val anyComputeInFlight = inflightValid.asUInt.orR

  // FIFO-derived per-buffer hazard: a buffer has an in-flight compute that
  // hasn't yet completed its row-0 writeback.
  val accBufRow0Ready = VecInit.tabulate(2) { buf =>
    !(0 until inflightDepth).map { i =>
      inflightValid(i) &&
        (inflight(i).accSel === buf.U) &&
        (inflight(i).rowsWritten === 0.U)
    }.reduce(_ || _)
  }

  // ==========================================================================
  // Per-target "active" indicators for full hazard coverage
  //
  //   These cover every multi-cycle op currently touching a given wslot or
  //   accSel. Used to gate (and assert) all real concurrent-write / partial-
  //   read races: simultaneous pushes to the same target, compute starting
  //   while a push to its target is still streaming, etc.
  // ==========================================================================

  // Strictly necessary: a matmul "is reading" wslot w only on cycles where it
  // will *continue* to read wslot w next cycle. On the matmul's last p0Comp
  // cycle (p0NextRow >= tileRows), the lane has already sampled the final
  // wbuf row this cycle; a PUSH dispatched on that same cycle does not drive
  // weightWriteReq until the next cycle (when p1 enters p1Push), and the
  // first wbuf register write commits at posedge cycle+2. By then the matmul
  // has exited p0Comp and stopped reading. So we can let PUSH fire on the
  // last cycle without corrupting the matmul's reads.
  val wslotComputeReading = VecInit.tabulate(2) { w =>
    (p0State === p0Comp) && (p0Cmd.weightSlot === w.U) && (p0NextRow < tileRows.U)
  }
  // Strictly necessary (symmetric to wslotComputeReading above): a PUSH
  // "is writing" wslot w only on cycles where it will continue to drive a
  // new wbuf column next cycle. On the push's last cycle (pXNextRow ==
  // numLanes), the last column has been driven and will be latched at the
  // next clock edge. A MATMUL on wslot w dispatched this same cycle has
  // its first feed beat next cycle, by which time wbuf is fully coherent.
  // So we let MATMUL fire on the push's last cycle without reading stale
  // wbuf state.
  val pushWActive = VecInit.tabulate(2) { w =>
    ((p0State === p0Push) && (p0Cmd.op === MxuOp.PushWeight) &&
      (p0Cmd.weightSlot === w.U) && (p0NextRow < p.numLanes.U)) ||
    ((p1State === p1Push) && (p1Cmd.op === MxuOp.PushWeight) &&
      (p1Cmd.weightSlot === w.U) && (p1NextRow < p.numLanes.U))
  }
  val pushAccActive = VecInit.tabulate(2) { buf =>
    ((p0State === p0Push)   && (p0Cmd.op === MxuOp.PushAccFP8) && (p0Cmd.accSel === buf.U)) ||
    ((p1State === p1Push)   && (p1Cmd.op === MxuOp.PushAccFP8) && (p1Cmd.accSel === buf.U)) ||
    ((p0State === p0BF16Lo) && (p0Cmd.accSel === buf.U))   // BF16 push runs p0/p1 in lockstep
  }

  // ==========================================================================
  // Cmd dispatch — combinational accept signals
  //
  //   The dispatcher is the single point that decides routing and gates
  //   commands behind hazards. Every accepted cmd flows into exactly the
  //   FSM(s) it belongs to. Push prefers ReadP1 (keeps ReadP0 free for
  //   compute); falls back to ReadP0 if ReadP1 is busy.
  //
  //   Back-to-back issue: a new compute can be accepted on the last feed
  //   cycle of the current compute (p0CompLastCycle), eliminating the 1-cycle
  //   bubble. Encoded in p0Available so the FSM doesn't need a special case.
  // ==========================================================================

  val accReuseHazard = !accBufRow0Ready(cmdAccBuf)

  val pushWHazard          = isPushW       && wslotComputeReading(cmdWslot)
  val pushWSelfHazard      = isPushW       && pushWActive(cmdWslot)
  val pushAccFP8SelfHazard = isPushAccFP8  && pushAccActive(cmdAccBuf)
  val computeWslotHazard   = isCompute     && pushWActive(io.cmd.bits.weightSlot)
  val computePushAccHazard = isCompute     && pushAccActive(cmdAccBuf)

  val p0CompLastCycle = (p0State === p0Comp) && (p0NextRow >= tileRows.U)
  val p0Available     = p0Idl || p0CompLastCycle

  val acceptCompute  = io.cmd.valid && isCompute && p0Available && !fifoFull &&
                       !accReuseHazard && !computeWslotHazard && !computePushAccHazard
  val acceptPushP1   = io.cmd.valid && isPush && p1Idl &&
                       Mux(isPushW, !pushWHazard && !pushWSelfHazard,
                                    !accReuseHazard && !pushAccFP8SelfHazard)
  val acceptPushP0   = io.cmd.valid && isPush && !p1Idl && p0Idl &&
                       Mux(isPushW, !pushWHazard && !pushWSelfHazard,
                                    !accReuseHazard && !pushAccFP8SelfHazard)
  val acceptBF16Push = io.cmd.valid && isPushBF16 && p0Idl && p1Idl && !accReuseHazard
  val acceptPopFP8   = io.cmd.valid && isPopFP8 && w0Idl && !accReuseHazard
  val acceptPopBF16  = io.cmd.valid && isPopBF16 && w0Idl && w1Idl && !accReuseHazard

  // ==========================================================================
  // Hazard / structural assertions (debug only — do not gate execution).
  // The dispatch logic above already blocks unsafe cmds; these asserts make
  // any blocked cmd a loud failure so the bug surfaces at the source.
  // ==========================================================================

  // Compute
  assert(!(io.cmd.valid && isCompute && fifoFull),
    "IPT: Compute issued while in-flight FIFO is full")
  assert(!(io.cmd.valid && isCompute && !p0Available),
    "IPT: Compute issued while ReadP0 is busy and not on its last feed cycle")
  assert(!(io.cmd.valid && isCompute && accReuseHazard),
    "IPT: Compute issued before previous compute's row-0 writeback on same accSel")
  assert(!(io.cmd.valid && isCompute && computeWslotHazard),
    "IPT: Compute targets wslot currently being written by an active push")
  assert(!(io.cmd.valid && isCompute && computePushAccHazard),
    "IPT: Compute targets accSel currently being written by an active push-acc")

  // Push (single-port: weight or FP8 acc)
  assert(!(io.cmd.valid && isPush && !p0Idl && !p1Idl),
    "IPT: Push issued while both ReadP0 and ReadP1 are busy")
  assert(!(io.cmd.valid && isPushW && pushWHazard),
    "IPT: PushWeight targets wslot currently being read by compute feed")
  assert(!(io.cmd.valid && isPushW && pushWSelfHazard),
    "IPT: PushWeight targets wslot already being written by another active PushWeight")
  assert(!(io.cmd.valid && isPushAccFP8 && accReuseHazard),
    "IPT: PushAccFP8 issued before previous compute's row-0 writeback on same accSel")
  assert(!(io.cmd.valid && isPushAccFP8 && pushAccFP8SelfHazard),
    "IPT: PushAccFP8 targets accSel already being written by another active push-acc")

  // PushAccBF16 (two-port)
  assert(!(io.cmd.valid && isPushBF16 && (!p0Idl || !p1Idl)),
    "IPT: PushAccBF16 issued while ReadP0 or ReadP1 is busy")
  assert(!(io.cmd.valid && isPushBF16 && accReuseHazard),
    "IPT: PushAccBF16 issued before previous compute's row-0 writeback on same accSel")

  // Pop
  assert(!(io.cmd.valid && isPopFP8 && !w0Idl),
    "IPT: PopAccFP8 issued while WriteP0 is busy")
  assert(!(io.cmd.valid && isPopBF16 && (!w0Idl || !w1Idl)),
    "IPT: PopAccBF16 issued while WriteP0 or WriteP1 is busy")
  assert(!(io.cmd.valid && isPop && accReuseHazard),
    "IPT: Pop issued before previous compute's row-0 writeback on same accSel")

  // ── Mreg bank conflicts ──
  val popMregId  = io.cmd.bits.mregId
  val popMregId2 = io.cmd.bits.mregId + 1.U

  val mregBankConflictPop = io.cmd.valid && isPop && (
    (!p0Idl && (popMregId === p0Cmd.mregId)) ||
    (!p1Idl && (popMregId === p1Cmd.mregId)) ||
    (isPopBF16 && (
      (!p0Idl && (popMregId2 === p0Cmd.mregId)) ||
      (!p1Idl && (popMregId2 === p1Cmd.mregId))
    ))
  )

  val mregBankConflictPush = io.cmd.valid && (isPush || isPushBF16) && (
    (!w0Idl && (io.cmd.bits.mregId === w0Cmd.mregId)) ||
    (!w1Idl && (io.cmd.bits.mregId === w1Cmd.mregId)) ||
    (isPushBF16 && (
      (!w0Idl && ((io.cmd.bits.mregId + 1.U) === w0Cmd.mregId)) ||
      (!w1Idl && ((io.cmd.bits.mregId + 1.U) === w1Cmd.mregId))
    ))
  )

  assert(!mregBankConflictPop,
    "IPT: Pop writes to mreg bank currently being read by an active push/compute")
  assert(!mregBankConflictPush,
    "IPT: Push reads from mreg bank currently being written by an active pop")

  // ==========================================================================
  // Port helpers
  // ==========================================================================

  private def issueP0Read(mregId: UInt, row: UInt): Unit = {
    io.mregReadReq0.valid       := true.B
    io.mregReadReq0.bits.mregId := mregId
    io.mregReadReq0.bits.row    := row
  }

  private def issueP1Read(mregId: UInt, row: UInt): Unit = {
    io.mregReadReq1.valid       := true.B
    io.mregReadReq1.bits.mregId := mregId
    io.mregReadReq1.bits.row    := row
  }

  private def driveP0AccRead(cmd: MxuCmd, row: UInt): Unit = {
    io.accComputeReadAddr.accSel := cmd.accSel
    io.accComputeReadAddr.rowIdx := row(rowBits - 1, 0)
    io.accComputeReadEn          := (cmd.op === MxuOp.MatmulAcc)
  }

  private def issueAccStoreRead(cmd: MxuCmd, row: UInt): Unit = {
    io.accStoreAddr.accSel := cmd.accSel
    io.accStoreAddr.rowIdx := row(rowBits - 1, 0)
    io.accStoreReadEn      := true.B
  }

  private def pushRowLimit(op: MxuOp.Type): UInt =
    Mux(op === MxuOp.PushWeight, p.numLanes.U, tileRows.U)

  // Drive the per-row push outputs for the row whose mreg response just
  // arrived.  PushWeight writes one weight-buffer column; PushAccFP8
  // dequantizes and drives one accBuf load.
  private def processPushRow(cmd: MxuCmd, row: UInt, mregResp: UInt): Unit = {
    when(cmd.op === MxuOp.PushWeight) {
      io.weightWriteReq.valid           := true.B
      io.weightWriteReq.bits.weightSlot := cmd.weightSlot
      io.weightWriteReq.bits.laneIdx    := row(p.mxu.colIdxBits - 1, 0)
      io.weightWriteReq.bits.data       := unpackRow(mregResp, p.inputFmt.ieeeWidth, p.vecLen)
    }.elsewhen(cmd.op === MxuOp.PushAccFP8) {
      val fp8Row = unpackRow(mregResp, 8, p.numLanes)
      io.accLoadReq.valid       := true.B
      io.accLoadReq.bits.accSel := cmd.accSel
      io.accLoadReq.bits.rowIdx := row(rowBits - 1, 0)
      io.accLoadReq.bits.data   := deqBank(fp8Row)
    }
  }

  // Start (or restart) a compute feed: latch the cmd, issue the row-0 mreg
  // read, drive the row-0 accum-buffer read for accumulate, and reset row
  // counters. p0State stays / becomes p0Comp.
  private def startComputeFeed(cmd: MxuCmd): Unit = {
    p0Cmd := cmd
    issueP0Read(cmd.mregId, 0.U)
    driveP0AccRead(cmd, 0.U)
    p0Row     := 0.U
    p0NextRow := 1.U
    p0State   := p0Comp
  }

  // ==========================================================================
  // ReadP0 FSM — mreg read port 0
  //
  //   States: idle / push (single-port) / bf16 lo half / compute feed.
  //   Idle accepts compute, single-port push (when ReadP1 is busy), or BF16
  //   push (alongside ReadP1). Each non-idle state streams 32 mreg reads and
  //   processes their responses. The compute state additionally allows
  //   back-to-back issue: a new compute accepted on the last feed cycle
  //   reuses this FSM directly via startComputeFeed (no transition out).
  // ==========================================================================

  switch(p0State) {
    is(p0Idle) {
      when(acceptCompute) {
        startComputeFeed(io.cmd.bits)

      }.elsewhen(acceptPushP0) {
        p0Cmd := io.cmd.bits
        issueP0Read(io.cmd.bits.mregId, 0.U)
        p0Row     := 0.U
        p0NextRow := 1.U
        p0State   := p0Push

      }.elsewhen(acceptBF16Push) {
        p0Cmd := io.cmd.bits
        issueP0Read(io.cmd.bits.mregId, 0.U)
        p0Row     := 0.U
        p0NextRow := 1.U
        p0State   := p0BF16Lo
      }
    }

    is(p0Push) {
      processPushRow(p0Cmd, p0Row, io.mregReadResp0.bits)

      when(p0NextRow >= pushRowLimit(p0Cmd.op)) {
        p0State := p0Idle
      }.otherwise {
        issueP0Read(p0Cmd.mregId, p0NextRow)
        p0Row     := p0Row + 1.U
        p0NextRow := p0NextRow + 1.U
      }
    }

    is(p0BF16Lo) {
      // Combine lo (this port's response) with hi (ReadP1's response) and
      // drive a single accLoadReq.  Both FSMs run in lockstep on identical
      // counters, so io.mregReadResp1 here is the hi half for the same row.
      val lo = io.mregReadResp0.bits
      val hi = io.mregReadResp1.bits
      val bf16Row = Wire(Vec(p.numLanes, UInt(16.W)))
      for (i <- 0 until p.numLanes) {
        if (i < 16) bf16Row(i) := lo((i + 1) * 16 - 1, i * 16)
        else        bf16Row(i) := hi((i - 16 + 1) * 16 - 1, (i - 16) * 16)
      }
      io.accLoadReq.valid       := true.B
      io.accLoadReq.bits.accSel := p0Cmd.accSel
      io.accLoadReq.bits.rowIdx := p0Row(rowBits - 1, 0)
      io.accLoadReq.bits.data   := bf16Row

      when(p0NextRow >= tileRows.U) {
        p0State := p0Idle
      }.otherwise {
        issueP0Read(p0Cmd.mregId, p0NextRow)
        p0Row     := p0Row + 1.U
        p0NextRow := p0NextRow + 1.U
      }
    }

    is(p0Comp) {
      // Drive compute beat for row p0Row arriving this cycle.
      io.compute.valid             := true.B
      io.compute.bits.act          := unpackRow(io.mregReadResp0.bits, p.inputFmt.ieeeWidth, p.vecLen)
      io.compute.bits.psum         := Mux(p0Cmd.op === MxuOp.MatmulAcc,
                                          io.accComputeReadData,
                                          VecInit.fill(p.numLanes)(0.U(p.accumFmt.ieeeWidth.W)))
      io.compute.bits.accumulate   := (p0Cmd.op === MxuOp.MatmulAcc)
      io.compute.bits.weightBufSel := p0Cmd.weightSlot

      when(p0NextRow >= tileRows.U) {
        // Last feed cycle. Either accept a new compute (back-to-back issue)
        // or fall back to idle.
        when(acceptCompute) {
          startComputeFeed(io.cmd.bits)
        }.otherwise {
          p0State := p0Idle
        }
      }.otherwise {
        issueP0Read(p0Cmd.mregId, p0NextRow)
        driveP0AccRead(p0Cmd, p0NextRow)
        p0Row     := p0Row + 1.U
        p0NextRow := p0NextRow + 1.U
      }
    }
  }

  // ==========================================================================
  // ReadP1 FSM — mreg read port 1
  //
  //   States: idle / push (single-port) / BF16 hi half. No compute support.
  //   BF16 hi half runs in lockstep with ReadP0's `p0BF16Lo`.
  // ==========================================================================

  switch(p1State) {
    is(p1Idle) {
      when(acceptPushP1) {
        p1Cmd := io.cmd.bits
        issueP1Read(io.cmd.bits.mregId, 0.U)
        p1Row     := 0.U
        p1NextRow := 1.U
        p1State   := p1Push

      }.elsewhen(acceptBF16Push) {
        p1Cmd := io.cmd.bits
        issueP1Read(io.cmd.bits.mregId + 1.U, 0.U)
        p1Row     := 0.U
        p1NextRow := 1.U
        p1State   := p1BF16Hi
      }
    }

    is(p1Push) {
      processPushRow(p1Cmd, p1Row, io.mregReadResp1.bits)

      when(p1NextRow >= pushRowLimit(p1Cmd.op)) {
        p1State := p1Idle
      }.otherwise {
        issueP1Read(p1Cmd.mregId, p1NextRow)
        p1Row     := p1Row + 1.U
        p1NextRow := p1NextRow + 1.U
      }
    }

    is(p1BF16Hi) {
      // Stream hi-half mreg reads in lockstep with ReadP0. ReadP0 drives the
      // combined accLoadReq using io.mregReadResp1 (this port's response).
      when(p1NextRow >= tileRows.U) {
        p1State := p1Idle
      }.otherwise {
        issueP1Read(p1Cmd.mregId + 1.U, p1NextRow)
        p1Row     := p1Row + 1.U
        p1NextRow := p1NextRow + 1.U
      }
    }
  }

  // ==========================================================================
  // WriteP0 FSM — mreg write port 0
  //
  //   States: idle / pop FP8 / pop BF16 lo half.  Drives accStoreAddr/En for
  //   both FP8 and BF16 pops; the latter runs in lockstep with WriteP1 for
  //   the hi half.
  // ==========================================================================

  switch(w0State) {
    is(w0Idle) {
      when(acceptPopFP8) {
        w0Cmd := io.cmd.bits
        issueAccStoreRead(io.cmd.bits, 0.U)
        w0Row   := 0.U
        w0State := w0PopFP8

      }.elsewhen(acceptPopBF16) {
        w0Cmd := io.cmd.bits
        issueAccStoreRead(io.cmd.bits, 0.U)
        w0Row   := 0.U
        w0State := w0PopBF16Lo
      }
    }

    is(w0PopFP8) {
      val fp8Row = quantBank(io.accStoreData, w0Cmd.scaleE8M0)
      io.mregWriteReq0.valid       := true.B
      io.mregWriteReq0.bits.mregId := w0Cmd.mregId
      io.mregWriteReq0.bits.row    := w0Row
      io.mregWriteReq0.bits.data   := packMregRow(fp8Row.map(_.pad(8)))

      when(w0Row + 1.U >= tileRows.U) {
        w0State := w0Idle
      }.otherwise {
        issueAccStoreRead(w0Cmd, w0Row + 1.U)
        w0Row := w0Row + 1.U
      }
    }

    is(w0PopBF16Lo) {
      io.mregWriteReq0.valid       := true.B
      io.mregWriteReq0.bits.mregId := w0Cmd.mregId
      io.mregWriteReq0.bits.row    := w0Row
      io.mregWriteReq0.bits.data   := packMregRow(io.accStoreData.take(16))

      when(w0Row + 1.U >= tileRows.U) {
        w0State := w0Idle
      }.otherwise {
        issueAccStoreRead(w0Cmd, w0Row + 1.U)
        w0Row := w0Row + 1.U
      }
    }
  }

  // ==========================================================================
  // WriteP1 FSM — mreg write port 1
  //
  //   States: idle / pop BF16 hi half (lockstep with WriteP0).
  // ==========================================================================

  switch(w1State) {
    is(w1Idle) {
      when(acceptPopBF16) {
        w1Cmd   := io.cmd.bits
        w1Row   := 0.U
        w1State := w1PopBF16Hi
      }
    }

    is(w1PopBF16Hi) {
      io.mregWriteReq1.valid       := true.B
      io.mregWriteReq1.bits.mregId := w1Cmd.mregId + 1.U
      io.mregWriteReq1.bits.row    := w1Row
      io.mregWriteReq1.bits.data   := packMregRow(io.accStoreData.drop(16))

      when(w1Row + 1.U >= tileRows.U) {
        w1State := w1Idle
      }.otherwise {
        w1Row := w1Row + 1.U
      }
    }
  }

  // ==========================================================================
  // In-flight FIFO update — enqueue on accept, dequeue on last writeback
  // ==========================================================================

  // True when the head's last row writes back this cycle.
  val popThisCycle =
    io.coreOut.valid &&
      inflightValid(ifHead) &&
      (inflight(ifHead).rowsWritten === (tileRows - 1).U)

  // Writeback for the head's current row.
  when(io.coreOut.valid && inflightValid(ifHead)) {
    io.accComputeWrite.valid       := true.B
    io.accComputeWrite.bits.accSel := inflight(ifHead).accSel
    io.accComputeWrite.bits.rowIdx := inflight(ifHead).rowsWritten(rowBits - 1, 0)
    io.accComputeWrite.bits.data   := io.coreOut.bits

    when(popThisCycle) {
      inflightValid(ifHead) := false.B
    }.otherwise {
      inflight(ifHead).rowsWritten := inflight(ifHead).rowsWritten + 1.U
    }
  }

  // Pointer advance — depth 2 is a power of two; explicit wrap kept for
  // symmetry with SA in case depth ever changes.
  when(popThisCycle) {
    ifHead := Mux(ifHead === (inflightDepth - 1).U, 0.U, ifHead + 1.U)
  }

  // Enqueue on compute accept (covers both fresh accepts and back-to-back
  // accepts on the last feed cycle of the previous compute).
  when(acceptCompute) {
    inflight(ifTail).accSel      := io.cmd.bits.accSel
    inflight(ifTail).rowsWritten := 0.U
    inflightValid(ifTail)        := true.B
    ifTail                       := Mux(ifTail === (inflightDepth - 1).U, 0.U, ifTail + 1.U)
  }

  // ==========================================================================
  // Busy / active outputs (preserve current semantics bit-identically)
  // ==========================================================================

  io.compBusy    := (p0State === p0Comp) || anyComputeInFlight
  io.pushBusy    := (p0State === p0Push) || (p0State === p0BF16Lo) || !p1Idl
  io.popBusy     := !w0Idl || !w1Idl
  io.dataBusy    := io.pushBusy || io.popBusy
  io.computeBusy := io.compBusy

  // ── Active mreg bank reports ──
  io.activeReads(0).valid := !p0Idl
  io.activeReads(0).bits  := p0Cmd.mregId
  io.activeReads(1).valid := !p1Idl
  io.activeReads(1).bits  := Mux(p1State === p1BF16Hi, p1Cmd.mregId + 1.U, p1Cmd.mregId)

  io.activeWrites(0).valid := !w0Idl
  io.activeWrites(0).bits  := w0Cmd.mregId
  io.activeWrites(1).valid := !w1Idl
  io.activeWrites(1).bits  := w1Cmd.mregId + 1.U
}

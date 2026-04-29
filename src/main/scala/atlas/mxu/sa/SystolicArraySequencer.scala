package atlas.sa

import chisel3._
import chisel3.util._
import atlas.common.{SystolicArrayParams, MregParams, MregReadReq, MregWriteReq}
import atlas.mxu.{MxuOp, MxuCmd, ComputeReq, WeightWriteReq,
                   AccBufWriteReq, AccBufReadAddr, AccBufLoadReq, AccBufStoreAddr,
                   FPUtils}

/** Port-aligned concurrent sequencer for the systolic-array MXU.
  *
  * Architecture: one FSM per matrix-register-file port —
  *   ReadP0  (mreg read port 0):  compute issue, single-port push, BF16 lo half
  *   ReadP1  (mreg read port 1):  single-port push, BF16 hi half
  *   WriteP0 (mreg write port 0): pop FP8, pop BF16 lo half
  *   WriteP1 (mreg write port 1): pop BF16 hi half
  *
  * Plus one SA pipeline tracker (separate from any port FSM): a small ring
  * buffer of in-flight compute metadata `{accSel, rowsWritten}`. The head
  * advances on every `coreOut.valid`; the entry pops when all `tileRows`
  * writebacks land. The same FIFO drives both writeback routing (where does
  * the row emerging this cycle go?) and the per-buffer hazard signal
  * `accBufRow0Ready` (any in-flight entry with `rowsWritten === 0` blocks
  * new ops on that buffer).
  *
  * Key correctness invariants:
  *   - A new op (compute / push acc / pop) on accSel X waits until X's most
  *     recent compute has issued its row-0 writeback.
  *   - A `VMATPUSH.W` to wslot W waits until all in-flight matmuls reading W
  *     have drained — encoded as a per-wslot countdown reset to
  *     `(rows + cols - 2)` on each matmul issue.
  *   - New compute is rejected (with assertion) if the in-flight FIFO is
  *     full. Fixes the silent compute-drop bug present in the prior sequencer.
  *
  * Software-scheduled execution model: the compiler/scalar-core guarantees
  * no resource conflicts (slot, MREG bank). This module asserts on
  * violations to aid debugging. The cmd interface is fire-and-forget
  * (`Flipped(Valid)`); a violated cmd is blocked AND fires a loud assertion
  * (no silent drop).
  *
  * @param p      Systolic-array geometry parameters.
  * @param mregP  Tensor register file parameters.
  */
class SystolicArraySequencer(
    p:     SystolicArrayParams,
    mregP: MregParams
) extends Module {

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

    // ── Weight buffer ──
    val weightWriteReq = Valid(new WeightWriteReq(p.mxu))

    // ── Core (SystolicArray) ──
    val compute = Valid(new ComputeReq(p.mxu))
    val coreOut = Flipped(Valid(Vec(p.cols, UInt(p.outT.ieeeWidth.W))))

    // ── Accumulation buffer ──
    val accComputeWrite    = Valid(new AccBufWriteReq(p.mxu))
    val accComputeReadAddr = Output(new AccBufReadAddr(p.mxu))
    val accComputeReadEn   = Output(Bool())
    val accComputeReadData = Input(Vec(p.cols, UInt(p.outT.ieeeWidth.W)))
    val accLoadReq         = Valid(new AccBufLoadReq(p.mxu))
    val accStoreAddr       = Output(new AccBufStoreAddr(p.mxu))
    val accStoreReadEn     = Output(Bool())
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

  // PE(i,j) for output row k reads weights[i][j] at T_M+1+k+i+j; lane j's
  // last read (k=i=rows-1) is at T_M + (2*rows-2) + j.  A push lane j writes
  // at T_push+1+j and is visible at T_push+2+j.  The constraint
  // T_push+2+j > T_M + (2*rows-2) + j simplifies to T_push > T_M + 62 (the
  // auto-memory's empirical rule).  Countdown init = (rows + cols - 2)
  // reaches 0 at cycle T_M + 63.
  private val wbufDrainInit = (p.rows + p.cols - 2).U

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
  // SA Pipeline Tracker — in-flight FIFO
  //
  //   inflight(i) holds metadata for the i-th in-flight compute.  Entries
  //   are added on compute accept and removed when all `tileRows` row-
  //   writebacks have landed.  The head is the oldest entry — whoever's
  //   currently being written back.  Depth = 3: at the minimum software
  //   matmul stride of 33 cycles (Slot B push duration), M2 enqueues at
  //   T0+66 while M0's last row-writeback lands at T0+94, so three matmuls
  //   are simultaneously in flight.  The full assertion below catches any
  //   compute that would push depth past 3.
  // ==========================================================================

  private val inflightDepth = 3
  private val inflightPtrW  = log2Ceil(inflightDepth)

  class InFlightEntry extends Bundle {
    val accSel      = Bool()
    val rowsWritten = UInt((rowBits + 1).W)
  }

  val inflight      = Reg(Vec(inflightDepth, new InFlightEntry))
  val inflightValid = RegInit(VecInit(Seq.fill(inflightDepth)(false.B)))
  val ifHead        = RegInit(0.U(inflightPtrW.W))
  val ifTail        = RegInit(0.U(inflightPtrW.W))

  val saTrackerFull      = inflightValid.asUInt.andR
  val anyComputeInFlight = inflightValid.asUInt =/= 0.U

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
  // Per-wslot drain countdown (PUSH-after-MATMUL hazard)
  //
  //   PE(i,j) for output row k reads wslot column j at T_M + 1 + k + i + j;
  //   lane j's last read is at T_M + (2*rows - 2) + j (worst case
  //   k = i = rows - 1).  PUSH lane j writes at T_push + 1 + j and is visible
  //   at T_push + 2 + j.  Constraint: T_push + 1 + j > T_M + (2*rows - 2) + j
  //   → T_push > T_M + (rows + cols - 3).  We encode the auto-memory's
  //   conservative T_push > T_M_prev + 62 rule (initial countdown 62 → reaches
  //   0 at cycle T_M + 63).
  // ==========================================================================

  val wbufCountdown = RegInit(VecInit(Seq.fill(2)(0.U((rowCountW + 2).W))))
  val wbufReady     = VecInit(wbufCountdown.map(_ === 0.U))

  // ==========================================================================
  // Cmd dispatch — combinational accept signals
  //
  //   The dispatcher is the single point that decides routing and gates
  //   commands behind hazards.  Every accepted cmd flows into exactly the
  //   FSM(s) it belongs to.  Push prefers ReadP1 to keep ReadP0 free for
  //   compute; falls back to ReadP0 if ReadP1 is busy.
  // ==========================================================================

  // Shared hazard for any op that touches a same-buffer accumulator
  // (compute psum read, pop read, push acc write).
  val accReuseHazard = !accBufRow0Ready(cmdAccBuf)
  val computeHazard  = saTrackerFull || accReuseHazard
  val pushWHazard    = !wbufReady(cmdWslot)

  // Per-target-FSM accept signals.
  val acceptCompute  = io.cmd.valid && isCompute    && p0Idl && !computeHazard
  val acceptPushP1   = io.cmd.valid && isPush       && p1Idl &&
                       Mux(isPushW, !pushWHazard, !accReuseHazard)
  val acceptPushP0   = io.cmd.valid && isPush       && !p1Idl && p0Idl &&
                       Mux(isPushW, !pushWHazard, !accReuseHazard)
  val acceptBF16Push = io.cmd.valid && isPushBF16   && p0Idl && p1Idl && !accReuseHazard
  val acceptPopFP8   = io.cmd.valid && isPopFP8     && w0Idl && !accReuseHazard
  val acceptPopBF16  = io.cmd.valid && isPopBF16    && w0Idl && w1Idl && !accReuseHazard

  // ==========================================================================
  // Hazard / structural assertions (debug only — do not gate execution).
  // The dispatch logic above already blocks unsafe cmds; these asserts make
  // any blocked cmd a loud failure so the bug surfaces at the source.
  // ==========================================================================

  assert(!(io.cmd.valid && isCompute && saTrackerFull),
    "SA: compute issued while in-flight tracker is full")
  assert(!(io.cmd.valid && isCompute && accReuseHazard),
    "SA: compute issued before previous compute's row-0 writeback on same accSel")
  assert(!(io.cmd.valid && isCompute && !p0Idl),
    "SA: compute issued while ReadP0 is busy")

  assert(!(io.cmd.valid && isPushW && pushWHazard),
    "SA: PushWeight issued before previous matmul on same wslot has drained")
  assert(!(io.cmd.valid && isPushAccFP8 && accReuseHazard),
    "SA: PushAccFP8 issued before previous compute's row-0 writeback")
  assert(!(io.cmd.valid && isPushBF16 && accReuseHazard),
    "SA: PushAccBF16 issued before previous compute's row-0 writeback")

  assert(!(io.cmd.valid && isPushBF16 && !(p0Idl && p1Idl)),
    "SA: PushAccBF16 issued while ReadP0 or ReadP1 is busy")
  assert(!(io.cmd.valid && isPush && !(p0Idl || p1Idl)),
    "SA: Push issued while both Read ports are busy")

  assert(!(io.cmd.valid && isPopFP8 && !w0Idl),
    "SA: PopAccFP8 issued while WriteP0 is busy")
  assert(!(io.cmd.valid && isPopBF16 && !(w0Idl && w1Idl)),
    "SA: PopAccBF16 issued while WriteP0 or WriteP1 is busy")
  assert(!(io.cmd.valid && isPop && accReuseHazard),
    "SA: Pop issued before previous compute's row-0 writeback")

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
    "SA: Pop writes to mreg bank currently being read by an active push/compute")
  assert(!mregBankConflictPush,
    "SA: Push reads from mreg bank currently being written by an active pop")

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
    Mux(op === MxuOp.PushWeight, p.cols.U, tileRows.U)

  // Drive the per-row push outputs for the row whose mreg response just
  // arrived.  PushWeight writes one weight-buffer column; PushAccFP8
  // dequantizes and drives one accBuf load.
  private def processPushRow(cmd: MxuCmd, row: UInt, mregResp: UInt): Unit = {
    when(cmd.op === MxuOp.PushWeight) {
      io.weightWriteReq.valid           := true.B
      io.weightWriteReq.bits.weightSlot := cmd.weightSlot
      io.weightWriteReq.bits.laneIdx    := row(p.mxu.colIdxBits - 1, 0)
      io.weightWriteReq.bits.data       := unpackRow(mregResp, p.inT.ieeeWidth, p.rows)
    }.elsewhen(cmd.op === MxuOp.PushAccFP8) {
      val fp8Row = unpackRow(mregResp, 8, p.cols)
      io.accLoadReq.valid       := true.B
      io.accLoadReq.bits.accSel := cmd.accSel
      io.accLoadReq.bits.rowIdx := row(rowBits - 1, 0)
      io.accLoadReq.bits.data   := deqBank(fp8Row)
    }
  }

  // ==========================================================================
  // ReadP0 FSM — mreg read port 0
  //
  //   States: idle / push (single-port) / bf16 lo half / compute issue.
  //   Idle accepts compute, single-port push (when ReadP1 is busy), or BF16
  //   push (alongside ReadP1).  Each non-idle state streams 32 mreg reads
  //   and processes their responses.
  // ==========================================================================

  switch(p0State) {
    is(p0Idle) {
      when(acceptCompute) {
        p0Cmd := io.cmd.bits
        issueP0Read(io.cmd.bits.mregId, 0.U)
        driveP0AccRead(io.cmd.bits, 0.U)
        p0Row     := 0.U
        p0NextRow := 1.U
        p0State   := p0Comp

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
      val bf16Row = Wire(Vec(p.cols, UInt(16.W)))
      for (i <- 0 until p.cols) {
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
      io.compute.bits.act          := unpackRow(io.mregReadResp0.bits, p.inT.ieeeWidth, p.rows)
      io.compute.bits.psum         := io.accComputeReadData
      io.compute.bits.accumulate   := (p0Cmd.op === MxuOp.MatmulAcc)
      io.compute.bits.weightBufSel := p0Cmd.weightSlot

      when(p0NextRow >= tileRows.U) {
        p0State := p0Idle
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
  //   States: idle / push (single-port) / BF16 hi half.  No compute support.
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
      // Stream hi-half mreg reads in lockstep with ReadP0.  ReadP0 drives
      // the combined accLoadReq using io.mregReadResp1 (this port's
      // response).
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
      io.mregWriteReq0.bits.data   := packMregRow(fp8Row.map(_(7, 0)))

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
      io.mregWriteReq0.bits.data   := packMregRow(io.accStoreData.take(16).map(_(15, 0)))

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
      // Hi half of the BF16 pop; reads io.accStoreData (driven by WriteP0).
      io.mregWriteReq1.valid       := true.B
      io.mregWriteReq1.bits.mregId := w1Cmd.mregId + 1.U
      io.mregWriteReq1.bits.row    := w1Row
      io.mregWriteReq1.bits.data   := packMregRow(io.accStoreData.drop(16).map(_(15, 0)))

      when(w1Row + 1.U >= tileRows.U) {
        w1State := w1Idle
      }.otherwise {
        w1Row := w1Row + 1.U
      }
    }
  }

  // ==========================================================================
  // SA pipeline tracker — FIFO update + writeback dispatch
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

  // Pointer advance — depth 3 isn't a power of two, so wrap explicitly.
  when(popThisCycle) {
    ifHead := Mux(ifHead === (inflightDepth - 1).U, 0.U, ifHead + 1.U)
  }

  // Enqueue on compute accept.
  when(acceptCompute) {
    inflight(ifTail).accSel      := io.cmd.bits.accSel
    inflight(ifTail).rowsWritten := 0.U
    inflightValid(ifTail)        := true.B
    ifTail                       := Mux(ifTail === (inflightDepth - 1).U, 0.U, ifTail + 1.U)
  }

  // ==========================================================================
  // Per-wslot drain countdown
  // ==========================================================================

  for (w <- 0 until 2) {
    when(acceptCompute && (io.cmd.bits.weightSlot === w.U)) {
      wbufCountdown(w) := wbufDrainInit
    }.elsewhen(wbufCountdown(w) > 0.U) {
      wbufCountdown(w) := wbufCountdown(w) - 1.U
    }
  }

  // ==========================================================================
  // Busy / active outputs
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

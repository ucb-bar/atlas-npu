// ============================================================================
// Nvfp4Mxu.scala — NVFP4 matrix unit (configs with AtlasParams.nvfp4).
//
// acc[a] (32 x 32 FP32) = A @ W^T (+ acc[a]), where
//   A  = m[vs1]: 32 rows x 64 E2M1 elements, with m[vs1]'s scale sidecar
//        (UE4M3 per 16 elements, MregScaleFile)
//   W  = weight slot w: row j holds output column j's 64 elements and scales
//        (pushed from an m-register and its scales)
// Each of the 32 lanes is an fp4_fedp (ucb-bar/nvfp4@0c611e1:fpga/pe; NVIDIA
// tensor-core arithmetic): lane j computes d = C + A_r . W_j for one activation
// row r per cycle, so a matmul feeds 32 rows in 32 cycles and a K chain of
// vmatmul.acc equals a chain of NVIDIA m16n8k64 MMAs bit for bit, except for the
// known gap in docs/nvfp4-mxu.md (an all-zero 16-element block can set the
// alignment).
//
// Engines (one command register + step counter each, like the other MXUs):
//   E0  mreg read port 0: matmul feed, weight push (if E1 is busy), acc push
//   E1  mreg read port 1: weight push (preferred), acc push (with E0)
//   W   mreg write ports 0+1: pops
// FP32 transfers move 4 m-registers in 64 steps: step s is phase s/32, row
// s%32; register q of {m[r], m[r+1], m[r+2], m[r+3]} holds columns 8q..8q+7.
//
// cmdReady reports whether the command on io.cmd.bits would be accepted this
// cycle (the scalar core's interlock); io.cmd.valid is only raised then.
// ============================================================================

package atlas.nvfp4

import chisel3._
import chisel3.util._
import atlas.common.{Nvfp4MxuParams, MregParams, MregReadReq, MregWriteReq}

/** fp4_fedp (ucb-bar/nvfp4@0c611e1:fpga/pe/fp4_fedp.sv): 64-element NVFP4 dot product with FP32 accumulate. */
class Fp4Fedp extends BlackBox with HasBlackBoxResource {
  override def desiredName = "fp4_fedp"
  val io = IO(new Bundle {
    val clk     = Input(Clock())
    val rst     = Input(Bool())
    val in_vld  = Input(Bool())
    val a       = Input(UInt(256.W))
    val b       = Input(UInt(256.W))
    val mode    = Input(UInt(2.W))
    val sfa     = Input(UInt(32.W))
    val sfb     = Input(UInt(32.W))
    val c       = Input(UInt(32.W))
    val out_vld = Output(Bool())
    val d       = Output(UInt(32.W))
  })
  Seq("dotprod_pkg.sv", "front_end_nvfp4.sv", "mul_lane_nvfp4.sv", "align_nvfp4.sv",
      "exact_acc_tree.sv", "fp4_fedp_acc.sv", "fp4_fedp.sv")
    .foreach(f => addResource(s"/vsrc/nvfp4/$f"))
}

object NvOp extends ChiselEnum {
  val PushWeight, PushAccBF16, PushAccFP32, PopBF16, PopFP32, Matmul, MatmulAcc = Value
}

class NvCmd(mregIdBits: Int) extends Bundle {
  val op         = NvOp()
  val mregId     = UInt(mregIdBits.W)  // source (push, matmul) or destination (pop) register
  val accSel     = Bool()
  val weightSlot = Bool()
}

object Nvfp4 {
  /** FP32 -> BF16, round to nearest even; NaN -> quiet NaN with the input's sign. */
  def f32ToBf16(x: UInt): UInt = {
    val isNaN = x(30, 23).andR && x(22, 0).orR
    val rnd   = (x + (0x7FFF.U(32.W) + x(16)))(31, 16)
    Mux(isNaN, Cat(x(31), "h7fc0".U(15.W)), rnd)
  }
  def bf16ToF32(x: UInt): UInt = Cat(x(15, 0), 0.U(16.W))
}

class Nvfp4Mxu(p: Nvfp4MxuParams, mregP: MregParams) extends Module {
  import NvOp._
  require(p.lanes == 32 && p.rows == mregP.mregRows && mregP.mregRowBits == 256,
    "the NVFP4 MXU is 32 lanes over 32-row, 256-bit m-registers")

  private val L  = p.lanes
  private val R  = p.rows
  private val SW = 7  // step counter: 0..63

  val io = IO(new Bundle {
    val cmd      = Flipped(Valid(new NvCmd(mregP.mregIdBits)))
    val cmdReady = Output(Bool())

    val mregReadReq0  = Valid(new MregReadReq(mregP))
    val mregReadResp0 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregReadReq1  = Valid(new MregReadReq(mregP))
    val mregReadResp1 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWriteReq0 = Valid(new MregWriteReq(mregP))
    val mregWriteReq1 = Valid(new MregWriteReq(mregP))

    // Scale sidecar reads, issued with the mreg read of the same register and row.
    val scaleReadReq  = Vec(2, Valid(new MregReadReq(mregP)))
    val scaleReadResp = Input(Vec(2, UInt(32.W)))

    val busy = Output(Bool())
    val feeding = Output(Bool())   // a matmul feeds one row of A into the lanes (perf counter)
    // Registers still to be read / written after this cycle (MregBankTracker).
    val activeReads  = Output(Vec(4, Valid(UInt(mregP.mregIdBits.W))))
    val activeWrites = Output(Vec(4, Valid(UInt(mregP.mregIdBits.W))))
  })

  // ==========================================================================
  // Engine state
  // ==========================================================================

  private def steps(op: NvOp.Type): UInt =
    Mux(op === PushAccFP32 || op === PopFP32, (2 * R).U(SW.W), R.U(SW.W))
  private def row(s: UInt): UInt   = s(log2Ceil(R) - 1, 0)
  private def phase(s: UInt): UInt = s(log2Ceil(R))

  val e0v = RegInit(false.B); val e0 = Reg(new NvCmd(mregP.mregIdBits)); val e0n = Reg(UInt(SW.W))
  val e1v = RegInit(false.B); val e1 = Reg(new NvCmd(mregP.mregIdBits)); val e1n = Reg(UInt(SW.W))
  val wv  = RegInit(false.B); val w  = Reg(new NvCmd(mregP.mregIdBits)); val wn  = Reg(UInt(SW.W))

  val e0Boundary = !e0v || (e0n +& 1.U) >= steps(e0.op)
  val e1Boundary = !e1v || (e1n +& 1.U) >= steps(e1.op)
  val wBoundary  = !wv  || (wn  +& 1.U) >= steps(w.op)

  val e0Matmul  = e0v && (e0.op === Matmul || e0.op === MatmulAcc)
  val e0PushW   = e0v && e0.op === PushWeight
  val e1PushW   = e1v && e1.op === PushWeight
  val e0PushAcc = e0v && (e0.op === PushAccBF16 || e0.op === PushAccFP32)

  // ==========================================================================
  // Weight slots (registers: every lane reads its row every cycle)
  // ==========================================================================

  val wData  = Reg(Vec(2, Vec(L, UInt(256.W))))
  val wScale = Reg(Vec(2, Vec(L, UInt(32.W))))

  // ==========================================================================
  // Accumulators: per buffer one read port (psum or pop) and one masked write
  // port (matmul writeback or acc push); the hazard rules below keep the two
  // users of each port apart.
  // ==========================================================================

  val acc = Seq.fill(2)(SyncReadMem(R, Vec(L, UInt(32.W))))
  val accRdEn   = Wire(Vec(2, Bool())); val accRdRow = Wire(Vec(2, UInt(log2Ceil(R).W)))
  val accWrEn   = Wire(Vec(2, Bool())); val accWrRow = Wire(Vec(2, UInt(log2Ceil(R).W)))
  val accWrData = Wire(Vec(2, Vec(L, UInt(32.W)))); val accWrMask = Wire(Vec(2, Vec(L, Bool())))
  accRdEn := VecInit(false.B, false.B); accRdRow := VecInit(0.U, 0.U)
  accWrEn := VecInit(false.B, false.B); accWrRow := VecInit(0.U, 0.U)
  accWrData := 0.U.asTypeOf(accWrData); accWrMask := 0.U.asTypeOf(accWrMask)
  val accRdData = VecInit((0 until 2).map(b => acc(b).read(accRdRow(b), accRdEn(b))))
  for (b <- 0 until 2) {
    when(accWrEn(b)) { acc(b).write(accWrRow(b), accWrData(b), accWrMask(b)) }
  }
  private def accRead(sel: Bool, r: UInt): Unit = {
    for (b <- 0 until 2) when(sel === (b == 1).B) { accRdEn(b) := true.B; accRdRow(b) := r }
  }
  private def accWrite(sel: Bool, r: UInt, data: Seq[UInt], mask: Seq[Bool]): Unit = {
    for (b <- 0 until 2) when(sel === (b == 1).B) {
      accWrEn(b) := true.B; accWrRow(b) := r
      accWrData(b) := VecInit(data); accWrMask(b) := VecInit(mask)
    }
  }

  // ==========================================================================
  // Lanes. Inputs are registered, so the mreg read path ends at a register.
  // ==========================================================================

  val laneVld = RegInit(false.B)
  val laneA   = Reg(UInt(256.W))
  val laneSfa = Reg(UInt(32.W))
  val laneC   = Reg(Vec(L, UInt(32.W)))
  val laneWs  = Reg(Bool())
  val laneTag = Reg(new Bundle { val acc = Bool(); val row = UInt(log2Ceil(R).W) })

  val lanes = Seq.fill(L)(Module(new Fp4Fedp))
  val laneD = Wire(Vec(L, UInt(32.W)))
  for (j <- 0 until L) {
    val u = lanes(j).io
    u.clk := clock; u.rst := reset.asBool; u.in_vld := laneVld; u.mode := 0.U  // NVFP4
    u.a := laneA; u.sfa := laneSfa; u.c := laneC(j)
    u.b := wData(laneWs)(j); u.sfb := wScale(laneWs)(j)
    laneD(j) := u.d
  }

  // Writeback tags travel beside the lanes' fixed latency.
  val tagV   = ShiftRegister(laneVld, p.laneLatency, false.B, true.B)
  val tagAcc = ShiftRegister(laneTag.acc, p.laneLatency)
  val tagRow = ShiftRegister(laneTag.row, p.laneLatency)
  // Valid bits of every stage, for the "matmul writes still pending" hazard.
  val tagPipeV   = RegInit(VecInit(Seq.fill(p.laneLatency)(false.B)))
  val tagPipeAcc = Reg(Vec(p.laneLatency, Bool()))
  tagPipeV(0) := laneVld; tagPipeAcc(0) := laneTag.acc
  for (i <- 1 until p.laneLatency) { tagPipeV(i) := tagPipeV(i - 1); tagPipeAcc(i) := tagPipeAcc(i - 1) }
  assert(!tagV || lanes(0).io.out_vld, "NV: lane output valid does not line up with its writeback tag")

  when(tagV) { accWrite(tagAcc, tagRow, laneD, Seq.fill(L)(true.B)) }

  // ==========================================================================
  // Hazards and accept conditions
  // ==========================================================================

  val cmd   = io.cmd.bits
  val isMat = cmd.op === Matmul || cmd.op === MatmulAcc
  val isPushW   = cmd.op === PushWeight
  val isPushAcc = cmd.op === PushAccBF16 || cmd.op === PushAccFP32
  val isPop     = cmd.op === PopBF16 || cmd.op === PopFP32

  /** A matmul on accumulator b is feeding or has writebacks in flight. */
  private def computePending(b: Bool): Bool =
    (e0Matmul && e0.accSel === b) || laneVld && laneTag.acc === b ||
      (0 until p.laneLatency).map(i => tagPipeV(i) && tagPipeAcc(i) === b).reduce(_ || _)
  /** A weight push to slot s will still write after this cycle. */
  private def pushWActive(s: Bool): Bool =
    (e0PushW && e0.weightSlot === s && !e0Boundary) || (e1PushW && e1.weightSlot === s && !e1Boundary)
  val pushWWriting = (e0PushW && !e0Boundary) || (e1PushW && !e1Boundary)
  /** The matmul in E0 will still read weight slot s after this cycle. */
  private def matmulReadingSlot(s: Bool): Bool = e0Matmul && e0.weightSlot === s && !e0Boundary
  /** An accumulator push to b will still write after this cycle. */
  private def pushAccWriting(b: Bool): Bool = e0PushAcc && e0.accSel === b && !e0Boundary
  /** A pop of b will still read it after this cycle. */
  private def popReading(b: Bool): Bool = wv && w.accSel === b && !wBoundary

  // Matmul: E0 free; its slot fully pushed; no acc push still writing the buffer (write port);
  // no pop still reading it (read port, and a pop must read the old rows).
  val canMat     = isMat && e0Boundary && !pushWActive(cmd.weightSlot) &&
                   !pushAccWriting(cmd.accSel) && !popReading(cmd.accSel)
  // Weight push: one write path; the slot must not be read by the matmul feeding now.
  val pushWOk    = isPushW && !pushWWriting && !matmulReadingSlot(cmd.weightSlot)
  val canPushWE1 = pushWOk && e1Boundary
  val canPushWE0 = pushWOk && !e1Boundary && e0Boundary
  // Acc push: both read ports; no matmul writebacks and no pop on the buffer.
  val canPushAcc = isPushAcc && e0Boundary && e1Boundary &&
                   !computePending(cmd.accSel) && !popReading(cmd.accSel)
  // Pop: all matmul writes to the buffer landed, no acc push still writing it.
  val canPop     = isPop && wBoundary && !computePending(cmd.accSel) && !pushAccWriting(cmd.accSel)

  io.cmdReady := canMat || canPushWE1 || canPushWE0 || canPushAcc || canPop
  assert(!(io.cmd.valid && !io.cmdReady), "NV: command issued while the unit cannot accept it")

  // ==========================================================================
  // Read port helpers
  // ==========================================================================

  io.mregReadReq0.valid := false.B; io.mregReadReq0.bits := 0.U.asTypeOf(io.mregReadReq0.bits)
  io.mregReadReq1.valid := false.B; io.mregReadReq1.bits := 0.U.asTypeOf(io.mregReadReq1.bits)
  io.scaleReadReq.foreach { r => r.valid := false.B; r.bits := 0.U.asTypeOf(r.bits) }
  io.mregWriteReq0.valid := false.B; io.mregWriteReq0.bits := 0.U.asTypeOf(io.mregWriteReq0.bits)
  io.mregWriteReq1.valid := false.B; io.mregWriteReq1.bits := 0.U.asTypeOf(io.mregWriteReq1.bits)

  /** Register E0/E1 reads for step s of c: PushAccFP32 moves to m[id+2] in phase 1. */
  private def e0Reg(c: NvCmd, s: UInt): UInt = c.mregId + Mux(c.op === PushAccFP32, phase(s) << 1, 0.U)
  private def e1Reg(c: NvCmd, s: UInt): UInt =
    c.mregId + Mux(c.op === PushAccBF16, 1.U, Mux(c.op === PushAccFP32, (phase(s) << 1) + 1.U, 0.U))

  private def issueE0(c: NvCmd, s: UInt): Unit = {
    io.mregReadReq0.valid := true.B
    io.mregReadReq0.bits.mregId := e0Reg(c, s)
    io.mregReadReq0.bits.row := row(s)
    when(c.op === Matmul || c.op === MatmulAcc || c.op === PushWeight) {
      io.scaleReadReq(0).valid := true.B
      io.scaleReadReq(0).bits.mregId := c.mregId
      io.scaleReadReq(0).bits.row := row(s)
    }
    when(c.op === MatmulAcc) { accRead(c.accSel, row(s)) }
  }
  private def issueE1(c: NvCmd, s: UInt): Unit = {
    io.mregReadReq1.valid := true.B
    io.mregReadReq1.bits.mregId := e1Reg(c, s)
    io.mregReadReq1.bits.row := row(s)
    when(c.op === PushWeight) {
      io.scaleReadReq(1).valid := true.B
      io.scaleReadReq(1).bits.mregId := c.mregId
      io.scaleReadReq(1).bits.row := row(s)
    }
  }

  // ==========================================================================
  // E0: process the step whose read returns now, then issue the next step's read
  // ==========================================================================

  laneVld := false.B
  private def words(x: UInt, n: Int, w: Int): Seq[UInt] = (0 until n).map(i => x(w * i + w - 1, w * i))

  when(e0v) {
    val r = row(e0n)
    when(e0.op === Matmul || e0.op === MatmulAcc) {
      laneVld := true.B
      laneA   := io.mregReadResp0.bits
      laneSfa := io.scaleReadResp(0)
      laneC   := Mux(e0.op === MatmulAcc, accRdData(e0.accSel), VecInit(Seq.fill(L)(0.U(32.W))))
      laneWs  := e0.weightSlot
      laneTag.acc := e0.accSel
      laneTag.row := r
    }.elsewhen(e0.op === PushWeight) {
      wData(e0.weightSlot)(r)  := io.mregReadResp0.bits
      wScale(e0.weightSlot)(r) := io.scaleReadResp(0)
    }.elsewhen(e0.op === PushAccBF16) {
      val lo = words(io.mregReadResp0.bits, 16, 16).map(Nvfp4.bf16ToF32)
      val hi = words(io.mregReadResp1.bits, 16, 16).map(Nvfp4.bf16ToF32)
      accWrite(e0.accSel, r, lo ++ hi, Seq.fill(L)(true.B))
    }.elsewhen(e0.op === PushAccFP32) {
      val part = words(io.mregReadResp0.bits, 8, 32) ++ words(io.mregReadResp1.bits, 8, 32)
      val ph   = phase(e0n)
      accWrite(e0.accSel, r, (0 until L).map(j => part(j % 16)),
               (0 until L).map(j => (j / 16 == 1).B === ph.asBool))
    }
    when(!e0Boundary) {
      issueE0(e0, e0n + 1.U)
      e0n := e0n + 1.U
    }
  }

  val accept = io.cmd.valid
  when(e0Boundary) {
    when(accept && (canMat || canPushWE0 || canPushAcc)) {
      e0 := cmd; e0n := 0.U; e0v := true.B
      issueE0(cmd, 0.U)
    }.otherwise { e0v := false.B }
  }

  // ==========================================================================
  // E1: weight push, or the second half of an acc push (lockstep with E0)
  // ==========================================================================

  when(e1v) {
    when(e1.op === PushWeight) {
      wData(e1.weightSlot)(row(e1n))  := io.mregReadResp1.bits
      wScale(e1.weightSlot)(row(e1n)) := io.scaleReadResp(1)
    }
    when(!e1Boundary) {
      issueE1(e1, e1n + 1.U)
      e1n := e1n + 1.U
    }
  }
  when(e1Boundary) {
    when(accept && (canPushWE1 || canPushAcc)) {
      e1 := cmd; e1n := 0.U; e1v := true.B
      issueE1(cmd, 0.U)
    }.otherwise { e1v := false.B }
  }

  // ==========================================================================
  // W: pops (accumulator row read one cycle ahead of the mreg writes)
  // ==========================================================================

  when(wv) {
    val r    = row(wn)
    val data = accRdData(w.accSel)
    when(w.op === PopBF16) {
      val bf = data.map(Nvfp4.f32ToBf16)
      io.mregWriteReq0.valid := true.B
      io.mregWriteReq0.bits.mregId := w.mregId
      io.mregWriteReq0.bits.row := r
      io.mregWriteReq0.bits.data := Cat(bf.take(16).reverse)
      io.mregWriteReq1.valid := true.B
      io.mregWriteReq1.bits.mregId := w.mregId + 1.U
      io.mregWriteReq1.bits.row := r
      io.mregWriteReq1.bits.data := Cat(bf.drop(16).reverse)
    }.otherwise {  // PopFP32: phase 0 -> m[d], m[d+1] (columns 0-15); phase 1 -> m[d+2], m[d+3]
      val ph   = phase(wn)
      val half = Mux(ph.asBool, VecInit(data.drop(16)), VecInit(data.take(16)))
      io.mregWriteReq0.valid := true.B
      io.mregWriteReq0.bits.mregId := w.mregId + (ph << 1)
      io.mregWriteReq0.bits.row := r
      io.mregWriteReq0.bits.data := Cat(half.take(8).reverse)
      io.mregWriteReq1.valid := true.B
      io.mregWriteReq1.bits.mregId := w.mregId + (ph << 1) + 1.U
      io.mregWriteReq1.bits.row := r
      io.mregWriteReq1.bits.data := Cat(half.drop(8).reverse)
    }
    when(!wBoundary) {
      accRead(w.accSel, row(wn + 1.U))
      wn := wn + 1.U
    }
  }
  when(wBoundary) {
    when(accept && canPop) {
      w := cmd; wn := 0.U; wv := true.B
      accRead(cmd.accSel, 0.U)
    }.otherwise { wv := false.B }
  }

  // ==========================================================================
  // Port-use assertions (the hazard rules above should make these unreachable)
  // ==========================================================================

  for (b <- 0 until 2) {
    val psum = e0v && e0.op === MatmulAcc && e0.accSel === (b == 1).B
    val pop  = wv && w.accSel === (b == 1).B
    assert(!(psum && pop && !e0Boundary && !wBoundary), s"NV: acc$b psum read and pop read overlap")
  }
  assert(!(tagV && e0PushAcc && tagAcc === e0.accSel), "NV: matmul writeback and acc push write the same buffer")

  // ==========================================================================
  // Status
  // ==========================================================================

  io.busy := e0v || e1v || wv || laneVld || tagPipeV.asUInt.orR
  io.feeding := e0Matmul

  val e0Next = e0n + 1.U
  val e1Next = e1n + 1.U
  val wNext  = wn + 1.U
  io.activeReads(0).valid := e0v && !e0Boundary
  io.activeReads(0).bits  := e0Reg(e0, e0Next)
  io.activeReads(1).valid := e1v && !e1Boundary
  io.activeReads(1).bits  := e1Reg(e1, e1Next)
  io.activeReads(2).valid := e0v && !e0Boundary && e0.op === PushAccFP32 && !phase(e0Next).asBool
  io.activeReads(2).bits  := e0.mregId + 2.U
  io.activeReads(3).valid := e1v && !e1Boundary && e1.op === PushAccFP32 && !phase(e1Next).asBool
  io.activeReads(3).bits  := e1.mregId + 3.U

  val wPh = Mux(w.op === PopFP32, phase(wNext) << 1, 0.U)
  io.activeWrites(0).valid := wv && !wBoundary
  io.activeWrites(0).bits  := w.mregId + wPh
  io.activeWrites(1).valid := wv && !wBoundary
  io.activeWrites(1).bits  := w.mregId + wPh + 1.U
  io.activeWrites(2).valid := wv && !wBoundary && w.op === PopFP32 && !phase(wNext).asBool
  io.activeWrites(2).bits  := w.mregId + 2.U
  io.activeWrites(3).valid := wv && !wBoundary && w.op === PopFP32 && !phase(wNext).asBool
  io.activeWrites(3).bits  := w.mregId + 3.U
}

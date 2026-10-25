/*
CSRFile.scala — TileLink-accessible CSR register file.

Host reads/writes via TileLink.  Scalar core reads/writes via internal port.
Scale registers are NOT here — see ScaleRegFile.

TileLink byte-address map within the first 0x20 bytes of the CSR window:
  0x00  cycle counter      (RW)
  0x04  instruction counter(RW)
  0x08  status             (RO — bit 0: halted, bits [2:1]: halt_reason)
  0x0C  illegal-instr PC   (RO — hardware-driven)
  0x10  dbg0               (RW)
  0x14  dbg1               (RW)
  0x18  execControl        (RW — bit 0: 1=start, 0=stop; auto-clears on scalar halt)
  0x1C  ISA                (RO — [31:24] major, [23:16] minor, [3:0] AtlasCore features)

Performance counters (the scheme of RISC-V mhpmcounter / mcountinhibit, on MMIO):
  0x100 PERF_RESET   (WO — write 1 to bit i to clear counter i; reads 0)
  0x104 PERF_INHIBIT (RW — bit i = 1 stops counter i; reset 0)
  0x108 PERF_NUM     (RO — number of counters)
  0x200 + 8i         counter i, bits 31:0  (RO)
  0x204 + 8i         counter i, bits 63:32 (RO)
Every counter counts its event in the cycles from a start until the core has halted and every
unit is idle (a halt does not wait for units), so it stops when a kernel finishes and the host
reads it afterwards; PerfCounters.names lists the events.

The scalar core reads the same words as CSRs (ScalarISA): cycle 0xC00, instructions 0xC01,
status 0xC02, illegal PC 0xC03, dbg0 0xC10, dbg1 0xC11, ISA 0xCC0, PERF_NUM 0xCC1, and counter i
at 0xCC2 + 2i (bits 31:0) and 0xCC3 + 2i (bits 63:32). It can write only cycle, instructions,
dbg0 and dbg1. An access to any other CSR, or a write to a read-only one, is illegal.

halt_reason encoding:
  0 = none, 1 = illegal instruction, 2 = ecall, 3 = ebreak
*/

package atlas.scalar

import chisel3._
import chisel3.util._

import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

import ScalarISA._

/** The performance counters: index i counts event names(i) (AtlasCore drives the events). */
object PerfCounters {
  val names: Seq[String] = Seq(
    "cycles",       // the core runs, or a unit is still busy after the halt
    "instret",      // an instruction issues
    "stall_unit",   // S1 holds an instruction: its unit cannot take it (structural hazard)
    "stall_data",   // S1 holds an instruction: MREG / VMEM / scalar-load / scale data hazard
    "stall_dma",    // dma.wait on a busy channel
    "stall_delay",  // a delay instruction counts down
    "mxu0_busy",    // MXU slot 0 (systolic array or NVFP4 MXU): any engine active
    "mxu0_feed",    // MXU slot 0: a matmul feeds one row of A (full MAC-array use)
    "mxu1_busy",    // MXU1: any engine active
    "mxu1_feed",    // MXU1: a matmul feeds one row of A
    "vpu_busy",     // the VPU has an instruction in either slot
    "xlu_busy",     // the XLU transposes
    "vload_busy",   // the LSU's vload path is busy
    "vstore_busy",  // the LSU's vstore path is busy
    "dma_busy",     // some DMA channel is busy
    "dma_beats",    // a 32-byte DMA beat completes on the memory port
    // The causes of the stall_unit and stall_data cycles; a cycle counts in every cause that holds.
    "stall_unit_mxu0",      // MXU slot 0 cannot take the command
    "stall_unit_mxu1",      // MXU1 cannot take the command
    "stall_unit_vpu",       // the VPU cannot take the command
    "stall_unit_xlu",       // the XLU is busy
    "stall_unit_lsu",       // the vload or vstore path is busy, or a scalar load is pending
    "stall_unit_dma",       // the DMA channel or the next command slot is busy
    "stall_data_mreg",      // MREG reservation or physical port
    "stall_data_vmem",      // an LSU path holds the VMEM bank
    "stall_data_dma_vmem",  // a DMA transfer would overtake an LSU transfer
    "stall_data_xreg",      // a scalar load result or the register write port is pending
    "stall_data_scale")     // a seld result or the scale register write port is pending
  val num: Int = names.size
  val IsaOffset     = 0x1C
  val ResetOffset   = 0x100
  val InhibitOffset = 0x104
  val NumOffset     = 0x108
  val CounterBase   = 0x200
}

/** TileLink-visible CSR register file shared with the scalar core.
  *
  * @param tlBP  TileLink bundle parameters for the host-facing CSR port.
  */
class CSRFile(tlBP: TLBundleParameters) extends Module {

  val io = IO(new Bundle {
    val tl  = Flipped(new TLBundle(tlBP))
    val csr = new CSRInternalPort
    // Direct outputs for debug / convenience
    val dbg0         = Output(UInt(32.W))
    val dbg1         = Output(UInt(32.W))
    val execRun      = Output(Bool())
    val execRunWrite = Output(Bool())
    val perfEvents   = Input(Vec(PerfCounters.num, Bool()))
    val features     = Input(UInt(4.W))   // AtlasCore io.features, in the ISA word
  })

  // ── Registers ────────────────────────────────────────────────────
  val reg_cycle       = RegInit(0.U(32.W))
  val reg_inst        = RegInit(0.U(32.W))
  val reg_illegal     = RegInit(0.U(32.W))
  val reg_halt_reason = RegInit(0.U(2.W))
  val reg_dbg0        = RegInit(0.U(32.W))
  val reg_dbg1        = RegInit(0.U(32.W))

  val reg_execRun      = RegInit(false.B)
  val reg_execRunWrite = RegInit(false.B)
  reg_execRunWrite := false.B
  io.execRun      := reg_execRun
  io.execRunWrite := reg_execRunWrite

  // Hardware auto-updates
  reg_cycle := reg_cycle + 1.U
  when(io.csr.inst_retire) { reg_inst := reg_inst + 1.U }
  when(io.csr.set_illegal) {
    reg_illegal     := io.csr.illegal_pc
    reg_halt_reason := 1.U
  }
  when(io.csr.set_ecall) {
    reg_halt_reason := 2.U
  }
  when(io.csr.set_ebreak) {
    reg_halt_reason := 3.U
  }

  // ── Performance counters ─────────────────────────────────────────
  private val nPerf = PerfCounters.num
  val perfCnt     = RegInit(VecInit(Seq.fill(nPerf)(0.U(64.W))))
  val perfInhibit = RegInit(0.U(nPerf.W))
  val perfClear   = WireDefault(0.U(nPerf.W))
  for (i <- 0 until nPerf) {
    when(perfClear(i)) {
      perfCnt(i) := 0.U
    }.elsewhen(io.perfEvents(i) && !perfInhibit(i)) {
      perfCnt(i) := perfCnt(i) + 1.U
    }
  }
  private val isaIdx         = (PerfCounters.IsaOffset / 4).U
  private val perfResetIdx   = (PerfCounters.ResetOffset / 4).U
  private val perfInhibitIdx = (PerfCounters.InhibitOffset / 4).U
  private val perfNumIdx     = (PerfCounters.NumOffset / 4).U
  private val perfWords = (0 until nPerf).flatMap { i =>
    val w = PerfCounters.CounterBase / 4 + 2 * i
    Seq(w.U -> perfCnt(i)(31, 0), (w + 1).U -> perfCnt(i)(63, 32))
  }

  // ── Word-index read helper ───────────────────────────────────────
  private def readWord(idx: UInt): UInt = MuxLookup(idx, 0.U)(Seq(
    0.U -> reg_cycle,
    1.U -> reg_inst,
    2.U -> Cat(0.U(29.W), reg_halt_reason, io.csr.halted),
    3.U -> reg_illegal,
    4.U -> reg_dbg0,
    5.U -> reg_dbg1,
    6.U -> Cat(0.U(31.W), reg_execRun),
    isaIdx         -> Cat(ISA_MAJOR.U(8.W), ISA_MINOR.U(8.W), 0.U(12.W), io.features),
    perfInhibitIdx -> perfInhibit,
    perfNumIdx     -> nPerf.U
  ) ++ perfWords)

  // ── Internal (scalar-core) access ────────────────────────────────
  // 12-bit CSR address -> (word index, writable). Any other address is illegal, and so is a
  // write to a read-only CSR (RISC-V privileged spec).
  private val csrMap = Seq(
    CSR_CYCLE_COUNTER -> (0.U, true),
    CSR_INST_COUNTER  -> (1.U, true),
    CSR_STATUS        -> (2.U, false),
    CSR_ILLEGAL_INSTR -> (3.U, false),
    CSR_DBG0          -> (4.U, true),
    CSR_DBG1          -> (5.U, true),
    CSR_ISA           -> (isaIdx, false),
    CSR_PERF_NUM      -> (perfNumIdx, false))
  private def csrIn(m: Seq[(UInt, (UInt, Boolean))]): Bool = m.map(_._1 === io.csr.addr).reduce(_ || _)
  private val perfCsr   = io.csr.addr - CSR_PERF_BASE
  private val isPerfCsr = io.csr.addr >= CSR_PERF_BASE && perfCsr < (2 * nPerf).U

  val intIdx = Mux(isPerfCsr, (PerfCounters.CounterBase / 4).U + perfCsr,
    MuxLookup(io.csr.addr, 0.U)(csrMap.map { case (a, (w, _)) => a -> w }))
  io.csr.illegal := !(isPerfCsr || csrIn(csrMap)) || (io.csr.write && !csrIn(csrMap.filter(_._2._2)))
  val intRdata = readWord(intIdx)
  io.csr.rdata := intRdata

  // Read-modify-write for CSR ops
  val newVal = WireDefault(intRdata)
  switch(io.csr.op) {
    is(CSR_RW)  { newVal := io.csr.wdata }
    is(CSR_RS)  { newVal := intRdata | io.csr.wdata }
    is(CSR_RC)  { newVal := intRdata & (~io.csr.wdata).asUInt }
    is(CSR_RWI) { newVal := io.csr.wdata }
    is(CSR_RSI) { newVal := intRdata | io.csr.wdata }
    is(CSR_RCI) { newVal := intRdata & (~io.csr.wdata).asUInt }
  }

  // Internal writes — lower priority (TL overwrites via last-connect). A write to a read-only
  // CSR is illegal and never valid.
  when(io.csr.valid && io.csr.write) {
    when(intIdx === 0.U) { reg_cycle := newVal }
    when(intIdx === 1.U) { reg_inst  := newVal }
    when(intIdx === 4.U) { reg_dbg0  := newVal }
    when(intIdx === 5.U) { reg_dbg1  := newVal }
  }

  // ── TileLink manager ───────────────────────────────────────────────
  // Single-beat, 1-cycle latency (registered response).
  val tl = io.tl

  val respValid  = RegInit(false.B)
  val respData   = Reg(UInt(tl.params.dataBits.W))
  val respSource = Reg(UInt(tl.params.sourceBits.W))
  val respSize   = Reg(UInt(tl.params.sizeBits.W))
  val respOpcode = Reg(UInt(3.W))

  tl.a.ready := !respValid || tl.d.ready

  when(tl.a.fire) {
    respValid  := true.B
    respSource := tl.a.bits.source
    respSize   := tl.a.bits.size
    val wordIdx = tl.a.bits.address(AtlasMemMap.CSR_ADDR_BITS - 1, 2)

    when(tl.a.bits.opcode === TLMessages.Get) {
      respOpcode := TLMessages.AccessAckData
      respData   := readWord(wordIdx)
    }.otherwise { // PutFullData / PutPartialData
      respOpcode := TLMessages.AccessAck
      respData   := 0.U
      // TL writes — higher priority than internal (last-connect wins)
      when(wordIdx === 0.U) { reg_cycle := tl.a.bits.data }
      when(wordIdx === 1.U) { reg_inst  := tl.a.bits.data }
      // 2 (status) and 3 (illegal) are read-only from TL as well
      when(wordIdx === 4.U) { reg_dbg0  := tl.a.bits.data }
      when(wordIdx === 5.U) { reg_dbg1  := tl.a.bits.data }
      when(wordIdx === perfResetIdx)   { perfClear   := tl.a.bits.data(nPerf - 1, 0) }
      when(wordIdx === perfInhibitIdx) { perfInhibit := tl.a.bits.data(nPerf - 1, 0) }
      when(wordIdx === 6.U) {
        reg_execRun      := tl.a.bits.data(0)
        reg_execRunWrite := true.B
        when(tl.a.bits.data(0)) {
          reg_halt_reason := 0.U
          reg_illegal     := 0.U
        }
      }
    }
  }

  // Only clear execRun on a committed scalar halt event. Using the broader
  // combinational "halted" signal can drop execRun early while the scalar
  // program is still draining to completion.
  val csrTlPuts = tl.a.bits.opcode === TLMessages.PutFullData ||
    tl.a.bits.opcode === TLMessages.PutPartialData
  val csrTlWritesExec =
    tl.a.fire && csrTlPuts &&
      tl.a.bits.address(AtlasMemMap.CSR_ADDR_BITS - 1, 2) === 6.U
  val scalarHaltEvent = io.csr.set_illegal || io.csr.set_ecall || io.csr.set_ebreak
  when(scalarHaltEvent && !csrTlWritesExec) {
    reg_execRun := false.B
  }

  when(tl.d.fire && !tl.a.fire) { respValid := false.B }

  tl.d.valid       := respValid
  tl.d.bits        := DontCare
  tl.d.bits.opcode := respOpcode
  tl.d.bits.param  := 0.U
  tl.d.bits.size   := respSize
  tl.d.bits.source := respSource
  tl.d.bits.sink   := 0.U
  tl.d.bits.denied := false.B
  tl.d.bits.corrupt := false.B
  tl.d.bits.data   := respData

  // ── Debug outputs ────────────────────────────────────────────────
  io.dbg0 := reg_dbg0
  io.dbg1 := reg_dbg1
}

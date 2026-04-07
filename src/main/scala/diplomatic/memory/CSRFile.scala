/*
CSRFile.scala — TileLink-accessible CSR register file.

Host reads/writes via TileLink.  Scalar core reads/writes via internal port.
Scale registers are NOT here — see ScaleRegFile.

TileLink byte-address map (offsets from CSR_BASE):
  0x00  cycle counter      (RW)
  0x04  instruction counter(RW)
  0x08  status             (RO — bit 0: halted, bits [2:1]: halt_reason)
  0x0C  illegal-instr PC   (RO — hardware-driven)
  0x10  dbg0               (RW)
  0x14  dbg1               (RW)
  0x18  softReset          (W0 - write 1 to pulse-reset the core)

halt_reason encoding:
  0 = none, 1 = illegal instruction, 2 = ecall, 3 = ebreak
*/

package atlas.scalar

import chisel3._
import chisel3.util._

import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

import ScalarISA._

/** TileLink-visible CSR register file shared with the scalar core.
  *
  * @param tlBP  TileLink bundle parameters for the host-facing CSR port.
  */
class CSRFile(tlBP: TLBundleParameters) extends Module {

  val io = IO(new Bundle {
    val tl  = Flipped(new TLBundle(tlBP))
    val csr = new CSRInternalPort
    // Direct outputs for debug / convenience
    val dbg0      = Output(UInt(32.W))
    val dbg1      = Output(UInt(32.W))
    val softReset = Output(Bool())
  })

  // ── Registers ────────────────────────────────────────────────────
  val reg_cycle       = RegInit(0.U(32.W))
  val reg_inst        = RegInit(0.U(32.W))
  val reg_illegal     = RegInit(0.U(32.W))
  val reg_halt_reason = RegInit(0.U(2.W))
  val reg_dbg0        = RegInit(0.U(32.W))
  val reg_dbg1        = RegInit(0.U(32.W))

  val reg_softReset = RegInit(false.B)
  reg_softReset := false.B
  io.softReset  := reg_softReset

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

  // ── Word-index read helper ───────────────────────────────────────
  private def readWord(idx: UInt): UInt = MuxLookup(idx, 0.U)(Seq(
    0.U -> reg_cycle,
    1.U -> reg_inst,
    2.U -> Cat(0.U(29.W), reg_halt_reason, io.csr.halted),
    3.U -> reg_illegal,
    4.U -> reg_dbg0,
    5.U -> reg_dbg1,
    6.U -> 0.U
  ))

  // ── Internal (scalar-core) access ────────────────────────────────
  // Map 12-bit CSR addresses to word indices
  private def csrAddrToWordIdx(addr: UInt): UInt = MuxLookup(addr, 7.U)(Seq(
    CSR_CYCLE_COUNTER -> 0.U,
    CSR_INST_COUNTER  -> 1.U,
    CSR_STATUS        -> 2.U,
    CSR_ILLEGAL_INSTR -> 3.U,
    CSR_DBG0          -> 4.U,
    CSR_DBG1          -> 5.U
  ))

  val intIdx   = csrAddrToWordIdx(io.csr.addr)
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

  // Internal writes — lower priority (TL overwrites via last-connect)
  when(io.csr.valid && io.csr.op =/= CSR_NONE) {
    when(intIdx === 0.U) { reg_cycle := newVal }
    when(intIdx === 1.U) { reg_inst  := newVal }
    // indices 2,3 are read-only (status, illegal)
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
    val wordIdx = tl.a.bits.address(4, 2) // bits [4:2] select word

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
      when(wordIdx === 6.U) { reg_softReset := tl.a.bits.data(0) }
    }
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

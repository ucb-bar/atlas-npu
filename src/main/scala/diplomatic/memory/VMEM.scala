// ============================================================================
// Vmem.scala — 8-bank block-banked vector scratchpad with masked writes.
//
// Storage: numBanks × SyncReadMem(linesPerBank, Vec(lineBytes, UInt(8.W)))
//          Each bank: 1 read port, 1 write port (1R1W SRAM).
//          Byte-granularity masking via SyncReadMem native mask.
//
// Clients (all present pre-decomposed bankIdx + bankAddr):
//   LSU scalar:  scalar loads (read) and scalar stores (masked write).
//   LSU vector:  VLOAD (read) and VSTORE (full-line write).
//   DMA:         DRAM→VMEM loads (full-line write) and VMEM→DRAM stores (read).
//   TileLink:    Saturn / SBUS host access (byte-masked read/write).
//
// Software guarantees LSU scalar and LSU vector never target the same
// block bank on the same VMEM port simultaneously.  Hardware asserts.
// Combined LSU has unconditional priority over DMA and TileLink.
//
// Per-bank per-port priority:
//   Read:  LSU (scalar | vector) > DMA > TileLink
//   Write: LSU (scalar | vector) > DMA > TileLink
// ============================================================================

package atlas.vmem

import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}
import atlas.common._

class Vmem(p: VmemParams, bundle: TLBundleParameters) extends Module {

  val io = IO(new Bundle {
    val tl = Flipped(new TLBundle(bundle))

    // ── LSU scalar (always granted — deterministic latency) ──
    val lsuScalarRead     = Flipped(Valid(new VmemLineReadPort(p)))
    val lsuScalarReadData = Valid(UInt(p.lineWidthBits.W))
    val lsuScalarWrite    = Flipped(Valid(new MaskedVmemLineWritePort(p)))

    // ── LSU vector (always granted — deterministic latency) ──
    val lsuVecRead     = Flipped(Valid(new VmemLineReadPort(p)))
    val lsuVecReadData = Valid(UInt(p.lineWidthBits.W))
    val lsuVecWrite    = Flipped(Valid(new VmemLineWritePort(p)))

    // ── DMA (grant-based backpressure) ──
    val dmaRead      = Flipped(Valid(new VmemLineReadPort(p)))
    val dmaReadGrant = Output(Bool())
    val dmaReadData  = Valid(UInt(p.lineWidthBits.W))
    val dmaWrite      = Flipped(Valid(new VmemLineWritePort(p)))
    val dmaWriteGrant = Output(Bool())
  })

  // ==========================================================================
  // Bank storage
  // ==========================================================================

  val banks = Seq.fill(p.numBanks) {
    SyncReadMem(p.linesPerBank, Vec(p.lineBytes, UInt(8.W)))
  }

  // ==========================================================================
  // TileLink decoding
  // ==========================================================================

  val tl = io.tl
  val tlByteAddr = tl.a.bits.address(p.byteAddrBits - 1, 0)
  val tlLineAddr = tlByteAddr(p.byteAddrBits - 1, p.lineOffBits)
  val tlBankIdx  = p.getBankIdx(tlLineAddr)
  val tlBankAddr = p.getBankAddr(tlLineAddr)
  val tlIsGet    = tl.a.bits.opcode === TLMessages.Get
  val tlIsPut    = tl.a.bits.opcode === TLMessages.PutFullData ||
                   tl.a.bits.opcode === TLMessages.PutPartialData

  // ==========================================================================
  // Software-scheduling assertions
  // ==========================================================================

  assert(!(io.lsuScalarRead.valid && io.lsuVecRead.valid &&
           io.lsuScalarRead.bits.bankIdx === io.lsuVecRead.bits.bankIdx),
    "ASSERT FAIL: scalar load and VLOAD target same VMEM bank")

  assert(!(io.lsuScalarWrite.valid && io.lsuVecWrite.valid &&
           io.lsuScalarWrite.bits.bankIdx === io.lsuVecWrite.bits.bankIdx),
    "ASSERT FAIL: scalar store and VSTORE target same VMEM bank")

  // ==========================================================================
  // Client IDs for read-response routing
  // ==========================================================================

  val CLIENT_NONE       = 0.U(3.W)
  val CLIENT_LSU_SCALAR = 1.U(3.W)
  val CLIENT_LSU_VEC    = 2.U(3.W)
  val CLIENT_DMA        = 3.U(3.W)
  val CLIENT_TL         = 4.U(3.W)

  val r1_bankReadClient = Reg(Vec(p.numBanks, UInt(3.W)))
  val r1_bankReadValid  = RegInit(VecInit(Seq.fill(p.numBanks)(false.B)))
  val r1_bankReadData   = Wire(Vec(p.numBanks, Vec(p.lineBytes, UInt(8.W))))

  val r1_tlSource = RegNext(tl.a.bits.source)
  val r1_tlSize   = RegNext(tl.a.bits.size)
  val r1_tlRead   = RegInit(false.B)
  val r1_tlWrite  = RegInit(false.B)

  val dmaReadGranted  = WireDefault(false.B)
  val dmaWriteGranted = WireDefault(false.B)
  val tlAccepted      = WireDefault(false.B)

  // ==========================================================================
  // Per-bank arbitration
  // ==========================================================================

  for (b <- 0 until p.numBanks) {
    val bank = banks(b)

    // ── Read port ──────────────────────────────────────────────
    val lsuScalarWantsRead = io.lsuScalarRead.valid && io.lsuScalarRead.bits.bankIdx === b.U
    val lsuVecWantsRead    = io.lsuVecRead.valid    && io.lsuVecRead.bits.bankIdx === b.U
    val dmaWantsRead       = io.dmaRead.valid       && io.dmaRead.bits.bankIdx === b.U
    val tlWantsRead        = tl.a.valid && tlIsGet   && tlBankIdx === b.U

    // Scalar loads and VLOAD are mutually exclusive per bank (asserted above).
    val lsuWantsRead = lsuScalarWantsRead || lsuVecWantsRead

    val readAddr   = Wire(UInt(p.bankLineAddrBits.W))
    val readEn     = Wire(Bool())
    val readClient = Wire(UInt(3.W))

    when(lsuScalarWantsRead) {
      readAddr   := io.lsuScalarRead.bits.bankAddr
      readEn     := true.B
      readClient := CLIENT_LSU_SCALAR
    }.elsewhen(lsuVecWantsRead) {
      readAddr   := io.lsuVecRead.bits.bankAddr
      readEn     := true.B
      readClient := CLIENT_LSU_VEC
    }.elsewhen(dmaWantsRead && !lsuWantsRead) {
      readAddr   := io.dmaRead.bits.bankAddr
      readEn     := true.B
      readClient := CLIENT_DMA
    }.elsewhen(tlWantsRead && !lsuWantsRead && !dmaWantsRead) {
      readAddr   := tlBankAddr
      readEn     := true.B
      readClient := CLIENT_TL
    }.otherwise {
      readAddr   := 0.U
      readEn     := false.B
      readClient := CLIENT_NONE
    }

    when(readEn && readClient === CLIENT_DMA) { dmaReadGranted := true.B }
    when(readEn && readClient === CLIENT_TL)  { tlAccepted     := true.B }

    r1_bankReadData(b)   := bank.read(readAddr, readEn)
    r1_bankReadClient(b) := Mux(readEn, readClient, CLIENT_NONE)
    r1_bankReadValid(b)  := readEn

    // ── Write port ─────────────────────────────────────────────
    val lsuScalarWantsWrite = io.lsuScalarWrite.valid && io.lsuScalarWrite.bits.bankIdx === b.U
    val lsuVecWantsWrite    = io.lsuVecWrite.valid    && io.lsuVecWrite.bits.bankIdx === b.U
    val dmaWantsWrite       = io.dmaWrite.valid       && io.dmaWrite.bits.bankIdx === b.U
    val tlWantsWrite        = tl.a.valid && tlIsPut    && tlBankIdx === b.U

    val lsuWantsWrite = lsuScalarWantsWrite || lsuVecWantsWrite

    // 1. Create a single set of multiplexed write signals
    val writeEn   = WireDefault(false.B)
    val writeAddr = WireDefault(0.U(p.bankLineAddrBits.W))
    val writeData = WireDefault(VecInit(Seq.fill(p.lineBytes)(0.U(8.W))))
    val writeMask = WireDefault(VecInit(Seq.fill(p.lineBytes)(false.B)))

    when(lsuScalarWantsWrite) {
      // ── LSU scalar: per-byte masked write ──
      writeEn   := true.B
      writeAddr := io.lsuScalarWrite.bits.bankAddr
      writeMask := io.lsuScalarWrite.bits.mask
      for (i <- 0 until p.lineBytes) {
        writeData(i) := io.lsuScalarWrite.bits.data(i * 8 + 7, i * 8)
      }

    }.elsewhen(lsuVecWantsWrite) {
      // ── LSU vector: full-line write ──
      writeEn   := true.B
      writeAddr := io.lsuVecWrite.bits.bankAddr
      for (i <- 0 until p.lineBytes) {
        writeData(i) := io.lsuVecWrite.bits.data(i * 8 + 7, i * 8)
        writeMask(i) := true.B
      }

    }.elsewhen(dmaWantsWrite && !lsuWantsWrite) {
      // ── DMA: full-line write ──
      writeEn   := true.B
      writeAddr := io.dmaWrite.bits.bankAddr
      for (i <- 0 until p.lineBytes) {
        writeData(i) := io.dmaWrite.bits.data(i * 8 + 7, i * 8)
        writeMask(i) := true.B
      }
      dmaWriteGranted := true.B

    }.elsewhen(tlWantsWrite && !lsuWantsWrite && !dmaWantsWrite) {
      // ── TileLink: byte-masked write ──
      writeEn   := true.B
      writeAddr := tlBankAddr
      for (i <- 0 until p.lineBytes) {
        writeData(i) := tl.a.bits.data(i * 8 + 7, i * 8)
        writeMask(i) := tl.a.bits.mask(i)
      }
      tlAccepted := true.B
    }

    // 2. Execute the write using a SINGLE hardware port
    when(writeEn) {
      bank.write(writeAddr, writeData, writeMask)
    }
  }

  // ==========================================================================
  // Read-response routing
  // ==========================================================================

  def flattenBankData(bdata: Vec[UInt]): UInt = Cat(bdata.reverse)

  val lsuScalarRespHot = VecInit((0 until p.numBanks).map(b =>
    r1_bankReadValid(b) && r1_bankReadClient(b) === CLIENT_LSU_SCALAR))
  io.lsuScalarReadData.valid := lsuScalarRespHot.asUInt.orR
  io.lsuScalarReadData.bits  := Mux1H(lsuScalarRespHot, r1_bankReadData.map(flattenBankData))

  val lsuVecRespHot = VecInit((0 until p.numBanks).map(b =>
    r1_bankReadValid(b) && r1_bankReadClient(b) === CLIENT_LSU_VEC))
  io.lsuVecReadData.valid := lsuVecRespHot.asUInt.orR
  io.lsuVecReadData.bits  := Mux1H(lsuVecRespHot, r1_bankReadData.map(flattenBankData))

  val dmaRespHot = VecInit((0 until p.numBanks).map(b =>
    r1_bankReadValid(b) && r1_bankReadClient(b) === CLIENT_DMA))
  io.dmaReadData.valid := dmaRespHot.asUInt.orR
  io.dmaReadData.bits  := Mux1H(dmaRespHot, r1_bankReadData.map(flattenBankData))

  // ==========================================================================
  // Grants
  // ==========================================================================

  io.dmaReadGrant  := dmaReadGranted
  io.dmaWriteGrant := dmaWriteGranted

  // ==========================================================================
  // TileLink D-channel
  // ==========================================================================

  tl.a.ready := tlAccepted
  r1_tlRead  := tl.a.fire && tlIsGet
  r1_tlWrite := tl.a.fire && tlIsPut

  val tlRespHot = VecInit((0 until p.numBanks).map(b =>
    r1_bankReadValid(b) && r1_bankReadClient(b) === CLIENT_TL))

  tl.d.valid        := r1_tlRead || r1_tlWrite
  tl.d.bits.opcode  := Mux(r1_tlRead, TLMessages.AccessAckData, TLMessages.AccessAck)
  tl.d.bits.param   := 0.U
  tl.d.bits.size    := r1_tlSize
  tl.d.bits.source  := r1_tlSource
  tl.d.bits.sink    := 0.U
  tl.d.bits.denied  := false.B
  tl.d.bits.data    := Mux(r1_tlRead, Mux1H(tlRespHot, r1_bankReadData.map(flattenBankData)), 0.U)
  tl.d.bits.corrupt := false.B
}

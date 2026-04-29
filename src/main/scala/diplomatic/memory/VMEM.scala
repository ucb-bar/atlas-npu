// ============================================================================
// Vmem.scala — 8-bank block-banked vector scratchpad with masked writes.
//
// Storage: numBanks × SyncReadMem(linesPerBank, Vec(lineBytes, UInt(8.W)))
//          Each bank: 1 shared read/write port (1RW SRAM).
//          Byte-granularity masking via SyncReadMem native mask.
//
// Clients (all present pre-decomposed bankIdx + bankAddr):
//   LSU scalar:  scalar loads (read) and scalar stores (masked write).
//   LSU vector:  VLOAD (read) and VSTORE (full-line write).
//   DMA:         DRAM→VMEM loads (full-line write) and VMEM→DRAM stores (read).
//   TileLink:    Saturn / SBUS host access (byte-masked read/write).
//
// Software guarantees LSU scalar and LSU vector never target the same block
// bank simultaneously, whether the requests are reads or writes. Hardware
// asserts.
// Combined LSU has unconditional priority over DMA and TileLink.
//
// Per-bank access priority:
//   scalar store > VSTORE > scalar load > VLOAD > DMA write > DMA read
//   > TileLink write > TileLink read
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

  assert(!(io.lsuScalarRead.valid && io.lsuScalarWrite.valid &&
           io.lsuScalarRead.bits.bankIdx === io.lsuScalarWrite.bits.bankIdx),
    "ASSERT FAIL: scalar load and scalar store target same VMEM bank")

  assert(!(io.lsuScalarRead.valid && io.lsuVecWrite.valid &&
           io.lsuScalarRead.bits.bankIdx === io.lsuVecWrite.bits.bankIdx),
    "ASSERT FAIL: scalar load and VSTORE target same VMEM bank")

  assert(!(io.lsuScalarWrite.valid && io.lsuVecRead.valid &&
           io.lsuScalarWrite.bits.bankIdx === io.lsuVecRead.bits.bankIdx),
    "ASSERT FAIL: scalar store and VLOAD target same VMEM bank")

  assert(!(io.lsuVecRead.valid && io.lsuVecWrite.valid &&
           io.lsuVecRead.bits.bankIdx === io.lsuVecWrite.bits.bankIdx),
    "ASSERT FAIL: VLOAD and VSTORE target same VMEM bank")

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

  val r1_tlRead   = RegInit(false.B)
  val r1_tlWrite  = RegInit(false.B)
  val r1_tlSource = Reg(UInt(tl.params.sourceBits.W))
  val r1_tlSize   = Reg(UInt(tl.params.sizeBits.W))

  // Held-until-fire D-response slot (mirrors CSRFile.scala).
  val dValid  = RegInit(false.B)
  val dOpcode = Reg(UInt(3.W))
  val dSize   = Reg(UInt(tl.params.sizeBits.W))
  val dSource = Reg(UInt(tl.params.sourceBits.W))
  val dData   = Reg(UInt(tl.params.dataBits.W))

  val dmaReadGranted  = WireDefault(false.B)
  val dmaWriteGranted = WireDefault(false.B)
  val tlAccepted      = WireDefault(false.B)

  // ==========================================================================
  // Per-bank arbitration
  // ==========================================================================

  def splitLine(data: UInt): Vec[UInt] =
    VecInit((0 until p.lineBytes).map(i => data(i * 8 + 7, i * 8)))

  // dCanAccept gates both tl.a.ready and the per-bank decode below; the
  // latter prevents the bank from performing an SRAM access for a TL
  // request that has not fired.
  val r1_tlValid = r1_tlRead || r1_tlWrite
  val dCanAccept = !r1_tlValid && (!dValid || tl.d.ready)

  for (b <- 0 until p.numBanks) {
    val bank = banks(b)

    // ── Request decode ─────────────────────────────────────────
    val lsuScalarWantsRead = io.lsuScalarRead.valid && io.lsuScalarRead.bits.bankIdx === b.U
    val lsuVecWantsRead    = io.lsuVecRead.valid    && io.lsuVecRead.bits.bankIdx === b.U
    val dmaWantsRead       = io.dmaRead.valid       && io.dmaRead.bits.bankIdx === b.U
    val tlWantsRead        = tl.a.valid && dCanAccept && tlIsGet && tlBankIdx === b.U

    val lsuScalarWantsWrite = io.lsuScalarWrite.valid && io.lsuScalarWrite.bits.bankIdx === b.U
    val lsuVecWantsWrite    = io.lsuVecWrite.valid    && io.lsuVecWrite.bits.bankIdx === b.U
    val dmaWantsWrite       = io.dmaWrite.valid       && io.dmaWrite.bits.bankIdx === b.U
    val tlWantsWrite        = tl.a.valid && dCanAccept && tlIsPut && tlBankIdx === b.U

    // ── Shared 1RW access port ─────────────────────────────────
    val accessEn      = WireDefault(false.B)
    val accessIsWrite = WireDefault(false.B)
    val accessAddr    = WireDefault(0.U(p.bankLineAddrBits.W))
    val readClient    = WireDefault(CLIENT_NONE)
    val writeData = WireDefault(VecInit(Seq.fill(p.lineBytes)(0.U(8.W))))
    val writeMask = WireDefault(VecInit(Seq.fill(p.lineBytes)(false.B)))
    val fullMask  = VecInit(Seq.fill(p.lineBytes)(true.B))

    when(lsuScalarWantsWrite) {
      accessEn      := true.B
      accessIsWrite := true.B
      accessAddr    := io.lsuScalarWrite.bits.bankAddr
      writeData     := splitLine(io.lsuScalarWrite.bits.data)
      writeMask     := io.lsuScalarWrite.bits.mask

    }.elsewhen(lsuVecWantsWrite) {
      accessEn      := true.B
      accessIsWrite := true.B
      accessAddr    := io.lsuVecWrite.bits.bankAddr
      writeData     := splitLine(io.lsuVecWrite.bits.data)
      writeMask     := fullMask

    }.elsewhen(lsuScalarWantsRead) {
      accessEn   := true.B
      accessAddr := io.lsuScalarRead.bits.bankAddr
      readClient := CLIENT_LSU_SCALAR

    }.elsewhen(lsuVecWantsRead) {
      accessEn   := true.B
      accessAddr := io.lsuVecRead.bits.bankAddr
      readClient := CLIENT_LSU_VEC

    }.elsewhen(dmaWantsWrite) {
      accessEn      := true.B
      accessIsWrite := true.B
      accessAddr    := io.dmaWrite.bits.bankAddr
      writeData     := splitLine(io.dmaWrite.bits.data)
      writeMask     := fullMask
      dmaWriteGranted := true.B

    }.elsewhen(dmaWantsRead) {
      accessEn   := true.B
      accessAddr := io.dmaRead.bits.bankAddr
      readClient := CLIENT_DMA
      dmaReadGranted := true.B

    }.elsewhen(tlWantsWrite) {
      accessEn      := true.B
      accessIsWrite := true.B
      accessAddr    := tlBankAddr
      writeData     := splitLine(tl.a.bits.data)
      writeMask     := VecInit((0 until p.lineBytes).map(i => tl.a.bits.mask(i)))
      tlAccepted := true.B

    }.elsewhen(tlWantsRead) {
      accessEn   := true.B
      accessAddr := tlBankAddr
      readClient := CLIENT_TL
      tlAccepted := true.B
    }

    r1_bankReadData(b)   := bank.readWrite(accessAddr, writeData, writeMask, accessEn, accessIsWrite)
    r1_bankReadClient(b) := Mux(accessEn && !accessIsWrite, readClient, CLIENT_NONE)
    r1_bankReadValid(b)  := accessEn && !accessIsWrite
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

  tl.a.ready := tlAccepted && dCanAccept
  r1_tlRead  := tl.a.fire && tlIsGet
  r1_tlWrite := tl.a.fire && tlIsPut
  when (tl.a.fire) {
    r1_tlSource := tl.a.bits.source
    r1_tlSize   := tl.a.bits.size
  }

  val tlRespHot = VecInit((0 until p.numBanks).map(b =>
    r1_bankReadValid(b) && r1_bankReadClient(b) === CLIENT_TL))

  val r1_tlData = Mux(r1_tlRead,
    Mux1H(tlRespHot, r1_bankReadData.map(flattenBankData)), 0.U)

  when (r1_tlValid) {
    dValid  := true.B
    dOpcode := Mux(r1_tlRead, TLMessages.AccessAckData, TLMessages.AccessAck)
    dSize   := r1_tlSize
    dSource := r1_tlSource
    dData   := r1_tlData
  } .elsewhen (tl.d.fire) {
    dValid := false.B
  }

  assert(!(r1_tlValid && dValid && !tl.d.ready),
    "VMEM TL D response slot overflow")

  tl.d.valid        := dValid
  tl.d.bits.opcode  := dOpcode
  tl.d.bits.param   := 0.U
  tl.d.bits.size    := dSize
  tl.d.bits.source  := dSource
  tl.d.bits.sink    := 0.U
  tl.d.bits.denied  := false.B
  tl.d.bits.data    := dData
  tl.d.bits.corrupt := false.B
}

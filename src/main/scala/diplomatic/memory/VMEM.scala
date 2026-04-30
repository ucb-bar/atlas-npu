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

  class AccessTreeSel extends Bundle {
    val valid         = Bool()
    val isWrite       = Bool()
    val addr          = UInt(p.bankLineAddrBits.W)
    val readClient    = UInt(3.W)
    val writeData     = Vec(p.lineBytes, UInt(8.W))
    val writeMask     = Vec(p.lineBytes, Bool())
    val dmaReadGrant  = Bool()
    val dmaWriteGrant = Bool()
    val tlAccepted    = Bool()
  }

  class RespTreeSel extends Bundle {
    val valid = Bool()
    val data  = UInt(p.lineWidthBits.W)
  }

  def mergeAccessSel(lhs: AccessTreeSel, rhs: AccessTreeSel): AccessTreeSel = {
    val out = Wire(new AccessTreeSel)
    out.valid         := lhs.valid || rhs.valid
    out.isWrite       := Mux(lhs.valid, lhs.isWrite, rhs.isWrite)
    out.addr          := Mux(lhs.valid, lhs.addr, rhs.addr)
    out.readClient    := Mux(lhs.valid, lhs.readClient, rhs.readClient)
    out.writeData     := Mux(lhs.valid, lhs.writeData, rhs.writeData)
    out.writeMask     := Mux(lhs.valid, lhs.writeMask, rhs.writeMask)
    out.dmaReadGrant  := Mux(lhs.valid, lhs.dmaReadGrant, rhs.dmaReadGrant)
    out.dmaWriteGrant := Mux(lhs.valid, lhs.dmaWriteGrant, rhs.dmaWriteGrant)
    out.tlAccepted    := Mux(lhs.valid, lhs.tlAccepted, rhs.tlAccepted)
    out
  }

  def mergeRespSel(lhs: RespTreeSel, rhs: RespTreeSel): RespTreeSel = {
    val out = Wire(new RespTreeSel)
    out.valid := lhs.valid || rhs.valid
    out.data  := Mux(lhs.valid, lhs.data, rhs.data)
    out
  }

  def treeReduce[T](leaves: Seq[T])(merge: (T, T) => T): T = {
    require(leaves.nonEmpty, "treeReduce requires at least one leaf")
    if (leaves.length == 1) {
      leaves.head
    } else {
      val nextLevel = leaves.grouped(2).map { group =>
        if (group.length == 1) group.head else merge(group.head, group(1))
      }.toSeq
      treeReduce(nextLevel)(merge)
    }
  }

  def makeAccessLeaf(
    valid:         Bool,
    isWrite:       Bool,
    addr:          UInt,
    readClient:    UInt,
    writeData:     Vec[UInt],
    writeMask:     Vec[Bool],
    dmaReadGrant:  Bool,
    dmaWriteGrant: Bool,
    tlAccepted:    Bool
  ): AccessTreeSel = {
    val leaf = Wire(new AccessTreeSel)
    leaf.valid         := valid
    leaf.isWrite       := isWrite
    leaf.addr          := addr
    leaf.readClient    := readClient
    leaf.writeData     := writeData
    leaf.writeMask     := writeMask
    leaf.dmaReadGrant  := dmaReadGrant
    leaf.dmaWriteGrant := dmaWriteGrant
    leaf.tlAccepted    := tlAccepted
    leaf
  }

  def makeRespLeaf(valid: Bool, data: UInt): RespTreeSel = {
    val leaf = Wire(new RespTreeSel)
    leaf.valid := valid
    leaf.data  := data
    leaf
  }

  // dCanAccept gates both tl.a.ready and the per-bank decode below; the
  // latter prevents the bank from performing an SRAM access for a TL
  // request that has not fired.
  val r1_tlValid = r1_tlRead || r1_tlWrite
  val dCanAccept = !r1_tlValid && (!dValid || tl.d.ready)

  val zeroData = VecInit(Seq.fill(p.lineBytes)(0.U(8.W)))
  val zeroMask = VecInit(Seq.fill(p.lineBytes)(false.B))
  val fullMask = VecInit(Seq.fill(p.lineBytes)(true.B))

  val lsuScalarWriteData = splitLine(io.lsuScalarWrite.bits.data)
  val lsuVecWriteData    = splitLine(io.lsuVecWrite.bits.data)
  val dmaWriteData       = splitLine(io.dmaWrite.bits.data)
  val tlWriteData        = splitLine(tl.a.bits.data)
  val tlWriteMask        = VecInit((0 until p.lineBytes).map(i => tl.a.bits.mask(i)))

  def bankOH(bankIdx: UInt, valid: Bool): UInt =
    UIntToOH(bankIdx, p.numBanks) & Fill(p.numBanks, valid)

  val lsuScalarReadBankOH  = bankOH(io.lsuScalarRead.bits.bankIdx, io.lsuScalarRead.valid)
  val lsuVecReadBankOH     = bankOH(io.lsuVecRead.bits.bankIdx, io.lsuVecRead.valid)
  val dmaReadBankOH        = bankOH(io.dmaRead.bits.bankIdx, io.dmaRead.valid)
  val tlReadBankOH         = bankOH(tlBankIdx, tl.a.valid && dCanAccept && tlIsGet)
  val lsuScalarWriteBankOH = bankOH(io.lsuScalarWrite.bits.bankIdx, io.lsuScalarWrite.valid)
  val lsuVecWriteBankOH    = bankOH(io.lsuVecWrite.bits.bankIdx, io.lsuVecWrite.valid)
  val dmaWriteBankOH       = bankOH(io.dmaWrite.bits.bankIdx, io.dmaWrite.valid)
  val tlWriteBankOH        = bankOH(tlBankIdx, tl.a.valid && dCanAccept && tlIsPut)

  val bankDmaReadGrant  = Wire(Vec(p.numBanks, Bool()))
  val bankDmaWriteGrant = Wire(Vec(p.numBanks, Bool()))
  val bankTlAccepted    = Wire(Vec(p.numBanks, Bool()))

  for (b <- 0 until p.numBanks) {
    val bank = banks(b)

    val accessSel = treeReduce(Seq(
      makeAccessLeaf(lsuScalarWriteBankOH(b), true.B, io.lsuScalarWrite.bits.bankAddr,
        CLIENT_NONE, lsuScalarWriteData, io.lsuScalarWrite.bits.mask,
        false.B, false.B, false.B),
      makeAccessLeaf(lsuVecWriteBankOH(b), true.B, io.lsuVecWrite.bits.bankAddr,
        CLIENT_NONE, lsuVecWriteData, fullMask,
        false.B, false.B, false.B),
      makeAccessLeaf(lsuScalarReadBankOH(b), false.B, io.lsuScalarRead.bits.bankAddr,
        CLIENT_LSU_SCALAR, zeroData, zeroMask,
        false.B, false.B, false.B),
      makeAccessLeaf(lsuVecReadBankOH(b), false.B, io.lsuVecRead.bits.bankAddr,
        CLIENT_LSU_VEC, zeroData, zeroMask,
        false.B, false.B, false.B),
      makeAccessLeaf(dmaWriteBankOH(b), true.B, io.dmaWrite.bits.bankAddr,
        CLIENT_NONE, dmaWriteData, fullMask,
        false.B, true.B, false.B),
      makeAccessLeaf(dmaReadBankOH(b), false.B, io.dmaRead.bits.bankAddr,
        CLIENT_DMA, zeroData, zeroMask,
        true.B, false.B, false.B),
      makeAccessLeaf(tlWriteBankOH(b), true.B, tlBankAddr,
        CLIENT_NONE, tlWriteData, tlWriteMask,
        false.B, false.B, true.B),
      makeAccessLeaf(tlReadBankOH(b), false.B, tlBankAddr,
        CLIENT_TL, zeroData, zeroMask,
        false.B, false.B, true.B)
    ))(mergeAccessSel)

    r1_bankReadData(b)   := bank.readWrite(
      accessSel.addr,
      accessSel.writeData,
      accessSel.writeMask,
      accessSel.valid,
      accessSel.isWrite
    )
    r1_bankReadClient(b) := Mux(accessSel.valid && !accessSel.isWrite, accessSel.readClient, CLIENT_NONE)
    r1_bankReadValid(b)  := accessSel.valid && !accessSel.isWrite
    bankDmaReadGrant(b)  := accessSel.dmaReadGrant
    bankDmaWriteGrant(b) := accessSel.dmaWriteGrant
    bankTlAccepted(b)    := accessSel.tlAccepted
  }

  dmaReadGranted  := treeReduce((0 until p.numBanks).map(b => bankDmaReadGrant(b)))(_ || _)
  dmaWriteGranted := treeReduce((0 until p.numBanks).map(b => bankDmaWriteGrant(b)))(_ || _)
  tlAccepted      := treeReduce((0 until p.numBanks).map(b => bankTlAccepted(b)))(_ || _)

  // ==========================================================================
  // Read-response routing
  // ==========================================================================

  def flattenBankData(bdata: Vec[UInt]): UInt = Cat(bdata.reverse)
  val flatBankReadData = r1_bankReadData.map(flattenBankData)

  def selectReadResp(client: UInt): RespTreeSel = {
    val leaves = (0 until p.numBanks).map { b =>
      makeRespLeaf(r1_bankReadValid(b) && r1_bankReadClient(b) === client, flatBankReadData(b))
    }
    treeReduce(leaves)(mergeRespSel)
  }

  val lsuScalarRespSel = selectReadResp(CLIENT_LSU_SCALAR)
  io.lsuScalarReadData.valid := lsuScalarRespSel.valid
  io.lsuScalarReadData.bits  := lsuScalarRespSel.data

  val lsuVecRespSel = selectReadResp(CLIENT_LSU_VEC)
  io.lsuVecReadData.valid := lsuVecRespSel.valid
  io.lsuVecReadData.bits  := lsuVecRespSel.data

  val dmaRespSel = selectReadResp(CLIENT_DMA)
  io.dmaReadData.valid := dmaRespSel.valid
  io.dmaReadData.bits  := dmaRespSel.data

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

  val tlRespSel = selectReadResp(CLIENT_TL)
  val r1_tlData = Mux(r1_tlRead, tlRespSel.data, 0.U)

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

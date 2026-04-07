// ============================================================================
// AtlasTileTestHarness.scala — Simulation-only wrapper for AtlasCore.
//
// Provides:
//   • TLBundleSRAM  — simple TileLink SRAM responder for DMA testing.
//   • AtlasTileTestHarness — wires AtlasCore to a fake DRAM, ties off CSR,
//     and exposes test-friendly write/read ports for IMEM and DRAM.
// ============================================================================

package atlas.tile

import chisel3._
import chisel3.util._
import atlas.common._
import atlas.scalar.{AtlasMemMap, ScalarISA}
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

// ============================================================================
// TestImemWritePort — word-addressed instruction write for test setup
// ============================================================================

class TestImemWritePort extends Bundle {
  val addr = UInt(AtlasMemMap.IMEM_ADDR_BITS.W)
  val data = UInt(32.W)
}

// ============================================================================
// TLBundleSRAM — minimal TileLink SRAM responder
// ============================================================================

/** Single-cycle TileLink Get/Put responder backed by SyncReadMem.
  *
  * Also exposes direct test read/write ports for DRAM preloading and
  * result checking without going through TileLink.
  *
  * @param baseAddr     Base byte address of this SRAM region.
  * @param sizeBytes    Total capacity in bytes.
  * @param tlBP         TileLink bundle parameters.
  */
class TLBundleSRAM(
    baseAddr:  Long,
    sizeBytes: Int,
    tlBP:      TLBundleParameters
) extends Module {

  private val dataBytes = tlBP.dataBits / 8
  private val numWords  = sizeBytes / dataBytes
  private val idxBits   = log2Ceil(numWords)
  private val offBits   = log2Ceil(dataBytes)

  val io = IO(new Bundle {
    val tl             = Flipped(new TLBundle(tlBP))
    val testWriteValid = Input(Bool())
    val testWriteIdx   = Input(UInt(idxBits.W))
    val testWriteData  = Input(UInt(tlBP.dataBits.W))
    val testReadIdx    = Input(UInt(idxBits.W))
    val testReadData   = Output(UInt(tlBP.dataBits.W))
  })

  val mem = SyncReadMem(numWords, UInt(tlBP.dataBits.W))

  // ── Test backdoor write ──
  when(io.testWriteValid) { mem.write(io.testWriteIdx, io.testWriteData) }

  // ── Test backdoor read ──
  io.testReadData := mem.read(io.testReadIdx)

  // ── TileLink FSM ──
  val sIdle :: sResp :: Nil = Enum(2)
  val state      = RegInit(sIdle)
  val respIsGet  = Reg(Bool())
  val respSource = Reg(UInt(tlBP.sourceBits.W))
  val respSize   = Reg(UInt(tlBP.sizeBits.W))

  private def toIdx(addr: UInt): UInt =
    ((addr - baseAddr.U(tlBP.addressBits.W)) >> offBits)(idxBits - 1, 0)

  val rdAddr = Wire(UInt(idxBits.W)); rdAddr := 0.U
  val rdEn   = Wire(Bool());          rdEn   := false.B
  val rdData = mem.read(rdAddr, rdEn)

  // Channel D defaults.
  io.tl.a.ready        := (state === sIdle)
  io.tl.d.valid        := (state === sResp)
  io.tl.d.bits.opcode  := Mux(respIsGet, TLMessages.AccessAckData, TLMessages.AccessAck)
  io.tl.d.bits.param   := 0.U
  io.tl.d.bits.size    := respSize
  io.tl.d.bits.source  := respSource
  io.tl.d.bits.sink    := 0.U
  io.tl.d.bits.denied  := false.B
  io.tl.d.bits.data    := Mux(respIsGet, rdData, 0.U)
  io.tl.d.bits.corrupt := false.B

  switch(state) {
    is(sIdle) {
      when(io.tl.a.fire) {
        val a = io.tl.a.bits
        respSource := a.source
        respSize   := a.size
        val idx = toIdx(a.address)
        when(a.opcode === TLMessages.Get) {
          rdAddr := idx; rdEn := true.B; respIsGet := true.B
        }.otherwise {
          mem.write(idx, a.data); respIsGet := false.B
        }
        state := sResp
      }
    }
    is(sResp) {
      when(io.tl.d.fire) { state := sIdle }
    }
  }
}

// ============================================================================
// AtlasTileTestHarness
// ============================================================================

/** Simulation wrapper that connects AtlasCore to a fake DRAM SRAM, ties
  * off the VMEM port, and provides IMEM + DRAM backdoor access.
  * A softReset input issues a TL write to the CSR softReset register
  * to start execution.
  *
  * @param tp             Atlas design parameters.
  * @param dramBase        Byte base address of the fake DRAM region.
  * @param dramSizeBytes   Capacity of the fake DRAM in bytes.
  */
class AtlasTileTestHarness(
    tp:            AtlasParams = AtlasParams(),
    dramBase:      Long        = 0x60000000L,
    dramSizeBytes: Int         = 4 * 1024 * 1024
) extends Module {

  private val dmaP         = tp.dma
  private val beatBits     = dmaP.beatBytes * 8
  private val sramWords    = dramSizeBytes / dmaP.beatBytes
  private val sramIdxBits  = log2Ceil(sramWords)

  // TileLink bundle parameters for each port.
  private val imemBP = TLBundleParameters(
    addressBits = 32, dataBits = 32, sourceBits = 4,
    sinkBits = 1, sizeBits = 3,
    echoFields = Nil, requestFields = Nil, responseFields = Nil, hasBCE = false)

  private val csrBP = TLBundleParameters(
    addressBits = 32, dataBits = 32, sourceBits = 4,
    sinkBits = 1, sizeBits = 3,
    echoFields = Nil, requestFields = Nil, responseFields = Nil, hasBCE = false)

  private val dmaBP = TLBundleParameters(
    addressBits = 64, dataBits = beatBits, sourceBits = dmaP.tagBits,
    sinkBits = 1, sizeBits = 3,
    echoFields = Nil, requestFields = Nil, responseFields = Nil, hasBCE = false)

  private val vmemBP = TLBundleParameters(
    addressBits = 32, dataBits = 32, sourceBits = 4,
    sinkBits = 1, sizeBits = 3,
    echoFields = Nil, requestFields = Nil, responseFields = Nil, hasBCE = false)

  val io = IO(new Bundle {
    /** IMEM backdoor write (word-addressed). */
    val imemWrite    = Flipped(Valid(new TestImemWritePort))
    /** Pulse high to issue a softReset to the core via CSR TL write. */
    val softReset    = Input(Bool())
    /** Core halted flag. */
    val halted       = Output(Bool())
    /** DRAM backdoor write. */
    val ramWriteValid = Input(Bool())
    val ramWriteIdx   = Input(UInt(sramIdxBits.W))
    val ramWriteData  = Input(UInt(beatBits.W))
    /** DRAM backdoor read. */
    val ramReadIdx    = Input(UInt(sramIdxBits.W))
    val ramReadData   = Output(UInt(beatBits.W))
    /** Debug outputs. */
    val dbg = Output(new Bundle {
      val mxu1DataBusy = Bool()
      val mxu1CompBusy = Bool()
      val xluBusy      = Bool()
      val dmaBusy      = Vec(dmaP.numChannels, Bool())
      val lsuBusy      = Bool()
      val scaleRegs    = Vec(ScalarISA.NUM_SCALE_REGS, UInt(8.W))
      val dbg0         = UInt(32.W)
      val dbg1         = UInt(32.W)
    })
  })

  // ==========================================================================
  // Core + fake DRAM
  // ==========================================================================
    
  val core = Module(new AtlasCore(tp, imemBP, csrBP, dmaBP, vmemBP))
  val dmaRam = Module(new TLBundleSRAM(dramBase, dramSizeBytes, dmaBP))

  dmaRam.io.tl             <> core.io.dmaTL
  dmaRam.io.testWriteValid := io.ramWriteValid
  dmaRam.io.testWriteIdx   := io.ramWriteIdx
  dmaRam.io.testWriteData  := io.ramWriteData
  dmaRam.io.testReadIdx    := io.ramReadIdx
  io.ramReadData           := dmaRam.io.testReadData

  // ==========================================================================
  // CSR TL — softReset via TileLink write to CSR offset 0x18
  // ==========================================================================

  val csrIdle :: csrWaitD :: Nil = Enum(2)
  val csrState = RegInit(csrIdle)
  val csrSoftResetAddr = (AtlasMemMap.CSR_BASE + 0x18).U(32.W)

  core.io.csrTL.a.valid        := io.softReset && (csrState === csrIdle)
  core.io.csrTL.a.bits         := DontCare
  core.io.csrTL.a.bits.opcode  := TLMessages.PutFullData
  core.io.csrTL.a.bits.param   := 0.U
  core.io.csrTL.a.bits.size    := 2.U
  core.io.csrTL.a.bits.source  := 0.U
  core.io.csrTL.a.bits.address := csrSoftResetAddr
  core.io.csrTL.a.bits.mask    := "hF".U(4.W)
  core.io.csrTL.a.bits.data    := 1.U
  core.io.csrTL.a.bits.corrupt := false.B
  core.io.csrTL.d.ready        := (csrState === csrWaitD)

  switch(csrState) {
    is(csrIdle)  { when(core.io.csrTL.a.fire) { csrState := csrWaitD } }
    is(csrWaitD) { when(core.io.csrTL.d.fire) { csrState := csrIdle } }
  }

  // VMEM TL — tied off 
  core.io.vmemTL.a.valid := false.B
  core.io.vmemTL.a.bits  := DontCare
  core.io.vmemTL.d.ready := true.B

  // IMEM write FSM
  val imIdle :: imWaitD :: Nil = Enum(2)
  val imState   = RegInit(imIdle)
  val doImWrite = io.imemWrite.valid && (imState === imIdle)
  val imByteAddr = AtlasMemMap.IMEM_BASE.U(32.W) + (io.imemWrite.bits.addr << 2)

  core.io.imemTL.a.valid        := doImWrite
  core.io.imemTL.a.bits.opcode  := TLMessages.PutFullData
  core.io.imemTL.a.bits.param   := 0.U
  core.io.imemTL.a.bits.size    := 2.U
  core.io.imemTL.a.bits.source  := 0.U
  core.io.imemTL.a.bits.address := imByteAddr
  core.io.imemTL.a.bits.mask    := "hF".U(4.W)
  core.io.imemTL.a.bits.data    := io.imemWrite.bits.data
  core.io.imemTL.a.bits.corrupt := false.B
  core.io.imemTL.d.ready        := (imState === imWaitD)

  switch(imState) {
    is(imIdle)  { when(core.io.imemTL.a.fire) { imState := imWaitD } }
    is(imWaitD) { when(core.io.imemTL.d.fire) { imState := imIdle } }
  }

  // ==========================================================================
  // Top-level outputs
  // ==========================================================================

  io.halted := core.io.halted
  io.dbg    := core.io.dbg
}

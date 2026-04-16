// ============================================================================
// DMA.scala — Multi-channel DMA engine (banked VMEM architecture).
//
// VMEM ports use pre-decomposed bankIdx/bankAddr via VmemParams helpers.
// Grant signals provide backpressure when LSU holds priority on a bank.
// ============================================================================

package atlas.dma

import atlas.common._
import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

class MemoryRequest(dataBytes: Int, tagBits: Int) extends Bundle {
  val address = UInt(64.W)
  val data    = UInt((dataBytes * 8).W)
  val mask    = UInt(dataBytes.W)
  val tag     = UInt(tagBits.W)
  val isStore = Bool()
}

class MemoryResponse(dataBytes: Int, tagBits: Int) extends Bundle {
  val data = UInt((dataBytes * 8).W)
  val tag  = UInt(tagBits.W)
}

class TileLinkAdapter(
    dataBytes:      Int,
    tagBits:        Int,
    tlBundleParams: TLBundleParameters
) extends Module {
  private val alignmentBits = log2Ceil(dataBytes)
  private val numIds        = 1 << tagBits

  val io = IO(new Bundle {
    val request       = Flipped(Decoupled(new MemoryRequest(dataBytes, tagBits)))
    val response      = Decoupled(new MemoryResponse(dataBytes, tagBits))
    val busy          = Output(Bool())
    val inFlightCount = Output(UInt((tagBits + 1).W))
    val tl = new TLBundle(tlBundleParams)
  })

  // ── Per-ID in-flight bitmap ──────────────────────────────────
  //
  // Tracks exactly which source IDs are outstanding.  A request is
  // only issued when its tag is not already in flight, preventing
  // reuse regardless of response ordering.
  val idInFlight = RegInit(0.U(numIds.W))

  val tagIsFree = !idInFlight(io.request.bits.tag)

  // Update bitmap: set bit on A-fire, clear bit on D-fire.
  val setMask   = Mux(io.tl.a.fire, 1.U(numIds.W) << io.tl.a.bits.source, 0.U)
  val clearMask = Mux(io.tl.d.fire, 1.U(numIds.W) << io.tl.d.bits.source, 0.U)
  idInFlight := (idInFlight | setMask) & ~clearMask

  // ── In-flight count (for busy / DMA throttle) ────────────────
  val inFlightReg = RegInit(0.U((tagBits + 1).W))
  when(io.tl.a.fire || io.tl.d.fire) {
    inFlightReg := inFlightReg + io.tl.a.fire - io.tl.d.fire
  }
  io.busy          := inFlightReg =/= 0.U
  io.inFlightCount := inFlightReg

  // ── Channel A (requests) ─────────────────────────────────────
  // Gate on tagIsFree: even if the DMA's counter wraps, we won't
  // reuse a source ID that hasn't been freed yet.
  io.request.ready := io.tl.a.ready && tagIsFree
  io.tl.a.valid    := io.request.valid && tagIsFree

  val alignedAddress = (io.request.bits.address >> alignmentBits.U) << alignmentBits.U
  io.tl.a.bits.opcode  := Mux(io.request.bits.isStore, TLMessages.PutFullData, TLMessages.Get)
  io.tl.a.bits.param   := 0.U
  io.tl.a.bits.size    := alignmentBits.U
  io.tl.a.bits.source  := io.request.bits.tag
  io.tl.a.bits.address := alignedAddress
  io.tl.a.bits.mask    := Mux(io.request.bits.isStore, io.request.bits.mask, Fill(dataBytes, 1.U(1.W)))
  io.tl.a.bits.data    := io.request.bits.data
  io.tl.a.bits.corrupt := false.B

  // ── Channel D (responses) ────────────────────────────────────
  io.tl.d.ready         := io.response.ready
  io.response.valid     := io.tl.d.valid
  io.response.bits.data := io.tl.d.bits.data
  io.response.bits.tag  := io.tl.d.bits.source
}

object DmaDirection extends ChiselEnum {
  val LoadToVmem    = Value
  val StoreFromVmem = Value
}
import DmaDirection._

class DmaCommand(vmemParams: VmemParams, dmaParams: DmaParams) extends Bundle {
  val opType       = DmaDirection()
  val channelId    = UInt(dmaParams.channelIdBits.W)
  val vmemLineAddr = UInt(vmemParams.lineAddrBits.W)
  val dramAddress  = UInt(64.W)
  val transferSize = UInt(dmaParams.transferSizeBits.W)
}

class DmaEngine(
    params:         AtlasParams,
    tlBundleParams: TLBundleParameters
) extends Module {

  private val dmaP  = params.dma
  private val vmemP = params.vmem
  private val tagBits   = dmaP.tagBits
  private val beatBytes = dmaP.beatBytes
  private val beatBits  = beatBytes * 8

  val io = IO(new Bundle {
    val command     = Flipped(Valid(new DmaCommand(vmemP, dmaP)))
    val channelBusy = Output(Vec(dmaP.numChannels, Bool()))

    val vmemRead      = Valid(new VmemLineReadPort(vmemP))
    val vmemReadGrant = Input(Bool())
    val vmemReadData  = Flipped(Valid(UInt(beatBits.W)))
    val vmemWrite      = Valid(new VmemLineWritePort(vmemP))
    val vmemWriteGrant = Input(Bool())

    val tl = new TLBundle(tlBundleParams)
    val debugInFlight = Output(UInt((tagBits + 1).W))
  })

  val tlAdapter = Module(new TileLinkAdapter(beatBytes, tagBits, tlBundleParams))
  io.tl <> tlAdapter.io.tl

  val commandQueue = Reg(Vec(dmaP.numChannels, new DmaCommand(vmemP, dmaP)))
  val slotActive   = RegInit(VecInit(Seq.fill(dmaP.numChannels)(false.B)))
  val channelBusy  = RegInit(VecInit(Seq.fill(dmaP.numChannels)(false.B)))
  io.channelBusy := channelBusy

  val enqueueIdx  = RegInit(0.U(dmaP.channelIdBits.W))
  val requestIdx  = RegInit(0.U(dmaP.channelIdBits.W))
  val responseIdx = RegInit(0.U(dmaP.channelIdBits.W))

  val requestBeatCount  = RegInit(0.U((vmemP.lineAddrBits + 1).W))
  val responseBeatCount = RegInit(0.U((vmemP.lineAddrBits + 1).W))

  val requestBeatsNeeded  = commandQueue(requestIdx).transferSize >> log2Ceil(beatBytes).U
  val responseBeatsNeeded = commandQueue(responseIdx).transferSize >> log2Ceil(beatBytes).U

  val nextSourceId    = RegInit(0.U(tagBits.W))
  val currentInFlight = tlAdapter.io.inFlightCount
  val currentCmdIsStore = commandQueue(requestIdx).opType === StoreFromVmem

  val sourceIdTable = Reg(Vec(dmaP.maxInFlight, UInt(vmemP.lineAddrBits.W)))

  // ==========================================================================
  // VMEM read sequencer (STORE path) — stalls on bank denial
  // ==========================================================================

  val vmemReadBeatCount = RegInit(0.U(vmemP.lineAddrBits.W))
  val vmemReadComplete  = RegInit(false.B)
  val vmemReadLineAddr  = commandQueue(requestIdx).vmemLineAddr + vmemReadBeatCount

  io.vmemRead.valid          := slotActive(requestIdx) && currentCmdIsStore && !vmemReadComplete
  io.vmemRead.bits.bankIdx   := vmemP.getBankIdx(vmemReadLineAddr)
  io.vmemRead.bits.bankAddr  := vmemP.getBankAddr(vmemReadLineAddr)

  when(io.vmemRead.valid && io.vmemReadGrant) {
    vmemReadBeatCount := vmemReadBeatCount + 1.U
    when(vmemReadBeatCount === (requestBeatsNeeded - 1.U)) {
      vmemReadComplete := true.B
    }
  }

  val storeDataQueue = Module(new Queue(UInt(beatBits.W), entries = 128))
  storeDataQueue.io.enq.valid := io.vmemReadData.valid
  storeDataQueue.io.enq.bits  := io.vmemReadData.bits
  storeDataQueue.io.deq.ready := tlAdapter.io.request.fire && slotActive(requestIdx) && currentCmdIsStore

  // ==========================================================================
  // Command enqueueing
  // ==========================================================================

  when(io.command.valid) {
    commandQueue(enqueueIdx) := io.command.bits
    slotActive(enqueueIdx)   := true.B
    channelBusy(io.command.bits.channelId) := true.B
    enqueueIdx := enqueueIdx +% 1.U
  }

  // ==========================================================================
  // TileLink request issue
  // ==========================================================================

  tlAdapter.io.request.valid :=
    slotActive(requestIdx) &&
    (currentInFlight < dmaP.maxInFlight.U) &&
    Mux(currentCmdIsStore, storeDataQueue.io.deq.valid, true.B)

  tlAdapter.io.request.bits.address := commandQueue(requestIdx).dramAddress + (requestBeatCount * beatBytes.U)
  tlAdapter.io.request.bits.tag     := nextSourceId
  tlAdapter.io.request.bits.isStore := currentCmdIsStore
  tlAdapter.io.request.bits.data    := storeDataQueue.io.deq.bits
  tlAdapter.io.request.bits.mask    := Fill(beatBytes, 1.U(1.W))

  when(tlAdapter.io.request.fire) {
    sourceIdTable(nextSourceId) := commandQueue(requestIdx).vmemLineAddr + requestBeatCount
    requestBeatCount := requestBeatCount + 1.U
    nextSourceId     := nextSourceId +% 1.U
    when(requestBeatCount === (requestBeatsNeeded - 1.U)) {
      requestBeatCount  := 0.U
      vmemReadBeatCount := 0.U
      vmemReadComplete  := false.B
      requestIdx        := requestIdx +% 1.U
    }
  }

  // ==========================================================================
  // Response handling — backpressure on VMEM write denial
  // ==========================================================================

  val responseTag       = tlAdapter.io.response.bits.tag
  val respIsLoad        = commandQueue(responseIdx).opType === LoadToVmem
  val vmemWriteLineAddr = sourceIdTable(responseTag)

  io.vmemWrite.valid          := respIsLoad && tlAdapter.io.response.valid
  io.vmemWrite.bits.bankIdx   := vmemP.getBankIdx(vmemWriteLineAddr)
  io.vmemWrite.bits.bankAddr  := vmemP.getBankAddr(vmemWriteLineAddr)
  io.vmemWrite.bits.data      := tlAdapter.io.response.bits.data

  tlAdapter.io.response.ready := Mux(respIsLoad, io.vmemWriteGrant, true.B)

  when(tlAdapter.io.response.fire) {
    responseBeatCount := responseBeatCount + 1.U
    when(responseBeatCount === (responseBeatsNeeded - 1.U)) {
      channelBusy(commandQueue(responseIdx).channelId) := false.B
      slotActive(responseIdx) := false.B
      responseBeatCount       := 0.U
      responseIdx             := responseIdx +% 1.U
    }
  }

  io.debugInFlight := currentInFlight
}

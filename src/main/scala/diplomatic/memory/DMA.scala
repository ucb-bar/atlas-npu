// ============================================================================
// DMA.scala — Multi-channel DMA engine (block-banked VMEM architecture).
//
// VMEM ports use pre-decomposed bankIdx/bankAddr via VmemParams helpers.
// The default Atlas map gives each VMEM bank enough space for an aligned
// 1 KiB tensor transfer to stay within one bank for all 32 beats. Grant
// signals provide backpressure when LSU, another DMA side, or TileLink holds
// the selected access on that 1RW bank.
//
// Out-of-order safety:
//   - Per-source-ID metadata (direction, destination slot, VMEM line addr,
//     last-beat flag) is captured at A-fire time and consulted at D-fire.
//     This lets responses arrive in any order without cross-talk between
//     commands of differing direction.
//   - Per-slot outstanding-beat counters gate channelBusy deassertion;
//     a command is only marked complete when every one of its beats has
//     been retired, regardless of interleaving with later commands.
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
  val idInFlight = RegInit(0.U(numIds.W))

  val tagIsFree = !idInFlight(io.request.bits.tag)

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
  private val tagBits        = dmaP.tagBits
  private val numIds         = 1 << tagBits
  private val beatBytes      = dmaP.beatBytes
  private val beatBits       = beatBytes * 8
  private val numChannels    = dmaP.numChannels
  private val maxBeatsPerCmd = (1 << dmaP.transferSizeBits) / beatBytes
  private val beatCountBits  = log2Ceil(maxBeatsPerCmd + 1)

  require(
    dmaP.maxInFlight <= numIds,
    s"maxInFlight (${dmaP.maxInFlight}) must not exceed 2^tagBits ($numIds)"
  )

  val io = IO(new Bundle {
    val command     = Flipped(Valid(new DmaCommand(vmemP, dmaP)))
    val channelBusy = Output(Vec(numChannels, Bool()))

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

  // ==========================================================================
  // Command slots
  // ==========================================================================

  val commandQueue = Reg(Vec(numChannels, new DmaCommand(vmemP, dmaP)))
  val slotActive   = RegInit(VecInit(Seq.fill(numChannels)(false.B)))
  val channelBusy  = RegInit(VecInit(Seq.fill(numChannels)(false.B)))
  io.channelBusy := channelBusy

  // Per-slot outstanding-beat counter: incremented on A-fire for that slot,
  // decremented on D-fire for any source ID whose metadata points at it.
  // The slot is only retired when this count returns to zero AND all beats
  // have been issued (i.e., the command has been fully dispatched on A).
  val slotOutstanding = RegInit(VecInit(Seq.fill(numChannels)(0.U(beatCountBits.W))))
  val slotDispatched  = RegInit(VecInit(Seq.fill(numChannels)(false.B)))

  val enqueueIdx = RegInit(0.U(dmaP.channelIdBits.W))
  val requestIdx = RegInit(0.U(dmaP.channelIdBits.W))

  val requestBeatCount    = RegInit(0.U(beatCountBits.W))
  val requestBeatsNeeded  = commandQueue(requestIdx).transferSize >> log2Ceil(beatBytes).U

  val nextSourceId    = RegInit(0.U(tagBits.W))
  val currentInFlight = tlAdapter.io.inFlightCount
  val currentCmdIsStore = commandQueue(requestIdx).opType === StoreFromVmem

  // ==========================================================================
  // Per-source-ID metadata — the heart of OOO correctness.
  //
  // Captured at A-fire. On D-fire we consult this table (indexed by the
  // returning source ID) to decide: direction, destination slot, VMEM
  // line address, and whether this beat was the last for its command.
  // ==========================================================================

  class SourceMeta extends Bundle {
    val isLoad   = Bool()
    val slotIdx  = UInt(dmaP.channelIdBits.W)
    val lineAddr = UInt(vmemP.lineAddrBits.W)
    val isLast   = Bool()
  }
  val sourceMeta = Reg(Vec(numIds, new SourceMeta))

  // ==========================================================================
  // VMEM read sequencer (STORE path) — stalls on bank denial
  // ==========================================================================

  val vmemReadBeatCount = RegInit(0.U(beatCountBits.W))
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
    commandQueue(enqueueIdx)               := io.command.bits
    slotActive(enqueueIdx)                 := true.B
    slotDispatched(enqueueIdx)             := false.B
    channelBusy(io.command.bits.channelId) := true.B
    enqueueIdx := enqueueIdx +% 1.U
  }

  // ==========================================================================
  // TileLink request issue
  // ==========================================================================

  val isLastBeat = requestBeatCount === (requestBeatsNeeded - 1.U)

  tlAdapter.io.request.valid :=
    slotActive(requestIdx) && !slotDispatched(requestIdx) &&
    (currentInFlight < dmaP.maxInFlight.U) &&
    Mux(currentCmdIsStore, storeDataQueue.io.deq.valid, true.B)

  tlAdapter.io.request.bits.address := commandQueue(requestIdx).dramAddress + (requestBeatCount * beatBytes.U)
  tlAdapter.io.request.bits.tag     := nextSourceId
  tlAdapter.io.request.bits.isStore := currentCmdIsStore
  tlAdapter.io.request.bits.data    := storeDataQueue.io.deq.bits
  tlAdapter.io.request.bits.mask    := Fill(beatBytes, 1.U(1.W))

  // Track per-slot A-fire / D-fire deltas so we can retire correctly even
  // when A and D happen in the same cycle against the same slot.
  val slotAFireOH = Wire(Vec(numChannels, Bool()))
  val slotDFireOH = Wire(Vec(numChannels, Bool()))
  slotAFireOH.foreach(_ := false.B)
  slotDFireOH.foreach(_ := false.B)

  when(tlAdapter.io.request.fire) {
    // Capture metadata for this source ID.
    sourceMeta(nextSourceId).isLoad   := !currentCmdIsStore
    sourceMeta(nextSourceId).slotIdx  := requestIdx
    sourceMeta(nextSourceId).lineAddr := commandQueue(requestIdx).vmemLineAddr + requestBeatCount
    sourceMeta(nextSourceId).isLast   := isLastBeat

    slotAFireOH(requestIdx) := true.B

    requestBeatCount := requestBeatCount + 1.U
    nextSourceId     := nextSourceId +% 1.U

    when(isLastBeat) {
      requestBeatCount  := 0.U
      vmemReadBeatCount := 0.U
      vmemReadComplete  := false.B
      slotDispatched(requestIdx) := true.B
      requestIdx := requestIdx +% 1.U
    }
  }

  // ==========================================================================
  // Response handling — backpressure on VMEM write denial
  // ==========================================================================

  val responseTag  = tlAdapter.io.response.bits.tag
  val respMeta     = sourceMeta(responseTag)
  val respIsLoad   = respMeta.isLoad
  val respSlotIdx  = respMeta.slotIdx
  val respLineAddr = respMeta.lineAddr

  io.vmemWrite.valid          := respIsLoad && tlAdapter.io.response.valid
  io.vmemWrite.bits.bankIdx   := vmemP.getBankIdx(respLineAddr)
  io.vmemWrite.bits.bankAddr  := vmemP.getBankAddr(respLineAddr)
  io.vmemWrite.bits.data      := tlAdapter.io.response.bits.data

  // Loads need a VMEM write grant; stores (PutFullData ACKs) have no VMEM
  // side-effect so they retire unconditionally.
  tlAdapter.io.response.ready := Mux(respIsLoad, io.vmemWriteGrant, true.B)

  when(tlAdapter.io.response.fire) {
    slotDFireOH(respSlotIdx) := true.B
  }

  // Per-slot outstanding counter update (+1 on A-fire, −1 on D-fire).
  for (i <- 0 until numChannels) {
    val inc = slotAFireOH(i)
    val dec = slotDFireOH(i)
    when(inc && !dec) {
      slotOutstanding(i) := slotOutstanding(i) + 1.U
    }.elsewhen(!inc && dec) {
      slotOutstanding(i) := slotOutstanding(i) - 1.U
    }

    // Retirement: command fully dispatched, and no beats outstanding.
    // When dec fires and the counter is transitioning to zero, we also
    // retire in that same cycle.
    val willBeZero = Mux(inc && !dec, false.B,
                     Mux(!inc && dec, slotOutstanding(i) === 1.U,
                                      slotOutstanding(i) === 0.U))

    when(slotActive(i) && slotDispatched(i) && willBeZero) {
      slotActive(i)                              := false.B
      slotDispatched(i)                          := false.B
      channelBusy(commandQueue(i).channelId)     := false.B
    }
  }

  io.debugInFlight := currentInFlight
}

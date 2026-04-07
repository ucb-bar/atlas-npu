// ============================================================================
// DMA.scala — Multi-channel DMA engine for the Atlas accelerator.
//
// Transfers data between external DRAM (via TileLink) and on-chip VMEM
// (via dedicated line-granularity read/write ports).
//
// Data-path width matches DMA_ALIGN (default 32 bytes = 256 bits per beat).
//
// Key components:
//   • TileLinkAdapter  – Thin shim that maps generic load/store requests
//                        onto TileLink Channel-A/D transactions.
//   • DmaEngine        – Queues DMA commands across DMA_CHANNELS slots,
//                        sequences TileLink requests, and steers data
//                        between VMEM and DRAM.
// ============================================================================

package atlas.dma

import atlas.common._
import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

// ============================================================================
// Generic memory-request / memory-response bundles
// ============================================================================

/** A single load or store request to external memory.
  *
  * @param dataBytes  Width of the data bus in bytes (e.g. 32 for 256-bit).
  * @param tagBits    Width of the source-ID / tag field.
  */
class MemoryRequest(dataBytes: Int, tagBits: Int) extends Bundle {
  val address = UInt(64.W)              // Byte address in DRAM.
  val data    = UInt((dataBytes * 8).W) // Write payload (ignored for loads).
  val mask    = UInt(dataBytes.W)       // Per-byte write-enable mask.
  val tag     = UInt(tagBits.W)         // Caller-assigned ID echoed in resp.
  val isStore = Bool()                  // true → store, false → load.
}

/** Response returned for a completed memory transaction. */
class MemoryResponse(dataBytes: Int, tagBits: Int) extends Bundle {
  val data = UInt((dataBytes * 8).W)    // Read payload (undefined for stores).
  val tag  = UInt(tagBits.W)            // Echoed source ID.
}

// ============================================================================
// TileLinkAdapter — maps MemoryRequest/Response onto TileLink A/D channels
// ============================================================================

/** Adapter that converts a simple ready/valid request interface into TileLink
  * Get and PutFullData messages.  Responses are forwarded back with the
  * original tag so the caller can match them to outstanding requests.
  *
  * @param dataBytes       Beat width in bytes.
  * @param tagBits         Source-ID width (limits max in-flight requests).
  * @param tlBundleParams  TileLink bundle parameters for the attached port.
  */
class TileLinkAdapter(
    dataBytes:      Int,
    tagBits:        Int,
    tlBundleParams: TLBundleParameters
) extends Module {

  // Number of low-order address bits to mask for natural alignment.
  private val alignmentBits = log2Ceil(dataBytes)

  // ---------- IO ----------

  val io = IO(new Bundle {
    /** Incoming load/store requests. */
    val request  = Flipped(Decoupled(new MemoryRequest(dataBytes, tagBits)))
    /** Outgoing responses (one per completed TileLink D-channel beat). */
    val response = Valid(new MemoryResponse(dataBytes, tagBits))
    /** High when at least one request is still in flight. */
    val busy     = Output(Bool())
    /** Saturating count of outstanding (un-answered) requests. */
    val inFlightCount = Output(UInt(tagBits.W))
    /** Raw TileLink port to the system interconnect. */
    val tl = new TLBundle(tlBundleParams)
  })

  // ---------- In-flight tracking ----------

  val inFlightReg = RegInit(0.U(tagBits.W))

  when(io.tl.a.fire || io.tl.d.fire) {
    inFlightReg := inFlightReg + io.tl.a.fire - io.tl.d.fire
  }

  io.busy          := inFlightReg =/= 0.U
  io.inFlightCount := inFlightReg

  // ---------- Channel A (requests) ----------

  io.request.ready := io.tl.a.ready
  io.tl.a.valid    := io.request.valid

  // Force the address to be naturally aligned to the beat width.
  val alignedAddress = (io.request.bits.address >> alignmentBits.U) << alignmentBits.U

  io.tl.a.bits.opcode  := Mux(io.request.bits.isStore,
                               TLMessages.PutFullData,
                               TLMessages.Get)
  io.tl.a.bits.param   := 0.U
  io.tl.a.bits.size    := alignmentBits.U
  io.tl.a.bits.source  := io.request.bits.tag
  io.tl.a.bits.address := alignedAddress
  io.tl.a.bits.mask    := Mux(io.request.bits.isStore,
                               io.request.bits.mask,
                               Fill(dataBytes, 1.U(1.W)))
  io.tl.a.bits.data    := io.request.bits.data
  io.tl.a.bits.corrupt := false.B

  // ---------- Channel D (responses) ----------

  io.tl.d.ready        := true.B   // Always accept responses immediately.
  io.response.valid     := io.tl.d.valid
  io.response.bits.data := io.tl.d.bits.data
  io.response.bits.tag  := io.tl.d.bits.source
}

// ============================================================================
// DMA command definition
// ============================================================================

/** Direction of a DMA transfer. */
object DmaDirection extends ChiselEnum {
  val LoadToVmem    = Value  // DRAM → VMEM  (read from memory).
  val StoreFromVmem = Value  // VMEM → DRAM  (write to memory).
}

import DmaDirection._

/** A single DMA command describing one bulk transfer.
  *
  * @param vmemParams  VMEM geometry (provides `lineAddrBits`).
  * @param dmaParams   DMA engine parameters (provides `transferSizeBits`).
  */
class DmaCommand(vmemParams: VmemParams, dmaParams: DmaParams) extends Bundle {
  val opType       = DmaDirection()                       // Load or store.
  val channelId    = UInt(dmaParams.channelIdBits.W)      // Owning DMA channel.
  val vmemLineAddr = UInt(vmemParams.lineAddrBits.W)      // First VMEM line address.
  val dramAddress  = UInt(64.W)                           // Starting DRAM byte addr.
  val transferSize = UInt(dmaParams.transferSizeBits.W)   // Transfer length in bytes.
}

// ============================================================================
// DmaEngine — the main DMA controller
// ============================================================================

/** DMA engine that accepts bulk-transfer commands and sequences them into
  * individual TileLink beats, reading from or writing to VMEM as needed.
  *
  * Architecture overview (for a single transfer):
  *
  *   LOAD (DRAM → VMEM):
  *     1. Issue TileLink Get requests, one per VMEM line.
  *     2. On each response, write the returned data to the VMEM line whose
  *        address was recorded in `sourceIdTable` at issue time.
  *
  *   STORE (VMEM → DRAM):
  *     1. Pre-read VMEM lines into `storeDataQueue`.
  *     2. Issue TileLink PutFullData requests, attaching queued data.
  *     3. Count D-channel acknowledgements to know when done.
  *
  * Commands are queued in a circular buffer (`commandQueue`) and processed
  * strictly in order, one at a time, on both the request and response paths.
  *
  * @param params          Top-level Atlas parameters.
  * @param tlBundleParams  TileLink bundle parameters for the memory port.
  */
class DmaEngine(
    params:         AtlasParams,
    tlBundleParams: TLBundleParameters
) extends Module {

  // Shorthand for sub-parameter objects.
  private val dmaP  = params.dma
  private val vmemP = params.vmem

  private val tagBits   = dmaP.tagBits
  private val beatBytes = dmaP.beatBytes
  private val beatBits  = beatBytes * 8

  // ---------- IO ----------

  val io = IO(new Bundle {
    /** New DMA command from the control path. */
    val command     = Flipped(Valid(new DmaCommand(vmemP, dmaP)))
    /** Per-channel busy flags (high while that channel's transfer is active). */
    val channelBusy = Output(Vec(dmaP.numChannels, Bool()))

    // VMEM line-granularity access ports.
    val vmemRead     = Valid(new VmemLineReadPort(vmemP))
    val vmemReadData = Flipped(Valid(UInt(beatBits.W)))
    val vmemWrite    = Valid(new VmemLineWritePort(vmemP))

    /** TileLink port to the system interconnect. */
    val tl = new TLBundle(tlBundleParams)

    /** Debug: number of TileLink transactions currently in flight. */
    val debugInFlight = Output(UInt(tagBits.W))
  })

  // ==========================================================================
  // TileLink adapter instantiation
  // ==========================================================================

  val tlAdapter = Module(new TileLinkAdapter(beatBytes, tagBits, tlBundleParams))
  io.tl <> tlAdapter.io.tl

  // ==========================================================================
  // Command queue (circular buffer, one slot per DMA channel)
  // ==========================================================================

  /** Circular buffer holding pending DMA commands. */
  val commandQueue = Reg(Vec(dmaP.numChannels, new DmaCommand(vmemP, dmaP)))
  /** Slot-valid flags for the command queue. */
  val slotActive   = RegInit(VecInit(Seq.fill(dmaP.numChannels)(false.B)))
  /** Per-channel busy flags exposed to the outside world. */
  val channelBusy  = RegInit(VecInit(Seq.fill(dmaP.numChannels)(false.B)))
  io.channelBusy := channelBusy

  // Circular-buffer indices (wrap via +%).
  val enqueueIdx  = RegInit(0.U(dmaP.channelIdBits.W))  // Next free slot.
  val requestIdx  = RegInit(0.U(dmaP.channelIdBits.W))  // Slot being issued.
  val responseIdx = RegInit(0.U(dmaP.channelIdBits.W))  // Slot being retired.

  // ==========================================================================
  // Per-command beat counters
  // ==========================================================================

  /** Number of TileLink beats issued so far for the current request-side cmd. */
  val requestBeatCount  = RegInit(0.U((vmemP.lineAddrBits + 1).W))
  /** Number of TileLink responses received for the current response-side cmd. */
  val responseBeatCount = RegInit(0.U((vmemP.lineAddrBits + 1).W))

  // Total beats required = transferSize / beatBytes.
  val requestBeatsNeeded =
    commandQueue(requestIdx).transferSize >> log2Ceil(beatBytes).U
  val responseBeatsNeeded =
    commandQueue(responseIdx).transferSize >> log2Ceil(beatBytes).U

  // Rolling source-ID counter (wraps within tagBits width).
  val nextSourceId = RegInit(0.U(tagBits.W))

  // Current in-flight count from the TileLink adapter.
  val currentInFlight = tlAdapter.io.inFlightCount

  // Is the command at `requestIdx` a VMEM→DRAM store?
  val currentCmdIsStore =
    commandQueue(requestIdx).opType === StoreFromVmem

  // ==========================================================================
  // Source-ID → VMEM address lookup table
  // ==========================================================================

  /** Maps an in-flight source ID back to the VMEM line address that should
    * receive the load response.  Only meaningful for LoadToVmem commands.
    */
  val sourceIdTable = Reg(Vec(dmaP.maxInFlight, UInt(vmemP.lineAddrBits.W)))

  // ==========================================================================
  // VMEM read sequencer (used by STORE commands)
  // ==========================================================================
  //
  // Before we can issue PutFullData beats we must pre-read the corresponding
  // VMEM lines.  A small FSM walks through the lines and pushes the returned
  // data into `storeDataQueue`.  The queue decouples the VMEM read latency
  // from the TileLink request path.

  val vmemReadBeatCount = RegInit(0.U(vmemP.lineAddrBits.W))
  val vmemReadComplete  = RegInit(false.B)

  io.vmemRead.valid := slotActive(requestIdx) &&
                       currentCmdIsStore &&
                       !vmemReadComplete

  io.vmemRead.bits.addr := commandQueue(requestIdx).vmemLineAddr +
                           vmemReadBeatCount

  when(slotActive(requestIdx) && currentCmdIsStore && !vmemReadComplete) {
    vmemReadBeatCount := vmemReadBeatCount + 1.U
    when(vmemReadBeatCount === (requestBeatsNeeded - 1.U)) {
      vmemReadComplete := true.B
    }
  }

  /** FIFO holding VMEM data waiting to be sent as TileLink store payloads. */
  val storeDataQueue = Module(new Queue(UInt(beatBits.W), entries = 128))

  storeDataQueue.io.enq.valid := io.vmemReadData.valid
  storeDataQueue.io.enq.bits  := io.vmemReadData.bits

  // Dequeue one entry each time a store beat fires on TileLink.
  storeDataQueue.io.deq.ready := tlAdapter.io.request.fire &&
                                 slotActive(requestIdx) &&
                                 currentCmdIsStore

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
  // TileLink request issue (Channel A)
  // ==========================================================================
  //
  // We issue one beat per clock whenever:
  //   • The current slot is active,
  //   • The in-flight limit has not been reached,
  //   • For stores: the data queue has a payload ready.

  tlAdapter.io.request.valid :=
    slotActive(requestIdx) &&
    (currentInFlight < dmaP.maxInFlight.U) &&
    Mux(currentCmdIsStore, storeDataQueue.io.deq.valid, true.B)

  tlAdapter.io.request.bits.address := commandQueue(requestIdx).dramAddress +
                                       (requestBeatCount * beatBytes.U)
  tlAdapter.io.request.bits.tag     := nextSourceId
  tlAdapter.io.request.bits.isStore := currentCmdIsStore
  tlAdapter.io.request.bits.data    := storeDataQueue.io.deq.bits
  tlAdapter.io.request.bits.mask    := Fill(beatBytes, 1.U(1.W))

  when(tlAdapter.io.request.fire) {
    // Record which VMEM line this source ID maps to (needed for load writeback).
    sourceIdTable(nextSourceId) :=
      commandQueue(requestIdx).vmemLineAddr + requestBeatCount

    requestBeatCount := requestBeatCount + 1.U
    nextSourceId     := nextSourceId +% 1.U

    // If this was the last beat, advance to the next queued command.
    when(requestBeatCount === (requestBeatsNeeded - 1.U)) {
      requestBeatCount  := 0.U
      vmemReadBeatCount := 0.U
      vmemReadComplete  := false.B
      requestIdx        := requestIdx +% 1.U
    }
  }

  // ==========================================================================
  // Response handling (Channel D) — VMEM writeback for loads
  // ==========================================================================

  val responseTag = tlAdapter.io.response.bits.tag

  // For LoadToVmem: write the incoming data to the VMEM line recorded in
  // `sourceIdTable` when the request was originally issued.
  io.vmemWrite.valid :=
    (commandQueue(responseIdx).opType === LoadToVmem) &&
    tlAdapter.io.response.valid

  io.vmemWrite.bits.addr := sourceIdTable(responseTag)
  io.vmemWrite.bits.data := tlAdapter.io.response.bits.data

  // Count responses and retire the command once all beats are accounted for.
  when(tlAdapter.io.response.valid) {
    responseBeatCount := responseBeatCount + 1.U

    when(responseBeatCount === (responseBeatsNeeded - 1.U)) {
      channelBusy(commandQueue(responseIdx).channelId) := false.B
      slotActive(responseIdx) := false.B
      responseBeatCount       := 0.U
      responseIdx             := responseIdx +% 1.U
    }
  }

  // ==========================================================================
  // Debug output
  // ==========================================================================

  io.debugInFlight := currentInFlight
}

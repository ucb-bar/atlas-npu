/*
DMA.scala — 8-channel DMA engine.
Transfers data between external memory (via TileLink) and VMEM (via line ports).
Width matches system bus / VMEM line width (default 256 bits = 32 bytes).
*/

package atlas.dma

import atlas.common._
import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

class MemRequest(bytes: Int, tagBits: Int) extends Bundle {
  val addr    = UInt(64.W)
  val data    = UInt((bytes * 8).W)
  val mask    = UInt(bytes.W)
  val tag     = UInt(tagBits.W)
  val isStore = Bool()
}

class MemResponse(bytes: Int, tagBits: Int) extends Bundle {
  val data = UInt((bytes * 8).W)
  val tag  = UInt(tagBits.W)
}

class DMAInterface(
  widthBytes: Int, tagBits: Int, tlBundleParams: TLBundleParameters
) extends Module {
  private val offBits = log2Ceil(widthBytes)

  val io = IO(new Bundle {
    val req           = Flipped(Decoupled(new MemRequest(widthBytes, tagBits)))
    val resp          = Valid(new MemResponse(widthBytes, tagBits))
    val busy          = Output(Bool())
    val inflightCount = Output(UInt(tagBits.W))
    val tl            = new TLBundle(tlBundleParams)
  })

  val inflights = RegInit(0.U(tagBits.W))
  when(io.tl.a.fire || io.tl.d.fire) {
    inflights := inflights + io.tl.a.fire - io.tl.d.fire
  }
  io.busy          := inflights =/= 0.U
  io.inflightCount := inflights

  io.req.ready  := io.tl.a.ready
  io.tl.a.valid := io.req.valid
  val alignedAddr = (io.req.bits.addr >> offBits.U) << offBits.U
  io.tl.a.bits.opcode  := Mux(io.req.bits.isStore, TLMessages.PutFullData, TLMessages.Get)
  io.tl.a.bits.param   := 0.U
  io.tl.a.bits.size    := log2Ceil(widthBytes).U
  io.tl.a.bits.source  := io.req.bits.tag
  io.tl.a.bits.address := alignedAddr
  io.tl.a.bits.mask    := Mux(io.req.bits.isStore, io.req.bits.mask,
                               Fill(widthBytes, 1.U(1.W)))
  io.tl.a.bits.data    := io.req.bits.data
  io.tl.a.bits.corrupt := false.B

  io.tl.d.ready     := true.B
  io.resp.valid      := io.tl.d.valid
  io.resp.bits.data  := io.tl.d.bits.data
  io.resp.bits.tag   := io.tl.d.bits.source
}

// DMA instruction type
object dma_instruction_type extends ChiselEnum {
  val LOAD_TO_VMEM, STORE_FROM_VMEM = Value
}
import dma_instruction_type._

class dma_instruction(spP: ScratchpadParams, dmaP: DMAParams) extends Bundle {
  val type_of_instruction = dma_instruction_type()
  val channelId    = UInt(3.W)
  val vmemLineAddr = UInt(spP.lineAddrBits.W) // starting VMEM line address
  val address      = UInt(64.W)               // DRAM byte address
  val size         = UInt(dmaP.maxSizeBits.W) // transfer size in bytes
}

class DMASplitInterface(
  params: AtlasParams, tlBundleParams: TLBundleParameters
) extends Module {
  private val dma_p  = params.dma
  private val sp_p   = params.scratchpad
  private val tagBits    = dma_p.tagBits
  private val widthBytes = dma_p.widthBytes
  private val dataW      = widthBytes * 8

  val io = IO(new Bundle {
    val inst_received = Flipped(Valid(new dma_instruction(sp_p, dma_p)))
    val busy          = Output(Vec(dma_p.numInterfaces, Bool()))

    // VMEM line ports (replacing TRF ports)
    val vmemRead     = Valid(new ScratchpadLineReadPort(sp_p))
    val vmemReadData = Flipped(Valid(UInt(dataW.W)))
    val vmemWrite    = Valid(new ScratchpadLineWritePort(sp_p))

    val tl            = new TLBundle(tlBundleParams)
    val debug_inflight = Output(UInt(tagBits.W))
  })

  val tlClient = Module(new DMAInterface(widthBytes, tagBits, tlBundleParams))
  io.tl <> tlClient.io.tl

  val instructions_queue   = Reg(Vec(dma_p.numInterfaces, new dma_instruction(sp_p, dma_p)))
  val active_instructions  = RegInit(VecInit(Seq.fill(dma_p.numInterfaces)(false.B)))
  val empty_index          = RegInit(0.U(dma_p.numInterfaceBits.W))
  val inst_requesting_index = RegInit(0.U(dma_p.numInterfaceBits.W))
  val inst_responding_index = RegInit(0.U(dma_p.numInterfaceBits.W))
  val busy_slots = RegInit(VecInit(Seq.fill(dma_p.numInterfaces)(false.B)))
  io.busy := busy_slots

  val request_transaction_counter  = RegInit(0.U((sp_p.lineAddrBits + 1).W))
  val respond_transaction_counter  = RegInit(0.U((sp_p.lineAddrBits + 1).W))
  val sourceIdCounter              = RegInit(0.U(tagBits.W))
  val inflightCount = tlClient.io.inflightCount

  // Total transactions = size / widthBytes (each transaction moves one VMEM line)
  val total_request_transactions_needed =
    instructions_queue(inst_requesting_index).size >> log2Ceil(widthBytes).U
  val total_respond_transactions_needed =
    instructions_queue(inst_responding_index).size >> log2Ceil(widthBytes).U

  // Source ID table: maps in-flight tag → VMEM line address
  val sourceID_table = Reg(Vec(dma_p.maxInFlight, UInt(sp_p.lineAddrBits.W)))

  val is_current_inst_store =
    instructions_queue(inst_requesting_index).type_of_instruction === STORE_FROM_VMEM

  // ── VMEM read port (for STORE_FROM_VMEM: read VMEM → send to DRAM) ──
  val storeReadCounter = RegInit(0.U(sp_p.lineAddrBits.W))
  val storeReadDone    = RegInit(false.B)

  io.vmemRead.valid     := active_instructions(inst_requesting_index) &&
                           is_current_inst_store && !storeReadDone
  io.vmemRead.bits.addr := instructions_queue(inst_requesting_index).vmemLineAddr +
                           storeReadCounter

  when(active_instructions(inst_requesting_index) && is_current_inst_store && !storeReadDone) {
    storeReadCounter := storeReadCounter + 1.U
    when(storeReadCounter === (total_request_transactions_needed - 1.U)) {
      storeReadDone := true.B
    }
  }

  // Queue for store data read from VMEM
  val request_data_queue = Module(new Queue(UInt(dataW.W), 128))
  request_data_queue.io.enq.valid := io.vmemReadData.valid
  request_data_queue.io.enq.bits  := io.vmemReadData.bits
  request_data_queue.io.deq.ready := tlClient.io.req.fire &&
    active_instructions(inst_requesting_index) && is_current_inst_store

  // ── Instruction enqueueing ───────────────────────────────────────
  when(io.inst_received.valid) {
    instructions_queue(empty_index)  := io.inst_received.bits
    active_instructions(empty_index) := true.B
    empty_index                      := empty_index +% 1.U
    busy_slots(io.inst_received.bits.channelId) := true.B
  }

  // ── TileLink request issue ───────────────────────────────────────
  tlClient.io.req.valid := active_instructions(inst_requesting_index) &&
    (inflightCount < dma_p.maxInFlight.U) &&
    Mux(is_current_inst_store, request_data_queue.io.deq.valid, true.B)
  tlClient.io.req.bits.addr    := instructions_queue(inst_requesting_index).address +
                                  (request_transaction_counter * widthBytes.U)
  tlClient.io.req.bits.tag     := sourceIdCounter
  tlClient.io.req.bits.isStore := is_current_inst_store
  tlClient.io.req.bits.data    := request_data_queue.io.deq.bits
  tlClient.io.req.bits.mask    := Fill(widthBytes, 1.U(1.W))

  when(tlClient.io.req.fire) {
    // Record VMEM destination for this source ID (for loads)
    sourceID_table(sourceIdCounter) :=
      instructions_queue(inst_requesting_index).vmemLineAddr + request_transaction_counter
    request_transaction_counter := request_transaction_counter + 1.U
    sourceIdCounter             := sourceIdCounter +% 1.U
    when(request_transaction_counter === (total_request_transactions_needed - 1.U)) {
      request_transaction_counter := 0.U
      storeReadCounter := 0.U
      storeReadDone := false.B
      inst_requesting_index := inst_requesting_index +% 1.U
    }
  }

  // ── Response handling (LOAD: write to VMEM) ──────────────────────
  val respTag = tlClient.io.resp.bits.tag

  io.vmemWrite.valid     := (instructions_queue(inst_responding_index).type_of_instruction === LOAD_TO_VMEM) &&
                            tlClient.io.resp.valid
  io.vmemWrite.bits.addr := sourceID_table(respTag)
  io.vmemWrite.bits.data := tlClient.io.resp.bits.data

  when(tlClient.io.resp.valid) {
    respond_transaction_counter := respond_transaction_counter + 1.U
    when(respond_transaction_counter === (total_respond_transactions_needed - 1.U)) {
      busy_slots(instructions_queue(inst_responding_index).channelId) := false.B
      active_instructions(inst_responding_index) := false.B
      respond_transaction_counter := 0.U
      inst_responding_index := inst_responding_index +% 1.U
    }
  }

  io.debug_inflight := inflightCount
}

// ============================================================================
// VMEM.scala — Vector Memory Scratchpad with TileLink slave port.
//
// Single-ported SyncReadMem storing full lines (256 bits = 32 bytes each).
// Three clients — LSU, DMA, and TileLink (system bus) — share the single
// read port and single write port through priority arbitration:
//
//   Read priority:  LSU > DMA > TL
//   Write priority: LSU > DMA > TL
//
// The TL port allows the host to directly read/write VMEM over the system
// bus.  It is lowest priority so it never blocks accelerator datapath.
// ============================================================================

package atlas.vmem

import chisel3._
import chisel3.util._
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}
import atlas.common.{VmemParams, VmemLineReadPort, VmemLineWritePort}

/** On-chip vector scratchpad memory.
  *
  * @param p       VMEM geometry parameters (line width, capacity, addressing).
  * @param bundle  TileLink bundle parameters for the slave port.
  */
class VMEM(p: VmemParams, bundle: TLBundleParameters) extends Module {

  val io = IO(new Bundle {
    // ── TileLink slave port ──
    val tl = Flipped(new TLBundle(bundle))

    // ── DMA access ports ──
    /** DMA line-read request. */
    val dmaRead     = Flipped(Valid(new VmemLineReadPort(p)))
    /** DMA line-read response (valid one cycle after request). */
    val dmaReadData = Valid(UInt(p.lineWidthBits.W))
    /** DMA line-write request (full-line, no masking). */
    val dmaWrite    = Flipped(Valid(new VmemLineWritePort(p)))

    // ── LSU access ports ──
    /** LSU line-read request (higher priority than DMA). */
    val lsuRead     = Flipped(Valid(new VmemLineReadPort(p)))
    /** LSU line-read response (valid one cycle after request). */
    val lsuReadData = Valid(UInt(p.lineWidthBits.W))
    /** LSU line-write request (higher priority than DMA). */
    val lsuWrite    = Flipped(Valid(new VmemLineWritePort(p)))
  })

  // ==========================================================================
  // Storage
  // ==========================================================================

  val mem = SyncReadMem(p.numLines, UInt(p.lineWidthBits.W))

  val tl = io.tl

  // ==========================================================================
  // TileLink backpressure — accept A only when no LSU/DMA access
  // ==========================================================================

  tl.a.ready := !io.lsuRead.valid && !io.lsuWrite.valid &&
                !io.dmaRead.valid && !io.dmaWrite.valid

  // ==========================================================================
  // TL address conversion
  // ==========================================================================

  val tlAddr = tl.a.bits.address(p.byteAddrBits - 1, log2Ceil(p.lineBytes))

  // ==========================================================================
  // Read arbitration (LSU > DMA > TL)
  // ==========================================================================
  //
  // Only one read can be serviced per cycle.

  val readAddr = Wire(UInt(p.lineAddrBits.W))
  val readEn   = Wire(Bool())
  val readSel  = Wire(UInt(2.W))  // 0=None, 1=LSU, 2=DMA, 3=TL

  val tlReadValid = tl.a.valid && tl.a.bits.opcode === TLMessages.Get && tl.a.ready

  when(io.lsuRead.valid) {
    readAddr := io.lsuRead.bits.addr
    readEn   := true.B
    readSel  := 1.U
  }.elsewhen(io.dmaRead.valid) {
    readAddr := io.dmaRead.bits.addr
    readEn   := true.B
    readSel  := 2.U
  }.elsewhen(tlReadValid) {
    readAddr := tlAddr
    readEn   := true.B
    readSel  := 3.U
  }.otherwise {
    readAddr := 0.U
    readEn   := false.B
    readSel  := 0.U
  }

  // SyncReadMem read — data available on the next cycle.
  val rdData       = mem.read(readAddr, readEn)
  val r1_readSel   = RegNext(readSel)
  val r1_readEn    = RegNext(readEn, false.B)
  val r1_tlRdValid = RegNext(tlReadValid && readSel === 3.U, false.B)

  val r1_tlSource = RegNext(tl.a.bits.source)
  val r1_tlSize   = RegNext(tl.a.bits.size)

  // Route the response to the correct client.
  io.lsuReadData.valid := r1_readEn && r1_readSel === 1.U
  io.lsuReadData.bits  := rdData

  io.dmaReadData.valid := r1_readEn && r1_readSel === 2.U
  io.dmaReadData.bits  := rdData

  // ==========================================================================
  // Write arbitration (LSU > DMA > TL)
  // ==========================================================================

  val tlWriteValid = tl.a.valid && tl.a.ready &&
    (tl.a.bits.opcode === TLMessages.PutFullData ||
     tl.a.bits.opcode === TLMessages.PutPartialData)
  val r1_tlWrValid = RegNext(tlWriteValid, false.B)

  when(io.lsuWrite.valid) {
    mem.write(io.lsuWrite.bits.addr, io.lsuWrite.bits.data)
  }.elsewhen(io.dmaWrite.valid) {
    mem.write(io.dmaWrite.bits.addr, io.dmaWrite.bits.data)
  }.elsewhen(tlWriteValid) {
    mem.write(tlAddr, tl.a.bits.data)
  }

  // ==========================================================================
  // TileLink D channel response
  // ==========================================================================

  tl.d.valid        := r1_tlRdValid || r1_tlWrValid
  tl.d.bits.opcode  := Mux(r1_tlRdValid, TLMessages.AccessAckData, TLMessages.AccessAck)
  tl.d.bits.param   := 0.U
  tl.d.bits.size    := r1_tlSize
  tl.d.bits.source  := r1_tlSource
  tl.d.bits.sink    := 0.U
  tl.d.bits.denied  := false.B
  tl.d.bits.data    := Mux(r1_tlRdValid, rdData, 0.U)
  tl.d.bits.corrupt := false.B
}

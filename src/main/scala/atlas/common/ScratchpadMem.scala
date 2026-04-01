/*
ScratchpadMem.scala — VMEM (Vector Memory / Scratchpad).

Storage: SyncReadMem with UInt(lineWidth.W) elements — no Vec, no masks.
All writes are full-line (256-bit). The LSU performs read-modify-write
for scalar byte/half/word stores.

Read arbitration:  LSU > DMA
Write arbitration: LSU > DMA
*/

package atlas.common

import chisel3._
import chisel3.util._

// ── Port bundles ─────────────────────────────────────────────────────

class ScratchpadLineReadPort(p: ScratchpadParams) extends Bundle {
  val addr = UInt(p.lineAddrBits.W)
}

class ScratchpadLineWritePort(p: ScratchpadParams) extends Bundle {
  val addr = UInt(p.lineAddrBits.W)
  val data = UInt(p.lineWidth.W)
}

// ── ScratchpadMem ────────────────────────────────────────────────────

class ScratchpadMem(p: ScratchpadParams) extends Module {

  val io = IO(new Bundle {
    val dmaRead     = Flipped(Valid(new ScratchpadLineReadPort(p)))
    val dmaReadData = Valid(UInt(p.lineWidth.W))
    val dmaWrite    = Flipped(Valid(new ScratchpadLineWritePort(p)))

    val lsuRead     = Flipped(Valid(new ScratchpadLineReadPort(p)))
    val lsuReadData = Valid(UInt(p.lineWidth.W))
    val lsuWrite    = Flipped(Valid(new ScratchpadLineWritePort(p)))
  })

  val mem = SyncReadMem(p.numLines, UInt(p.lineWidth.W))

  // ── Read arbitration (LSU > DMA) ─────────────────────────────────
  val readAddr = Wire(UInt(p.lineAddrBits.W))
  val readEn   = Wire(Bool())
  val readSel  = Wire(Bool())   // false=LSU, true=DMA

  when(io.lsuRead.valid) {
    readAddr := io.lsuRead.bits.addr
    readEn   := true.B
    readSel  := false.B
  }.elsewhen(io.dmaRead.valid) {
    readAddr := io.dmaRead.bits.addr
    readEn   := true.B
    readSel  := true.B
  }.otherwise {
    readAddr := 0.U
    readEn   := false.B
    readSel  := false.B
  }

  val rdData     = mem.read(readAddr, readEn)
  val r1_readSel = RegNext(readSel)
  val r1_readEn  = RegNext(readEn, false.B)

  io.lsuReadData.valid := r1_readEn && !r1_readSel
  io.lsuReadData.bits  := rdData

  io.dmaReadData.valid := r1_readEn && r1_readSel
  io.dmaReadData.bits  := rdData

  // ── Write arbitration (LSU > DMA) ───────────────────────────────
  when(io.lsuWrite.valid) {
    mem.write(io.lsuWrite.bits.addr, io.lsuWrite.bits.data)
  }.elsewhen(io.dmaWrite.valid) {
    mem.write(io.dmaWrite.bits.addr, io.dmaWrite.bits.data)
  }
}

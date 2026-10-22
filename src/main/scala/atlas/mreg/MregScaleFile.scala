// ============================================================================
// MregScaleFile.scala — NVFP4 block-scale sidecar of the tensor register file.
//
// Each m-register gets 32 rows x 4 bytes: row r, byte g is the UE4M3 scale of
// row r's elements 16g..16g+15 (one m-register row holds 64 E2M1 elements).
// In VMEM a register's scales are 128 B = 4 lines; line l holds rows 8l..8l+7,
// row r at bytes 4*(r % 8) .. 4*(r % 8) + 3.
//
// Ports (all reads have the MregFile's one-cycle latency):
//   lineWrite   LSU vload.scale, one 32-byte line per cycle
//   lineRead    LSU vstore.scale, one line per cycle
//   rowRead(i)  NVFP4 MXU read port i, one row (32 bits) per cycle, in step
//               with the MregFile read of the same register and row
//
// Each reader has its own 1R1W copy of the array, written together, so every
// copy maps to plain block or distributed RAM.
// ============================================================================

package atlas.mreg

import chisel3._
import chisel3.util._
import atlas.common.{MregParams, MregReadReq}

class ScaleLineReq(p: MregParams) extends Bundle {
  val mregId = UInt(p.mregIdBits.W)
  val line   = UInt(log2Ceil(p.mregRows / 8).W)
}

class ScaleLineWrite(p: MregParams) extends ScaleLineReq(p) {
  val data = UInt(256.W)
}

class MregScaleFile(p: MregParams, rowReaders: Int = 2) extends Module {
  require(p.mregRows % 8 == 0)
  private val linesPerReg = p.mregRows / 8
  private val entries     = p.numMreg * linesPerReg

  val io = IO(new Bundle {
    val lineWrite = Flipped(Valid(new ScaleLineWrite(p)))
    val lineRead  = Flipped(Valid(new ScaleLineReq(p)))
    val lineResp  = Output(UInt(256.W))
    val rowRead   = Vec(rowReaders, Flipped(Valid(new MregReadReq(p))))
    val rowResp   = Output(Vec(rowReaders, UInt(32.W)))
  })

  private def index(mregId: UInt, line: UInt): UInt = Cat(mregId, line)

  val copies = Seq.fill(rowReaders + 1)(SyncReadMem(entries, UInt(256.W)))
  for (m <- copies) {
    when(io.lineWrite.valid) { m.write(index(io.lineWrite.bits.mregId, io.lineWrite.bits.line), io.lineWrite.bits.data) }
  }

  io.lineResp := copies(0).read(index(io.lineRead.bits.mregId, io.lineRead.bits.line), io.lineRead.valid)

  for (i <- 0 until rowReaders) {
    val req  = io.rowRead(i)
    val line = copies(i + 1).read(index(req.bits.mregId, req.bits.row >> 3), req.valid)
    val word = RegEnable(req.bits.row(2, 0), req.valid)
    io.rowResp(i) := VecInit((0 until 8).map(w => line(32 * w + 31, 32 * w)))(word)
  }
}

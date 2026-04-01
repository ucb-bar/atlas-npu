/*
TensorRegFile.scala — 64-bank tensor register file.
Each bank: 32 rows × 256 bits (32 × 32 fp8 values per register).
*/
package atlas.common

import chisel3._
import chisel3.util._

class RegFileReadInput(p: RegFileParams)  extends Bundle {
  val whichBank = UInt(p.NUM_BANKS_BITS.W)
  val rRow= UInt(p.SRAM_DEPTH_BITS.W)
}

class RegFileWriteInput(p: RegFileParams)  extends Bundle {
  val whichBank = UInt(p.NUM_BANKS_BITS.W)
  val wRow= UInt(p.SRAM_DEPTH_BITS.W)
  val wData = UInt(p.SRAM_WIDTH.W)
}

class TensorRegFile(p: RegFileParams) extends Module {
  val io = IO(new Bundle {
    // Read ports (8 total)
    val mxu0readinput0  = Flipped(Valid(new RegFileReadInput(p)))
    val mxu0readinput1  = Flipped(Valid(new RegFileReadInput(p)))
    val mxu1readinput0  = Flipped(Valid(new RegFileReadInput(p)))
    val mxu1readinput1  = Flipped(Valid(new RegFileReadInput(p)))
    val vpureadinput0   = Flipped(Valid(new RegFileReadInput(p)))
    val vpureadinput1   = Flipped(Valid(new RegFileReadInput(p)))
    val dmareadinput    = Flipped(Valid(new RegFileReadInput(p)))
    val xlureadinput    = Flipped(Valid(new RegFileReadInput(p)))

    // Write ports (8 total — added mxu0writeinput1, mxu1writeinput1)
    val mxu0writeinput0 = Flipped(Valid(new RegFileWriteInput(p)))
    val mxu0writeinput1 = Flipped(Valid(new RegFileWriteInput(p)))
    val mxu1writeinput0 = Flipped(Valid(new RegFileWriteInput(p)))
    val mxu1writeinput1 = Flipped(Valid(new RegFileWriteInput(p)))
    val vpuwriteinput0  = Flipped(Valid(new RegFileWriteInput(p)))
    val vpuwriteinput1  = Flipped(Valid(new RegFileWriteInput(p)))
    val dmawriteinput   = Flipped(Valid(new RegFileWriteInput(p)))
    val xluwriteinput   = Flipped(Valid(new RegFileWriteInput(p)))

    // Read outputs
    val mxu0readoutput0 = Valid(UInt(p.SRAM_WIDTH.W))
    val mxu0readoutput1 = Valid(UInt(p.SRAM_WIDTH.W))
    val mxu1readoutput0 = Valid(UInt(p.SRAM_WIDTH.W))
    val mxu1readoutput1 = Valid(UInt(p.SRAM_WIDTH.W))
    val vpureadoutput0  = Valid(UInt(p.SRAM_WIDTH.W))
    val vpureadoutput1  = Valid(UInt(p.SRAM_WIDTH.W))
    val dmareadoutput   = Valid(UInt(p.SRAM_WIDTH.W))
    val xlureadoutput   = Valid(UInt(p.SRAM_WIDTH.W))
  })

  val sramBanks = Seq.fill(p.NUM_BANKS)(SyncReadMem(p.SRAM_DEPTH, UInt(p.SRAM_WIDTH.W)))

  val readPorts = Seq(
    (io.mxu0readinput0, io.mxu0readoutput0),
    (io.mxu0readinput1, io.mxu0readoutput1),
    (io.mxu1readinput0, io.mxu1readoutput0),
    (io.mxu1readinput1, io.mxu1readoutput1),
    (io.vpureadinput0,  io.vpureadoutput0),
    (io.vpureadinput1,  io.vpureadoutput1),
    (io.dmareadinput,   io.dmareadoutput),
    (io.xlureadinput,   io.xlureadoutput)
  )

  for ((readInput, readOut) <- readPorts) {
    val sramBanksOutputs = Wire(Vec(p.NUM_BANKS, UInt(p.SRAM_WIDTH.W)))
    for (i <- 0 until p.NUM_BANKS) {
      sramBanksOutputs(i) := sramBanks(i).read(readInput.bits.rRow,
        readInput.valid && readInput.bits.whichBank === i.U)
    }
    readOut.valid := RegNext(readInput.valid, false.B)
    readOut.bits  := sramBanksOutputs(RegNext(readInput.bits.whichBank))
  }

  val writePorts = Seq(
    io.mxu0writeinput0, io.mxu0writeinput1,
    io.mxu1writeinput0, io.mxu1writeinput1,
    io.vpuwriteinput0,  io.vpuwriteinput1,
    io.dmawriteinput,   io.xluwriteinput
  )

  for (writeIn <- writePorts) {
    when(writeIn.valid) {
      for (i <- 0 until p.NUM_BANKS) {
        when(writeIn.bits.whichBank === i.U) {
          sramBanks(i).write(writeIn.bits.wRow, writeIn.bits.wData)
        }
      }
    }
  }
}

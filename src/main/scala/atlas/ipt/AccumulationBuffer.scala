package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.InnerProductTreeParams
import sp26FPUnits.{BF16ScaleToE4M3, E4M3ToBF16, BF16ToE4M3WithE8M0}

class AccumulationBuffer(p: InnerProductTreeParams) extends Module {
  val io = IO(new Bundle {
    val mxuWrite    = Flipped(Valid(new AccBufMxu0Write(p)))
    val mxuReadReq  = Input(new AccBufMxu0Read(p))
    val mxuReadData = Output(Vec(p.numLanes, UInt(p.psumFmt.ieeeWidth.W)))
    val loadFP8     = Flipped(Valid(new AccBufLoadFP8(p)))
    val loadBF16    = Flipped(Valid(new AccBufLoadBF16(p)))
    val storeReq     = Input(new AccBufStoreReq(p))
    val storeBF16Out = Output(Vec(p.numLanes, UInt(16.W)))
    val storeFP8Out  = Output(Vec(p.numLanes, UInt(8.W)))
  })

  val buf0 = Mem(p.tileRows, Vec(p.numLanes, UInt(16.W)))
  val buf1 = Mem(p.tileRows, Vec(p.numLanes, UInt(16.W)))

  val deq   = Seq.fill(p.numLanes)(Module(new E4M3ToBF16))
  val quant = Seq.fill(p.numLanes)(Module(new BF16ToE4M3WithE8M0))

  // Unconditional defaults
  for (i <- 0 until p.numLanes) {
    deq(i).io.e4m3In := io.loadFP8.bits.data(i)
  }

  // MXU compute write
  when(io.mxuWrite.valid) {
    when(io.mxuWrite.bits.accSel) {
      buf1.write(io.mxuWrite.bits.rowIdx, io.mxuWrite.bits.data)
    }.otherwise {
      buf0.write(io.mxuWrite.bits.rowIdx, io.mxuWrite.bits.data)
    }
  }

  // MXU compute read (combinational)
  val readRow0 = buf0.read(io.mxuReadReq.rowIdx)
  val readRow1 = buf1.read(io.mxuReadReq.rowIdx)
  io.mxuReadData := Mux(io.mxuReadReq.accSel, readRow1, readRow0)

  // Load FP8: dequantize and write
  when(io.loadFP8.valid) {
    val bf16Row = Wire(Vec(p.numLanes, UInt(16.W)))
    for (i <- 0 until p.numLanes) {
      bf16Row(i) := deq(i).io.bf16Out
    }
    when(io.loadFP8.bits.accSel) {
      buf1.write(io.loadFP8.bits.rowIdx, bf16Row)
    }.otherwise {
      buf0.write(io.loadFP8.bits.rowIdx, bf16Row)
    }
  }

  // Load BF16: direct write
  when(io.loadBF16.valid) {
    when(io.loadBF16.bits.accSel) {
      buf1.write(io.loadBF16.bits.rowIdx, io.loadBF16.bits.data)
    }.otherwise {
      buf0.write(io.loadBF16.bits.rowIdx, io.loadBF16.bits.data)
    }
  }

  // Store BF16: combinational read
  val storeRow0 = buf0.read(io.storeReq.rowIdx)
  val storeRow1 = buf1.read(io.storeReq.rowIdx)
  val storeRow  = Mux(io.storeReq.accSel, storeRow1, storeRow0)
  io.storeBF16Out := storeRow

  // Store FP8: quantize with E8M0 scale passed directly
  for (i <- 0 until p.numLanes) {
    quant(i).io.bf16In    := storeRow(i)
    quant(i).io.scaleE8M0 := io.storeReq.scaleE8M0
    io.storeFP8Out(i)     := quant(i).io.e4m3Out
  }
}

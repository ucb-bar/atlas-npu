/*
InnerProductTreesSequencer.scala: load + compute sequencer for one IPT MXU.

Read data appears 1 cycle after address issue.

Data FSM (push ops): dIdle → dSetup → dSteady → dIdle
  - dIdle: issue first TRF read
  - dSetup: first data arrives, process row 0, issue read for row 1
  - dSteady: data arrives each cycle, process and pipeline next read
  Pop ops use dPop (combinational acc buf read, no SyncReadMem latency).

Compute FSM: cIdle → cSetup → cActive → cDrain → cIdle
  Same pipeline pattern for activation reads.
*/

package atlas.ipt

import chisel3._
import chisel3.util._
import atlas.common.{InnerProductTreeParams, RegFileParams, RegFileReadInput, RegFileWriteInput}

class InnerProductTreesSequencer(
  p: InnerProductTreeParams = InnerProductTreeParams(),
  rfP: RegFileParams = RegFileParams()
) extends Module {

  require(p.vecLen * p.inputFmt.ieeeWidth == rfP.SRAM_WIDTH)
  require(p.tileRows <= rfP.SRAM_DEPTH)

  val io = IO(new Bundle {
    val cmd             = Flipped(Decoupled(new Mxu0Cmd(rfP)))
    val trfReadPort0In  = Valid(new RegFileReadInput(rfP))
    val trfReadPort0Out = Input(Valid(UInt(rfP.SRAM_WIDTH.W)))
    val trfReadPort1In  = Valid(new RegFileReadInput(rfP))
    val trfReadPort1Out = Input(Valid(UInt(rfP.SRAM_WIDTH.W)))
    val trfWritePort0   = Valid(new RegFileWriteInput(rfP))
    val trfWritePort1   = Valid(new RegFileWriteInput(rfP))
    val dataBusy        = Output(Bool())
    val computeBusy     = Output(Bool())
  })

  private def unpackRow(x: UInt, elemWidth: Int, n: Int): Vec[UInt] =
    VecInit((0 until n).map(i => x((i + 1) * elemWidth - 1, i * elemWidth)))
  private def packElems(x: Seq[UInt]): UInt = Cat(x.reverse)

  private val tileRows  = p.tileRows
  private val rowBits   = p.tileRowBits
  private val rowCountW = log2Ceil(tileRows + 1)

  // ── Submodules ──
  val core   = Module(new InnerProductTrees(p))
  val accBuf = Module(new AccumulationBuffer(p))

  // ── Defaults ──
  core.io.weightWrite.valid := false.B
  core.io.weightWrite.bits  := 0.U.asTypeOf(new WeightWriteReq(p))
  core.io.compute.valid     := false.B
  core.io.compute.bits      := 0.U.asTypeOf(new ComputeReq(p))
  accBuf.io.mxuWrite.valid  := false.B
  accBuf.io.mxuWrite.bits   := 0.U.asTypeOf(new AccBufMxu0Write(p))
  accBuf.io.mxuReadReq      := 0.U.asTypeOf(new AccBufMxu0Read(p))
  accBuf.io.loadFP8.valid   := false.B
  accBuf.io.loadFP8.bits    := 0.U.asTypeOf(new AccBufLoadFP8(p))
  accBuf.io.loadBF16.valid  := false.B
  accBuf.io.loadBF16.bits   := 0.U.asTypeOf(new AccBufLoadBF16(p))
  accBuf.io.storeReq        := 0.U.asTypeOf(new AccBufStoreReq(p))
  io.trfReadPort0In.valid   := false.B
  io.trfReadPort0In.bits    := 0.U.asTypeOf(new RegFileReadInput(rfP))
  io.trfReadPort1In.valid   := false.B
  io.trfReadPort1In.bits    := 0.U.asTypeOf(new RegFileReadInput(rfP))
  io.trfWritePort0.valid    := false.B
  io.trfWritePort0.bits     := 0.U.asTypeOf(new RegFileWriteInput(rfP))
  io.trfWritePort1.valid    := false.B
  io.trfWritePort1.bits     := 0.U.asTypeOf(new RegFileWriteInput(rfP))

  // ── Command classification ──
  val isDataOp = io.cmd.valid && (
    io.cmd.bits.op === Mxu0Op.PushWeight  || io.cmd.bits.op === Mxu0Op.PushAccFP8  ||
    io.cmd.bits.op === Mxu0Op.PushAccBF16 || io.cmd.bits.op === Mxu0Op.PopAccFP8   ||
    io.cmd.bits.op === Mxu0Op.PopAccBF16)
  val isComputeOp = io.cmd.valid && (
    io.cmd.bits.op === Mxu0Op.Matmul || io.cmd.bits.op === Mxu0Op.MatmulAcc)

  //  DATA FSM
  val dIdle :: dSetup :: dSteady :: dPop :: Nil = Enum(4)
  val dataState   = RegInit(dIdle)
  val dataCmd     = Reg(new Mxu0Cmd(rfP))
  val dataRow     = Reg(UInt(rowCountW.W))    // row currently being processed
  val dataNextRow = Reg(UInt(rowCountW.W))    // next row to issue TRF read for

  io.dataBusy := (dataState =/= dIdle)
  val dataReady = (dataState === dIdle) && isDataOp

  switch(dataState) {
    is(dIdle) {
      when(dataReady && io.cmd.fire) {
        dataCmd     := io.cmd.bits
        dataRow     := 0.U
        dataNextRow := 1.U

        val isPush = io.cmd.bits.op === Mxu0Op.PushWeight ||
                     io.cmd.bits.op === Mxu0Op.PushAccFP8 ||
                     io.cmd.bits.op === Mxu0Op.PushAccBF16
        when(isPush) {
          // Issue first TRF read → data arrives in dSetup
          io.trfReadPort0In.valid         := true.B
          io.trfReadPort0In.bits.whichBank := io.cmd.bits.trfBank
          io.trfReadPort0In.bits.rRow     := 0.U
          when(io.cmd.bits.op === Mxu0Op.PushAccBF16) {
            io.trfReadPort1In.valid         := true.B
            io.trfReadPort1In.bits.whichBank := io.cmd.bits.trfBank + 1.U
            io.trfReadPort1In.bits.rRow     := 0.U
          }
          dataState := dSetup
        }.otherwise {
          dataState := dPop
        }
      }
    }

    // ── dSetup: first TRF data arrives. Process row 0, pipeline row 1. ──
    is(dSetup) {
      // Process arrived data
      when(dataCmd.op === Mxu0Op.PushWeight) {
        val rowData = unpackRow(io.trfReadPort0Out.bits, p.inputFmt.ieeeWidth, p.vecLen)
        core.io.weightWrite.valid         := true.B
        core.io.weightWrite.bits.targetBuf := dataCmd.weightSlot
        core.io.weightWrite.bits.laneIdx  := dataRow(log2Ceil(p.numLanes) - 1, 0)
        core.io.weightWrite.bits.weights  := rowData
      }.elsewhen(dataCmd.op === Mxu0Op.PushAccFP8) {
        val rowData = unpackRow(io.trfReadPort0Out.bits, 8, p.numLanes)
        accBuf.io.loadFP8.valid       := true.B
        accBuf.io.loadFP8.bits.accSel := dataCmd.accSel
        accBuf.io.loadFP8.bits.rowIdx := dataRow(rowBits - 1, 0)
        accBuf.io.loadFP8.bits.data   := rowData
      }.elsewhen(dataCmd.op === Mxu0Op.PushAccBF16) {
        val lo = io.trfReadPort0Out.bits
        val hi = io.trfReadPort1Out.bits
        val bf16Row = Wire(Vec(p.numLanes, UInt(16.W)))
        for (i <- 0 until p.numLanes) {
          if (i < 16) bf16Row(i) := lo((i+1)*16-1, i*16)
          else        bf16Row(i) := hi((i-16+1)*16-1, (i-16)*16)
        }
        accBuf.io.loadBF16.valid       := true.B
        accBuf.io.loadBF16.bits.accSel := dataCmd.accSel
        accBuf.io.loadBF16.bits.rowIdx := dataRow(rowBits - 1, 0)
        accBuf.io.loadBF16.bits.data   := bf16Row
      }

      // Advance
      when(dataNextRow >= tileRows.U) {
        dataState := dIdle  // single-row tile or done
      }.otherwise {
        // Pipeline next read
        io.trfReadPort0In.valid         := true.B
        io.trfReadPort0In.bits.whichBank := dataCmd.trfBank
        io.trfReadPort0In.bits.rRow     := dataNextRow
        when(dataCmd.op === Mxu0Op.PushAccBF16) {
          io.trfReadPort1In.valid         := true.B
          io.trfReadPort1In.bits.whichBank := dataCmd.trfBank + 1.U
          io.trfReadPort1In.bits.rRow     := dataNextRow
        }
        dataRow     := dataRow + 1.U
        dataNextRow := dataNextRow + 1.U
        dataState   := dSteady
      }
    }

    // ── dSteady: TRF data from previous cycle. Process, pipeline next. ──
    is(dSteady) {
      // Process (identical logic to dSetup)
      when(dataCmd.op === Mxu0Op.PushWeight) {
        val rowData = unpackRow(io.trfReadPort0Out.bits, p.inputFmt.ieeeWidth, p.vecLen)
        core.io.weightWrite.valid         := true.B
        core.io.weightWrite.bits.targetBuf := dataCmd.weightSlot
        core.io.weightWrite.bits.laneIdx  := dataRow(log2Ceil(p.numLanes) - 1, 0)
        core.io.weightWrite.bits.weights  := rowData
      }.elsewhen(dataCmd.op === Mxu0Op.PushAccFP8) {
        val rowData = unpackRow(io.trfReadPort0Out.bits, 8, p.numLanes)
        accBuf.io.loadFP8.valid       := true.B
        accBuf.io.loadFP8.bits.accSel := dataCmd.accSel
        accBuf.io.loadFP8.bits.rowIdx := dataRow(rowBits - 1, 0)
        accBuf.io.loadFP8.bits.data   := rowData
      }.elsewhen(dataCmd.op === Mxu0Op.PushAccBF16) {
        val lo = io.trfReadPort0Out.bits
        val hi = io.trfReadPort1Out.bits
        val bf16Row = Wire(Vec(p.numLanes, UInt(16.W)))
        for (i <- 0 until p.numLanes) {
          if (i < 16) bf16Row(i) := lo((i+1)*16-1, i*16)
          else        bf16Row(i) := hi((i-16+1)*16-1, (i-16)*16)
        }
        accBuf.io.loadBF16.valid       := true.B
        accBuf.io.loadBF16.bits.accSel := dataCmd.accSel
        accBuf.io.loadBF16.bits.rowIdx := dataRow(rowBits - 1, 0)
        accBuf.io.loadBF16.bits.data   := bf16Row
      }

      // Advance
      when(dataNextRow >= tileRows.U) {
        dataState := dIdle
      }.otherwise {
        io.trfReadPort0In.valid         := true.B
        io.trfReadPort0In.bits.whichBank := dataCmd.trfBank
        io.trfReadPort0In.bits.rRow     := dataNextRow
        when(dataCmd.op === Mxu0Op.PushAccBF16) {
          io.trfReadPort1In.valid         := true.B
          io.trfReadPort1In.bits.whichBank := dataCmd.trfBank + 1.U
          io.trfReadPort1In.bits.rRow     := dataNextRow
        }
        dataRow     := dataRow + 1.U
        dataNextRow := dataNextRow + 1.U
      }
    }

    // ── dPop: combinational acc buf read → TRF write ──
    is(dPop) {
      when(dataCmd.op === Mxu0Op.PopAccFP8) {
        accBuf.io.storeReq.accSel    := dataCmd.accSel
        accBuf.io.storeReq.rowIdx    := dataRow(rowBits - 1, 0)
        accBuf.io.storeReq.scaleE8M0 := dataCmd.scaleE8M0
        io.trfWritePort0.valid          := true.B
        io.trfWritePort0.bits.whichBank := dataCmd.trfBank
        io.trfWritePort0.bits.wRow      := dataRow
        io.trfWritePort0.bits.wData     := packElems(accBuf.io.storeFP8Out.map(_.pad(8)))
      }.elsewhen(dataCmd.op === Mxu0Op.PopAccBF16) {
        accBuf.io.storeReq.accSel    := dataCmd.accSel
        accBuf.io.storeReq.rowIdx    := dataRow(rowBits - 1, 0)
        accBuf.io.storeReq.scaleE8M0 := 127.U
        io.trfWritePort0.valid          := true.B
        io.trfWritePort0.bits.whichBank := dataCmd.trfBank
        io.trfWritePort0.bits.wRow      := dataRow
        io.trfWritePort0.bits.wData     := packElems(accBuf.io.storeBF16Out.take(16))
        io.trfWritePort1.valid          := true.B
        io.trfWritePort1.bits.whichBank := dataCmd.trfBank + 1.U
        io.trfWritePort1.bits.wRow      := dataRow
        io.trfWritePort1.bits.wData     := packElems(accBuf.io.storeBF16Out.drop(16))
      }
      when(dataRow + 1.U >= tileRows.U) {
        dataState := dIdle
      }.otherwise {
        dataRow := dataRow + 1.U
      }
    }
  }

  //  COMPUTE FSM
  val cIdle :: cSetup :: cActive :: cDrain :: Nil = Enum(4)
  val compState       = RegInit(cIdle)
  val compCmd         = Reg(new Mxu0Cmd(rfP))
  val compNextRow     = Reg(UInt(rowBits.W))
  val compRowsIssued  = Reg(UInt(rowCountW.W))
  val compRowsSent    = Reg(UInt(rowCountW.W))
  val compRowsWritten = Reg(UInt(rowCountW.W))

  io.computeBusy := (compState =/= cIdle)

  private val tagDepth = math.max(1, p.numPipeCuts)
  val rowTagData = Reg(Vec(tagDepth, UInt(rowBits.W)))
  for (i <- (tagDepth - 1) to 1 by -1) { rowTagData(i) := rowTagData(i - 1) }
  rowTagData(0) := 0.U

  val computeReady = (compState === cIdle) && isComputeOp
  val bothIdle = (dataState === dIdle) && (compState === cIdle)
  io.cmd.ready := bothIdle || dataReady || computeReady

  val isAccumulate = (compCmd.op === Mxu0Op.MatmulAcc)

  def driveCoreBeat(rowIdx: UInt): Unit = {
    accBuf.io.mxuReadReq.accSel := compCmd.accSel
    accBuf.io.mxuReadReq.rowIdx := rowIdx
    core.io.compute.valid             := true.B
    core.io.compute.bits.act          := unpackRow(io.trfReadPort0Out.bits, p.inputFmt.ieeeWidth, p.vecLen)
    core.io.compute.bits.psum         := accBuf.io.mxuReadData
    core.io.compute.bits.accumulate   := isAccumulate
    core.io.compute.bits.weightBufSel := compCmd.weightSlot
    rowTagData(0) := rowIdx
  }

  switch(compState) {
    is(cIdle) {
      when(computeReady && io.cmd.fire) {
        compCmd := io.cmd.bits
        compNextRow := 0.U
        compRowsIssued := 0.U
        compRowsSent := 0.U
        compRowsWritten := 0.U
        io.trfReadPort0In.valid         := true.B
        io.trfReadPort0In.bits.whichBank := io.cmd.bits.trfBank
        io.trfReadPort0In.bits.rRow     := 0.U
        compNextRow := 1.U
        compRowsIssued := 1.U
        compState := cSetup
      }
    }
    is(cSetup) {
      driveCoreBeat(0.U(rowBits.W))
      compRowsSent := 1.U
      when(tileRows.U > 1.U) {
        io.trfReadPort0In.valid         := true.B
        io.trfReadPort0In.bits.whichBank := compCmd.trfBank
        io.trfReadPort0In.bits.rRow     := compNextRow
        compNextRow := compNextRow + 1.U
        compRowsIssued := compRowsIssued + 1.U
        compState := cActive
      }.otherwise { compState := cDrain }
    }
    is(cActive) {
      driveCoreBeat(compRowsSent(rowBits - 1, 0))
      compRowsSent := compRowsSent + 1.U
      when(compRowsIssued < tileRows.U) {
        io.trfReadPort0In.valid         := true.B
        io.trfReadPort0In.bits.whichBank := compCmd.trfBank
        io.trfReadPort0In.bits.rRow     := compNextRow
        compNextRow := compNextRow + 1.U
        compRowsIssued := compRowsIssued + 1.U
      }.otherwise { compState := cDrain }
    }
    is(cDrain) {
      when(compRowsWritten >= tileRows.U) { compState := cIdle }
    }
  }

  when(core.io.out.valid) {
    val writeRow = if (tagDepth == 1) rowTagData(0) else rowTagData(tagDepth - 1)
    accBuf.io.mxuWrite.valid       := true.B
    accBuf.io.mxuWrite.bits.accSel := compCmd.accSel
    accBuf.io.mxuWrite.bits.rowIdx := writeRow
    accBuf.io.mxuWrite.bits.data   := core.io.out.bits
    compRowsWritten := compRowsWritten + 1.U
  }
}

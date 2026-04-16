// ============================================================================
// MregFile.scala — Multi-banked matrix register file.
// 1R1W-per-bank version.
//
// Organisation:
//   NUM_MREG       = 64  matrix registers
//   MREG_ROWS      = 32  rows per register
//   MREG_ROW_BYTES = 32  bytes per row
//   MREG_BYTES     = 1024 bytes per register
//
// This version is written so that each physical bank sees at most:
//   - one read access per cycle
//   - one write access per cycle
//
// That makes each bank structurally compatible with a 1R1W SRAM.
// Any multi-port behavior is handled by arbitration *before* the SRAM.
// ============================================================================

package atlas.mreg

import chisel3._
import chisel3.util._
import atlas.common.{MregParams, MregReadReq, MregWriteReq}

class MregFile(p: MregParams) extends Module {

  val io = IO(new Bundle {

    // ── Read request ports ──────────────────────────────────────────
    val mxu0ReadReq0 = Flipped(Valid(new MregReadReq(p)))
    val mxu0ReadReq1 = Flipped(Valid(new MregReadReq(p)))
    val mxu1ReadReq0 = Flipped(Valid(new MregReadReq(p)))
    val mxu1ReadReq1 = Flipped(Valid(new MregReadReq(p)))
    val vpuReadReq0  = Flipped(Valid(new MregReadReq(p)))
    val vpuReadReq1  = Flipped(Valid(new MregReadReq(p)))
    val lsuReadReq   = Flipped(Valid(new MregReadReq(p)))
    val xluReadReq   = Flipped(Valid(new MregReadReq(p)))

    // ── Write request ports ─────────────────────────────────────────
    val mxu0WriteReq0 = Flipped(Valid(new MregWriteReq(p)))
    val mxu0WriteReq1 = Flipped(Valid(new MregWriteReq(p)))
    val mxu1WriteReq0 = Flipped(Valid(new MregWriteReq(p)))
    val mxu1WriteReq1 = Flipped(Valid(new MregWriteReq(p)))
    val vpuWriteReq0  = Flipped(Valid(new MregWriteReq(p)))
    val vpuWriteReq1  = Flipped(Valid(new MregWriteReq(p)))
    val lsuWriteReq   = Flipped(Valid(new MregWriteReq(p)))
    val xluWriteReq   = Flipped(Valid(new MregWriteReq(p)))

    // ── Read response ports ─────────────────────────────────────────
    val mxu0ReadResp0 = Valid(UInt(p.mregRowBits.W))
    val mxu0ReadResp1 = Valid(UInt(p.mregRowBits.W))
    val mxu1ReadResp0 = Valid(UInt(p.mregRowBits.W))
    val mxu1ReadResp1 = Valid(UInt(p.mregRowBits.W))
    val vpuReadResp0  = Valid(UInt(p.mregRowBits.W))
    val vpuReadResp1  = Valid(UInt(p.mregRowBits.W))
    val lsuReadResp   = Valid(UInt(p.mregRowBits.W))
    val xluReadResp   = Valid(UInt(p.mregRowBits.W))
  })

  // ==========================================================================
  // One physical 1R1W bank per matrix register
  // ==========================================================================

  val banks = Seq.fill(p.numMreg) {
    SyncReadMem(p.mregRows, UInt(p.mregRowBits.W))
  }

  // Gather ports into sequences for uniform handling.
  val readReqs = Seq(
    io.mxu0ReadReq0,
    io.mxu0ReadReq1,
    io.mxu1ReadReq0,
    io.mxu1ReadReq1,
    io.vpuReadReq0,
    io.vpuReadReq1,
    io.lsuReadReq,
    io.xluReadReq
  )

  val readResps = Seq(
    io.mxu0ReadResp0,
    io.mxu0ReadResp1,
    io.mxu1ReadResp0,
    io.mxu1ReadResp1,
    io.vpuReadResp0,
    io.vpuReadResp1,
    io.lsuReadResp,
    io.xluReadResp
  )

  val writeReqs = Seq(
    io.mxu0WriteReq0,
    io.mxu0WriteReq1,
    io.mxu1WriteReq0,
    io.mxu1WriteReq1,
    io.vpuWriteReq0,
    io.vpuWriteReq1,
    io.lsuWriteReq,
    io.xluWriteReq
  )

  val numReadPorts  = readReqs.length
  val numWritePorts = writeReqs.length

  // ==========================================================================
  // Per-bank arbitration
  // ==========================================================================

  // For each bank:
  //   - choose at most one read requester
  //   - choose at most one write requester
  //
  // Priority is fixed by port order in readReqs/writeReqs.
  // Conflicts are flagged with assertions.

  val bankReadValid = Wire(Vec(p.numMreg, Bool()))
  val bankReadRow   = Wire(Vec(p.numMreg, UInt(log2Ceil(p.mregRows).W)))
  val bankReadPort  = Wire(Vec(p.numMreg, UInt(log2Ceil(numReadPorts).W)))

  val bankWriteValid = Wire(Vec(p.numMreg, Bool()))
  val bankWriteRow   = Wire(Vec(p.numMreg, UInt(log2Ceil(p.mregRows).W)))
  val bankWriteData  = Wire(Vec(p.numMreg, UInt(p.mregRowBits.W)))
  val bankWritePort  = Wire(Vec(p.numMreg, UInt(log2Ceil(numWritePorts).W)))

  for (b <- 0 until p.numMreg) {
    val readHits  = Wire(Vec(numReadPorts, Bool()))
    val writeHits = Wire(Vec(numWritePorts, Bool()))

    for (rp <- 0 until numReadPorts) {
      readHits(rp) := readReqs(rp).valid && (readReqs(rp).bits.mregId === b.U)
    }
    for (wp <- 0 until numWritePorts) {
      writeHits(wp) := writeReqs(wp).valid && (writeReqs(wp).bits.mregId === b.U)
    }

    val readCount  = PopCount(readHits)
    val writeCount = PopCount(writeHits)

    assert(readCount <= 1.U,
      s"MregFile bank conflict: multiple read ports targeting same bank $b")
    assert(writeCount <= 1.U,
      s"MregFile bank conflict: multiple write ports targeting same bank $b")

    bankReadValid(b) := readHits.asUInt.orR
    bankReadPort(b)  := PriorityEncoder(readHits.asUInt)
    bankReadRow(b)   := Mux1H(readHits, readReqs.map(_.bits.row))

    bankWriteValid(b) := writeHits.asUInt.orR
    bankWritePort(b)  := PriorityEncoder(writeHits.asUInt)
    bankWriteRow(b)   := Mux1H(writeHits, writeReqs.map(_.bits.row))
    bankWriteData(b)  := Mux1H(writeHits, writeReqs.map(_.bits.data))
  }

  // ==========================================================================
  // Physical SRAM accesses: exactly one read + one write per bank
  // ==========================================================================

  val bankReadData = Wire(Vec(p.numMreg, UInt(p.mregRowBits.W)))

  for (b <- 0 until p.numMreg) {
    bankReadData(b) := banks(b).read(bankReadRow(b), bankReadValid(b))
    when(bankWriteValid(b)) {
      banks(b).write(bankWriteRow(b), bankWriteData(b))
    }
  }

  // ==========================================================================
  // Route read responses back to the originating read port
  // ==========================================================================

  // Register which port issued the read for each bank, since SyncReadMem
  // returns data one cycle later.
  val bankReadValid_d = RegNext(bankReadValid, init = VecInit(Seq.fill(p.numMreg)(false.B)))
  val bankReadPort_d  = RegNext(bankReadPort)

  // Default outputs
  for (rp <- 0 until numReadPorts) {
    readResps(rp).valid := false.B
    readResps(rp).bits  := 0.U
  }

  // Since we assert at most one read per bank, but multiple different banks
  // may target different read ports in the same cycle, each port should also
  // only get one returning response per cycle. Assert that too.
  for (rp <- 0 until numReadPorts) {
    val respHits = Wire(Vec(p.numMreg, Bool()))
    for (b <- 0 until p.numMreg) {
      respHits(b) := bankReadValid_d(b) && (bankReadPort_d(b) === rp.U)
    }

    assert(PopCount(respHits) <= 1.U,
      s"MregFile response conflict: multiple banks returning to read port $rp")

    readResps(rp).valid := respHits.asUInt.orR
    readResps(rp).bits  := Mux1H(respHits, bankReadData)
  }
}

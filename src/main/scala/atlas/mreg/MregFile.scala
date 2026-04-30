// ============================================================================
// MregFile.scala — Multi-banked matrix register file.
// 1R1W-per-bank version.
//
// Organisation:
//   NUM_MREG       = 64  matrix registers
//   NUM_MREG_BANKS = 32  physical SRAM banks
//   MREG_ROWS      = 32  rows per register
//   MREG_ROW_BYTES = 32  bytes per row
//   MREG_BYTES     = 1024 bytes per register
//
// Physical layout:
//   - Each SRAM bank is 32 B wide and 64 entries deep.
//   - m_i and m_{i+32} share physical bank i.
//   - Rows 0..31 hold m_i; rows 32..63 hold m_{i+32}.
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
  // One physical 1R1W bank per low/high architectural-register pair.
  // ==========================================================================

  val banks = Seq.fill(p.numMregBanks) {
    SyncReadMem(p.mregBankRows, UInt(p.mregRowBits.W))
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
  // Shared predecode + explicit tree-mux helpers
  // ==========================================================================

  // Decode the physical bank once per request port, then reuse the one-hot
  // selection for each bank's arbitration tree.
  val readBankOHs = readReqs.map(req =>
    UIntToOH(p.physicalBank(req.bits.mregId), p.numMregBanks) &
      Fill(p.numMregBanks, req.valid))
  val readPhysRows = readReqs.map(req =>
    p.physicalRow(req.bits.mregId, req.bits.row))

  val writeBankOHs = writeReqs.map(req =>
    UIntToOH(p.physicalBank(req.bits.mregId), p.numMregBanks) &
      Fill(p.numMregBanks, req.valid))
  val writePhysRows = writeReqs.map(req =>
    p.physicalRow(req.bits.mregId, req.bits.row))

  class ReadTreeSel extends Bundle {
    val valid = Bool()
    val row   = UInt(p.mregBankRowAddrBits.W)
    val port  = UInt(log2Ceil(numReadPorts).W)
  }

  class WriteTreeSel extends Bundle {
    val valid = Bool()
    val row   = UInt(p.mregBankRowAddrBits.W)
    val data  = UInt(p.mregRowBits.W)
  }

  class RespTreeSel extends Bundle {
    val valid = Bool()
    val data  = UInt(p.mregRowBits.W)
  }

  def mergeReadSel(lhs: ReadTreeSel, rhs: ReadTreeSel): ReadTreeSel = {
    val out = Wire(new ReadTreeSel)
    out.valid := lhs.valid || rhs.valid
    out.row   := Mux(lhs.valid, lhs.row, rhs.row)
    out.port  := Mux(lhs.valid, lhs.port, rhs.port)
    out
  }

  def mergeWriteSel(lhs: WriteTreeSel, rhs: WriteTreeSel): WriteTreeSel = {
    val out = Wire(new WriteTreeSel)
    out.valid := lhs.valid || rhs.valid
    out.row   := Mux(lhs.valid, lhs.row, rhs.row)
    out.data  := Mux(lhs.valid, lhs.data, rhs.data)
    out
  }

  def mergeRespSel(lhs: RespTreeSel, rhs: RespTreeSel): RespTreeSel = {
    val out = Wire(new RespTreeSel)
    out.valid := lhs.valid || rhs.valid
    out.data  := Mux(lhs.valid, lhs.data, rhs.data)
    out
  }

  def treeReduce[T](leaves: Seq[T])(merge: (T, T) => T): T = {
    require(leaves.nonEmpty, "treeReduce requires at least one leaf")
    if (leaves.length == 1) {
      leaves.head
    } else {
      val nextLevel = leaves.grouped(2).map { group =>
        if (group.length == 1) group.head else merge(group.head, group(1))
      }.toSeq
      treeReduce(nextLevel)(merge)
    }
  }

  // ==========================================================================
  // Per-bank arbitration
  // ==========================================================================

  // For each bank:
  //   - choose at most one read requester
  //   - choose at most one write requester
  //
  // Priority is fixed by port order in readReqs/writeReqs.
  // Conflicts are flagged with assertions.

  val bankReadValid = Wire(Vec(p.numMregBanks, Bool()))
  val bankReadRow   = Wire(Vec(p.numMregBanks, UInt(p.mregBankRowAddrBits.W)))
  val bankReadPort  = Wire(Vec(p.numMregBanks, UInt(log2Ceil(numReadPorts).W)))

  val bankWriteValid = Wire(Vec(p.numMregBanks, Bool()))
  val bankWriteRow   = Wire(Vec(p.numMregBanks, UInt(p.mregBankRowAddrBits.W)))
  val bankWriteData  = Wire(Vec(p.numMregBanks, UInt(p.mregRowBits.W)))

  for (b <- 0 until p.numMregBanks) {
    val readHits  = Wire(Vec(numReadPorts, Bool()))
    val writeHits = Wire(Vec(numWritePorts, Bool()))

    for (rp <- 0 until numReadPorts) {
      readHits(rp) := readBankOHs(rp)(b)
    }
    for (wp <- 0 until numWritePorts) {
      writeHits(wp) := writeBankOHs(wp)(b)
    }

    val readCount  = PopCount(readHits)
    val writeCount = PopCount(writeHits)

    assert(readCount <= 1.U,
      s"MregFile bank conflict: multiple read ports targeting physical bank $b (m$b or m${b + p.numMregBanks})")
    assert(writeCount <= 1.U,
      s"MregFile bank conflict: multiple write ports targeting physical bank $b (m$b or m${b + p.numMregBanks})")

    val readLeaves = (0 until numReadPorts).map { rp =>
      val leaf = Wire(new ReadTreeSel)
      leaf.valid := readHits(rp)
      leaf.row   := readPhysRows(rp)
      leaf.port  := rp.U
      leaf
    }

    val writeLeaves = (0 until numWritePorts).map { wp =>
      val leaf = Wire(new WriteTreeSel)
      leaf.valid := writeHits(wp)
      leaf.row   := writePhysRows(wp)
      leaf.data  := writeReqs(wp).bits.data
      leaf
    }

    val readSel  = treeReduce(readLeaves)(mergeReadSel)
    val writeSel = treeReduce(writeLeaves)(mergeWriteSel)

    bankReadValid(b) := readSel.valid
    bankReadPort(b)  := readSel.port
    bankReadRow(b)   := readSel.row

    bankWriteValid(b) := writeSel.valid
    bankWriteRow(b)   := writeSel.row
    bankWriteData(b)  := writeSel.data
  }

  // ==========================================================================
  // Physical SRAM accesses: exactly one read + one write per bank
  // ==========================================================================

  val bankReadData = Wire(Vec(p.numMregBanks, UInt(p.mregRowBits.W)))

  for (b <- 0 until p.numMregBanks) {
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
  val bankReadValid_d = RegNext(bankReadValid, init = VecInit(Seq.fill(p.numMregBanks)(false.B)))
  val bankReadPort_d  = RegNext(bankReadPort)

  val readRespValidRaw = Wire(Vec(numReadPorts, Bool()))
  val readRespDataRaw  = Wire(Vec(numReadPorts, UInt(p.mregRowBits.W)))

  // Since we assert at most one read per bank, but multiple different banks
  // may target different read ports in the same cycle, each port should also
  // only get one returning response per cycle. Assert that too.
  for (rp <- 0 until numReadPorts) {
    val respHits = Wire(Vec(p.numMregBanks, Bool()))
    for (b <- 0 until p.numMregBanks) {
      respHits(b) := bankReadValid_d(b) && (bankReadPort_d(b) === rp.U)
    }

    assert(PopCount(respHits) <= 1.U,
      s"MregFile response conflict: multiple banks returning to read port $rp")

    val respLeaves = (0 until p.numMregBanks).map { b =>
      val leaf = Wire(new RespTreeSel)
      leaf.valid := respHits(b)
      leaf.data  := bankReadData(b)
      leaf
    }

    val respSel = treeReduce(respLeaves)(mergeRespSel)

    readRespValidRaw(rp) := respSel.valid
    // Response bits are don't-care when valid is low; keeping the raw selected
    // bank data avoids injecting the respSel.valid control tree into every bit.
    readRespDataRaw(rp)  := respSel.data
  }

  // Keep the mreg itself at a single registered-SRAM response latency.
  // Timing-sensitive consumers add their own ingress flops so only the
  // problematic read cones pay an extra cycle.
  for (rp <- 0 until numReadPorts) {
    readResps(rp).valid := readRespValidRaw(rp)
    readResps(rp).bits  := readRespDataRaw(rp)
  }
}

// ============================================================================
// MregFile.scala — Multi-banked matrix register file.
//
// Organisation (from spec):
//   NUM_MREG       = 64  matrix registers
//   MREG_ROWS      = 32  rows per register
//   MREG_ROW_BYTES = 32  bytes per row  (256 bits = 32 × fp8)
//   MREG_BYTES     = 1024 bytes per register
//
// Access model:
//   • 8 read ports  — one-cycle latency (SyncReadMem).
//   • 8 write ports — combinational write, one register per port per cycle.
//   • Each port is dedicated to a specific functional unit so that
//     arbitration is handled structurally (no muxing needed).
//
// Port assignment:
//   Read  0–1 : MXU-0          Write 0–1 : MXU-0
//   Read  2–3 : MXU-1          Write 2–3 : MXU-1
//   Read  4–5 : VPU            Write 4–5 : VPU
//   Read  6   : LSU / DMA      Write 6   : LSU / DMA
//   Read  7   : XLU            Write 7   : XLU
// ============================================================================

package atlas.mreg

import chisel3._
import chisel3.util._
import atlas.common.{MregParams, MregReadReq, MregWriteReq}

/** Multi-ported matrix register file backed by per-register SyncReadMem.
  *
  * @param p  Matrix-register geometry parameters (spec-aligned).
  */
class MregFile(p: MregParams) extends Module {

  val io = IO(new Bundle {

    // ── Read request ports (valid = issue read) ────────────────────
    val mxu0ReadReq0 = Flipped(Valid(new MregReadReq(p)))
    val mxu0ReadReq1 = Flipped(Valid(new MregReadReq(p)))
    val mxu1ReadReq0 = Flipped(Valid(new MregReadReq(p)))
    val mxu1ReadReq1 = Flipped(Valid(new MregReadReq(p)))
    val vpuReadReq0  = Flipped(Valid(new MregReadReq(p)))
    val vpuReadReq1  = Flipped(Valid(new MregReadReq(p)))
    val lsuReadReq   = Flipped(Valid(new MregReadReq(p)))
    val xluReadReq   = Flipped(Valid(new MregReadReq(p)))

    // ── Write request ports ────────────────────────────────────────
    val mxu0WriteReq0 = Flipped(Valid(new MregWriteReq(p)))
    val mxu0WriteReq1 = Flipped(Valid(new MregWriteReq(p)))
    val mxu1WriteReq0 = Flipped(Valid(new MregWriteReq(p)))
    val mxu1WriteReq1 = Flipped(Valid(new MregWriteReq(p)))
    val vpuWriteReq0  = Flipped(Valid(new MregWriteReq(p)))
    val vpuWriteReq1  = Flipped(Valid(new MregWriteReq(p)))
    val lsuWriteReq   = Flipped(Valid(new MregWriteReq(p)))
    val xluWriteReq   = Flipped(Valid(new MregWriteReq(p)))

    // ── Read response ports (valid one cycle after the request) ────
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
  // SRAM bank array — one SyncReadMem per matrix register
  // ==========================================================================

  val banks = Seq.fill(p.numMreg)(
    SyncReadMem(p.mregRows, UInt(p.mregRowBits.W))
  )

  // ==========================================================================
  // Read logic — one-cycle synchronous reads with register selection
  // ==========================================================================
  //
  // For every read port we:
  //   1. Issue a synchronous read to every bank.  Only the bank whose index
  //      matches `mregId` has its chip-enable asserted; the others return
  //      don't-care data but consume no dynamic power in a real SRAM.
  //   2. Mux the correct bank's output on the *following* cycle using a
  //      registered copy of `mregId`.

  /** All (request, response) read port pairs, gathered for uniform wiring. */
  val readPorts: Seq[(Valid[MregReadReq], Valid[UInt])] = Seq(
    (io.mxu0ReadReq0, io.mxu0ReadResp0),
    (io.mxu0ReadReq1, io.mxu0ReadResp1),
    (io.mxu1ReadReq0, io.mxu1ReadResp0),
    (io.mxu1ReadReq1, io.mxu1ReadResp1),
    (io.vpuReadReq0,  io.vpuReadResp0),
    (io.vpuReadReq1,  io.vpuReadResp1),
    (io.lsuReadReq,   io.lsuReadResp),
    (io.xluReadReq,   io.xluReadResp)
  )

  for ((req, resp) <- readPorts) {
    // Read every bank in parallel; only the selected one is enabled.
    val bankOutputs = Wire(Vec(p.numMreg, UInt(p.mregRowBits.W)))

    for (i <- 0 until p.numMreg) {
      val isSelected = req.valid && (req.bits.mregId === i.U)
      bankOutputs(i) := banks(i).read(req.bits.row, isSelected)
    }

    // Output appears one cycle after the request (SyncReadMem latency).
    resp.valid := RegNext(req.valid, init = false.B)
    resp.bits  := bankOutputs(RegNext(req.bits.mregId))
  }

  // ==========================================================================
  // Write logic — combinational writes, one register per port per cycle
  // ==========================================================================

  /** All write ports, gathered for uniform wiring. */
  val writePorts: Seq[Valid[MregWriteReq]] = Seq(
    io.mxu0WriteReq0, io.mxu0WriteReq1,
    io.mxu1WriteReq0, io.mxu1WriteReq1,
    io.vpuWriteReq0,  io.vpuWriteReq1,
    io.lsuWriteReq,   io.xluWriteReq
  )

  for (req <- writePorts) {
    when(req.valid) {
      for (i <- 0 until p.numMreg) {
        when(req.bits.mregId === i.U) {
          banks(i).write(req.bits.row, req.bits.data)
        }
      }
    }
  }
}

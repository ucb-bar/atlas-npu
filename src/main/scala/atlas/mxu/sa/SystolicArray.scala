// ============================================================================
// SystolicArray.scala — Systolic-array MXU datapath.
//
// Computes the equivalent of PyTorch's F.linear (y = x·Aᵀ + b) in hardware.
//
// Uses the shared ComputeReq bundle: partial-sum data arrives through the
// `psum` field (pre-read by the sequencer), so the core never touches the
// accumulation buffer directly.
//
// Timing:
//   • Activations are skewed left-to-right across rows (delay = row index).
//   • Addends (partial sums) are skewed top-to-bottom across columns.
//   • Outputs are de-skewed bottom-to-top so they emerge as a single row.
//   • Total pipeline latency = MXU_ARRAY_ROWS + MXU_ARRAY_COLS − 2 cycles.
//     The bottom-row mac feeds the deskew chain combinationally; the accBuf
//     SyncReadMem write downstream provides the missing pipeline stage.
// ============================================================================

package atlas.sa

import chisel3._
import chisel3.util._
import atlas.common._
import atlas.mxu.ComputeReq

/** Systolic-array compute mesh.
  *
  * @param p  Systolic-array geometry and PE-architecture parameters.
  */
class SystolicArray(p: SystolicArrayParams) extends Module {

  val io = IO(new Bundle {
    /** Compute request: activations, partial sums, and control. */
    val computeReq = Flipped(Valid(new ComputeReq(p.mxu)))
    /** Weight tile from slot 0 — indexed as (row)(col). */
    val weights0   = Input(Vec(p.rows, Vec(p.cols, UInt(p.inT.ieeeWidth.W))))
    /** Weight tile from slot 1 — indexed as (row)(col). */
    val weights1   = Input(Vec(p.rows, Vec(p.cols, UInt(p.inT.ieeeWidth.W))))
    /** Result row (valid after pipeline flush). */
    val outputRow  = Valid(Vec(p.cols, UInt(p.outT.ieeeWidth.W)))
  })

  val req = io.computeReq.bits

  // ==========================================================================
  // PE mesh instantiation
  // ==========================================================================

  val peMesh = Module(new PEMesh(p.rows, p.cols, p.peArch))

  // ==========================================================================
  // Left edge — skew and format activations + weight-slot select
  // ==========================================================================

  for (i <- 0 until p.rows) {
    val actSkewed       = ShiftRegister(req.act(i), i)
    val weightSelSkewed = ShiftRegister(req.weightBufSel, i)
    peMesh.io.actVec(i)           := p.peArch.formatMul(actSkewed)
    peMesh.io.weightReadSelVec(i) := weightSelSkewed
  }

  // ==========================================================================
  // Top edge — skew and format addends (partial sums)
  // ==========================================================================

  for (j <- 0 until p.cols) {
    val zero   = 0.U(p.outT.ieeeWidth.W)
    val addend = Mux(req.accumulate, req.psum(j), zero)
    peMesh.io.addendVec(j) := p.peArch.formatAdd(ShiftRegister(addend, j))
  }

  // ==========================================================================
  // Bottom edge — format and de-skew outputs
  // ==========================================================================

  for (j <- 0 until p.cols) {
    val formatted = p.peArch.formatOut(peMesh.io.outVec(j))
    io.outputRow.bits(j) := ShiftRegister(formatted, (p.cols - 1) - j)
  }

  // ==========================================================================
  // Z-axis — format and feed weights
  // ==========================================================================

  for (i <- 0 until p.rows; j <- 0 until p.cols) {
    peMesh.io.weights0(i)(j) := p.peArch.formatMul(io.weights0(i)(j))
    peMesh.io.weights1(i)(j) := p.peArch.formatMul(io.weights1(i)(j))
  }

  // ==========================================================================
  // Valid pipeline — total latency = rows + cols − 2
  // ==========================================================================

  io.outputRow.valid := ShiftRegister(
    io.computeReq.valid, p.rows + p.cols - 2, false.B, true.B)
}

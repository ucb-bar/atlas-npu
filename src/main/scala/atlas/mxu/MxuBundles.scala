// ============================================================================
// MxuBundles.scala — Hardware interface bundles of the FP8 MXU datapath (mxu1).
//
// The MXU-facing bundle types live here: commands, compute requests, weight
// writes, and accumulation-buffer operations.
//
// Parameterized by MxuParams (no dependency on atlas.common).  The MxuCmd
// takes an Int for the mreg index width to avoid importing MregParams.
// ============================================================================

package atlas.mxu

import chisel3._
import chisel3.util._

// ============================================================================
// Command opcode & sequencer command
// ============================================================================

/** MXU command opcodes of the mxu1 sequencer: funct7[3:0] of the instruction.
  *
  * Push/Pop move data between the matrix register file (mreg) and the MXU's
  * internal weight slots or accumulation buffers.  Matmul launches a
  * matrix-multiply on the loaded weights and activations.
  */
object MxuOp extends ChiselEnum {
  val PushWeight       = Value(0.U)  // mreg → weight slot.
  val PushAccBF16      = Value(1.U)  // mreg (BF16) → accum buffer.
  val PushAccFP32      = Value(2.U)  // 4 mregs (FP32) → accum buffer, a copy.
  val PushAccFP8       = Value(3.U)  // mreg (FP8)  → accum buffer (converted to accumFmt).
  val PopAccBF16       = Value(4.U)  // accum buffer → mreg (BF16).
  val PopAccBF16Scaled = Value(5.U)  // accum buffer × 2^(scaleE8M0 - 127) → mreg (BF16).
  val PopAccFP32       = Value(6.U)  // accum buffer → 4 mregs (FP32), a copy.
  val PopAccFP8        = Value(7.U)  // accum buffer → mreg (quant to FP8).
  val Matmul           = Value(8.U)  // compute, result overwrites accum buffer.
  val MatmulAcc        = Value(9.U)  // compute, result accumulates into accum buffer.
}

/** Sequencer command issued by the scalar core.
  *
  * @param mregIdBits  Bit-width of the matrix-register index
  *                    (spec: log2(NUM_MREG) = 6).
  */
class MxuCmd(mregIdBits: Int) extends Bundle {
  val op         = MxuOp()              // Operation to perform.
  val mregId     = UInt(mregIdBits.W)   // Source/destination matrix register.
  val accSel     = Bool()               // Which accumulation buffer (0 or 1).
  val weightSlot = Bool()               // Which weight slot (0 or 1).
  val scaleE8M0  = UInt(8.W)            // E8M0 block-scaling factor (PopAccFP8, PopAccBF16Scaled).
  val rows       = UInt(5.W)            // Row count M, 0 for 32: rows 0..M-1 (PushWeight: all).
}

// ============================================================================
// Compute request
// ============================================================================

/** Compute (matmul) request of the MXU core.
  *
  * The sequencer reads the accumulation buffer externally and passes
  * partial-sum data here, so the core never touches the buffer directly.
  *
  * Dimensions follow the spec:
  *   act:  MXU_ARRAY_ROWS  × inputFmt   (one activation row)
  *   psum: MXU_ARRAY_COLS  × accumFmt   (one partial-sum row)
  */
class ComputeReq(p: MxuParams) extends Bundle {
  val act          = Vec(p.arrayRows, UInt(p.inputFmt.ieeeWidth.W))  // Activation row.
  val psum         = Vec(p.arrayCols, UInt(p.accumFmt.ieeeWidth.W))  // Partial-sum row.
  val accumulate   = Bool()                                          // true → add psum.
  val weightBufSel = Bool()                                          // Which weight slot.
}

// ============================================================================
// Weight buffer write request
// ============================================================================

/** Write one lane's weight vector into a weight slot.
  *
  * The sequencer pushes one column per cycle; after MXU_ARRAY_ROWS cycles
  * the full WEIGHT_SLOT_BYTES tile is loaded.
  */
class WeightWriteReq(p: MxuParams) extends Bundle {
  val weightSlot = Bool()                                            // Which slot (0 or 1).
  val laneIdx    = UInt(p.colIdxBits.W)                              // Output lane (column).
  val data       = Vec(p.arrayRows, UInt(p.inputFmt.ieeeWidth.W))   // Weight vector.
}

// ============================================================================
// Accumulation buffer bundles
// ============================================================================

/** Write a result row from the compute pipeline into an accum buffer. */
class AccBufWriteReq(p: MxuParams) extends Bundle {
  val accSel = Bool()                                                       // Target buffer.
  val rowIdx = UInt(p.rowIdxBits.W)                                         // Target row.
  val data   = Vec(p.accumBufferColsFP32, UInt(p.accumFmt.ieeeWidth.W))     // Row data.
}

/** Address an accum-buffer row for compute-side partial-sum read. */
class AccBufReadAddr(p: MxuParams) extends Bundle {
  val accSel = Bool()                                                       // Source buffer.
  val rowIdx = UInt(p.rowIdxBits.W)                                         // Source row.
}

/** Load a row into an accum buffer (push path, after conversion). */
class AccBufLoadReq(p: MxuParams) extends Bundle {
  val accSel = Bool()                                                       // Target buffer.
  val rowIdx = UInt(p.rowIdxBits.W)                                         // Target row.
  val data   = Vec(p.accumBufferColsFP32, UInt(p.accumFmt.ieeeWidth.W))     // Row data.
  val mask   = Vec(p.accumBufferColsFP32, Bool())                           // Columns to write.
}

/** Address an accum-buffer row for store/pop read-out (pre-quantization). */
class AccBufStoreAddr(p: MxuParams) extends Bundle {
  val accSel = Bool()                                                       // Source buffer.
  val rowIdx = UInt(p.rowIdxBits.W)                                         // Source row.
}

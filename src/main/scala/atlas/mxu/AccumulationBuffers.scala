// ============================================================================
// AccumulationBuffers.scala — Dual BF16 accumulation-buffer storage.
//
// Spec parameters (per buffer):
//   ACCUM_BUFFER_ROWS      = 32   rows
//   ACCUM_BUFFER_COLS_BF16 = 32   BF16 columns
//   ACCUM_BUFFER_BYTES     = 2048 bytes  (32 × 32 × 2)
//
// Two identically-sized buffers are selected by the `accSel` bit in each
// bundle.  The module is parameterized by MxuParams so both mxu0 (systolic
// array) and mxu1 (inner-product tree) share the same implementation.
//
// Three independent access paths:
//
//   1. Compute write / read — the MXU datapath writes result rows and reads
//      partial sums during matrix-multiply operations.
//
//   2. Load (push path) — the sequencer writes BF16 data into a buffer
//      after optional FP8→BF16 dequantization (dequant lives in sequencer).
//
//   3. Store (pop path) — the sequencer reads raw BF16 rows out of a buffer
//      for optional BF16→FP8 quantization (quant lives in sequencer).
//
// All (de)quantization happens in the sequencer, not here.
// ============================================================================

package atlas.mxu

import chisel3._
import chisel3.util._

/** Dual accumulation-buffer storage for one MXU.
  *
  * @param p  MXU geometry parameters (provides buffer dimensions and FP format).
  */
class AccumulationBuffers(p: MxuParams) extends Module {

  // Shorthand for the row type used throughout this module.
  private val numCols  = p.accumBufferColsBF16
  private val colWidth = p.accumFmt.ieeeWidth
  private type RowT    = Vec[UInt]

  val io = IO(new Bundle {

    // ── Compute pipeline (MXU datapath ↔ accum buffer) ─────────────
    /** Write a BF16 result row from the compute pipeline. */
    val computeWriteReq = Flipped(Valid(new AccBufWriteReq(p)))
    /** Address for a compute-side partial-sum read. */
    val computeReadAddr = Input(new AccBufReadAddr(p))
    /** Pulse to request a compute-side partial-sum read. */
    val computeReadEn   = Input(Bool())
    /** Partial-sum row returned one cycle after `computeReadEn`. */
    val computeReadData = Output(Vec(numCols, UInt(colWidth.W)))

    // ── Load path (sequencer → accum buffer) ───────────────────────
    /** Write a BF16 row into the buffer (post-dequant push). */
    val loadReq = Flipped(Valid(new AccBufLoadReq(p)))

    // ── Store path (accum buffer → sequencer) ──────────────────────
    /** Address for a store/pop read-out. */
    val storeAddr   = Input(new AccBufStoreAddr(p))
    /** Pulse to request a store/pop read-out. */
    val storeReadEn = Input(Bool())
    /** Raw BF16 row returned one cycle after `storeReadEn`. */
    val storeData   = Output(Vec(numCols, UInt(colWidth.W)))
  })

  // ==========================================================================
  // Storage — one synchronous-read memory per buffer
  // ==========================================================================

  val buffer0 = SyncReadMem(p.accumBufferRows, Vec(numCols, UInt(colWidth.W)))
  val buffer1 = SyncReadMem(p.accumBufferRows, Vec(numCols, UInt(colWidth.W)))

  // ==========================================================================
  // Safety: compute-write and load must not target the same buffer
  // ==========================================================================

  assert(!(io.computeWriteReq.valid && io.loadReq.valid &&
           io.computeWriteReq.bits.accSel === io.loadReq.bits.accSel),
    "AccumulationBuffers: compute write and load target the same buffer")

  // ==========================================================================
  // Helper: select buffer by accSel for reads
  // ==========================================================================

  /** Synchronous read from whichever buffer `sel` indicates. */
  private def readBuffer(sel: Bool, row: UInt, en: Bool): Vec[UInt] = {
    val readSel = RegEnable(sel, en)
    val data0   = buffer0.read(row, en && !sel)
    val data1   = buffer1.read(row, en && sel)
    Mux(readSel, data1, data0)
  }

  /** Write to whichever buffer `sel` indicates. */
  private def writeBuffer(sel: Bool, row: UInt, data: Vec[UInt]): Unit = {
    when(sel) {
      buffer1.write(row, data)
    }.otherwise {
      buffer0.write(row, data)
    }
  }

  // ==========================================================================
  // Compute pipeline write (result row writeback)
  // ==========================================================================

  when(io.computeWriteReq.valid) {
    writeBuffer(
      io.computeWriteReq.bits.accSel,
      io.computeWriteReq.bits.rowIdx,
      io.computeWriteReq.bits.data
    )
  }

  // ==========================================================================
  // Compute pipeline read (partial-sum fetch)
  // ==========================================================================

  io.computeReadData := readBuffer(
    io.computeReadAddr.accSel,
    io.computeReadAddr.rowIdx,
    io.computeReadEn
  )

  // ==========================================================================
  // Load path (sequencer push — BF16 data into buffer)
  // ==========================================================================

  when(io.loadReq.valid) {
    writeBuffer(
      io.loadReq.bits.accSel,
      io.loadReq.bits.rowIdx,
      io.loadReq.bits.data
    )
  }

  // ==========================================================================
  // Store path (sequencer pop — raw BF16 data out of buffer)
  // ==========================================================================

  io.storeData := readBuffer(
    io.storeAddr.accSel,
    io.storeAddr.rowIdx,
    io.storeReadEn
  )
}

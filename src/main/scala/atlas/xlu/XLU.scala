// ============================================================================
// XLU.scala — Cross-Lane Transpose Unit (optimized register-buffer design).
//
// Performs an in-place matrix transpose on one matrix register (mreg).
// The source mreg is read row-by-row into an internal buffer, then the
// transposed result is written column-by-column from the same buffer —
// the transpose is achieved purely by swapping the index order on read-out.
//
// No systolic mesh is needed: the "compute" is just re-indexing.
//
// Matrix register geometry (from spec):
//   MREG_ROWS      = 32   rows
//   MREG_ROW_BYTES = 32   bytes per row  (32 × FP8)
//
// Pipeline stages (non-overlapped, src == dst):
//   1. READ   — stream MREG_ROWS rows into internal buffer  (32 cycles).
//   2. WRITE  — emit transposed rows from buffer             (32 cycles).
//   Total: 64 cycles.
//
// Pipeline stages (overlapped, src != dst):
//   1. READ   — first row read                               (1 cycle).
//   2. READ+WRITE overlap — after first row arrives, begin
//      writing transposed rows that are fully populated while
//      reads continue.  Because row j of the output needs
//      element [j] from every input row, we can only write
//      output row j once ALL input rows have been read.
//      However we pipeline the request/response: write begins
//      the cycle after the last read response.
//      Total: 32 + 32 = 64 cycles (no overlap possible for a
//      full transpose — each output row depends on all inputs).
//
// Bank reporting:
//   activeMregRead  — valid during READ  (srcMreg being read).
//   activeMregWrite — valid during all non-idle states (dstMreg has a
//                     pending write, protecting against RAW/WAW hazards).
// ============================================================================

package atlas.xlu

import atlas.common._
import chisel3._
import chisel3.util._

// ============================================================================
// XluCommand — command bundle for the XLU engine
// ============================================================================

/** XLU command: transpose source mreg into destination mreg. */
class XluCommand(mregIdBits: Int) extends Bundle {
  val op        = UInt(2.W)
  val srcMregId = UInt(mregIdBits.W)
  val dstMregId = UInt(mregIdBits.W)
}

// ============================================================================
// XluEngine — top-level XLU controller (register-buffer transpose)
// ============================================================================

/** Cross-Lane Transpose Unit engine.
  *
  * Reads a full mreg into a register buffer, then writes transposed rows
  * back by swapping the indexing order.  No systolic mesh required.
  *
  * @param mregP  Matrix register file parameters.
  */
class XluEngine(mregP: MregParams) extends Module {

  private val n         = mregP.mregRows                       // 32
  private val elemWidth = mregP.mregRowBits / mregP.mregRows   // 8
  private val nBits     = log2Ceil(n)                          // 5

  val io = IO(new Bundle {
    val cmd = Flipped(Valid(new XluCommand(mregP.mregIdBits)))
    val busy = Output(Bool())

    val mregReadReq  = Valid(new MregReadReq(mregP))
    val mregReadResp = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWriteReq = Valid(new MregWriteReq(mregP))

    val activeMregRead  = Valid(UInt(mregP.mregIdBits.W))
    val activeMregWrite = Valid(UInt(mregP.mregIdBits.W))
  })

  // ==========================================================================
  // FSM
  // ==========================================================================

  object State extends ChiselEnum {
    val Idle, ReadMreg, WriteMreg = Value
  }
  import State._

  val state     = RegInit(Idle)
  val activeCmd = Reg(new XluCommand(mregP.mregIdBits))

  // ==========================================================================
  // Buffer — holds the full source mreg; transposed via index swap on read-out
  // ==========================================================================

  val buf = Reg(Vec(n, Vec(n, UInt(elemWidth.W))))

  // ==========================================================================
  // Counters
  // ==========================================================================

  val readReqCount  = RegInit(0.U((nBits + 1).W))
  val readRespCount = RegInit(0.U((nBits + 1).W))
  val writeIdx      = RegInit(0.U((nBits + 1).W))

  // ==========================================================================
  // Port defaults
  // ==========================================================================

  io.mregReadReq.valid  := false.B
  io.mregReadReq.bits   := 0.U.asTypeOf(new MregReadReq(mregP))
  io.mregWriteReq.valid := false.B
  io.mregWriteReq.bits  := 0.U.asTypeOf(new MregWriteReq(mregP))
  io.busy := state =/= Idle

  // ==========================================================================
  // Active mreg bank reports
  // ==========================================================================

  io.activeMregRead.valid  := (state === ReadMreg)
  io.activeMregRead.bits   := activeCmd.srcMregId

  io.activeMregWrite.valid := (state =/= Idle)
  io.activeMregWrite.bits  := activeCmd.dstMregId

  // ==========================================================================
  // FSM logic
  // ==========================================================================

  switch(state) {

    // ── Idle ─────────────────────────────────────────────────────
    is(Idle) {
      when(io.cmd.valid) {
        activeCmd     := io.cmd.bits
        readReqCount  := 0.U
        readRespCount := 0.U
        state         := ReadMreg
      }
    }

    // ── Read: stream all rows from source mreg into buf ─────────
    is(ReadMreg) {
      when(readReqCount < n.U) {
        io.mregReadReq.valid       := true.B
        io.mregReadReq.bits.mregId := activeCmd.srcMregId
        io.mregReadReq.bits.row    := readReqCount(nBits - 1, 0)
        readReqCount := readReqCount + 1.U
      }

      when(io.mregReadResp.valid) {
        for (i <- 0 until n) {
          buf(readRespCount(nBits - 1, 0))(i) :=
            io.mregReadResp.bits((i + 1) * elemWidth - 1, i * elemWidth)
        }
        readRespCount := readRespCount + 1.U
      }

      when(io.mregReadResp.valid && readRespCount === (n - 1).U) {
        writeIdx := 0.U
        state    := WriteMreg
      }
    }

    // ── Write: emit transposed rows (index-swapped read-out) ────
    //
    // Output row j = { buf[0][j], buf[1][j], ..., buf[n-1][j] }
    // i.e. column j of the original matrix becomes row j of the output.
    is(WriteMreg) {
      io.mregWriteReq.valid       := true.B
      io.mregWriteReq.bits.mregId := activeCmd.dstMregId
      io.mregWriteReq.bits.row    := writeIdx(nBits - 1, 0)

      // Assemble the transposed row: element i of output row j = buf[i][j].
      val transposedRow = Wire(Vec(n, UInt(elemWidth.W)))
      for (i <- 0 until n) {
        transposedRow(i) := buf(i)(writeIdx(nBits - 1, 0))
      }
      io.mregWriteReq.bits.data := transposedRow.asUInt

      writeIdx := writeIdx + 1.U

      when(writeIdx === (n - 1).U) {
        state := Idle
      }
    }
  }
}

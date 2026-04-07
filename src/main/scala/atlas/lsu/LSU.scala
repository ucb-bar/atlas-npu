// ============================================================================
// LSU.scala — Load-Store Unit.
//
// Single serialised gateway to VMEM for all data-path access.  Every VMEM
// read or write in the system goes through this module.
//
// Supported operations:
//
//   VLOAD  (VMEM → mreg)
//     Streams MREG_ROWS lines from VMEM into one matrix register.
//     Latency: MREG_ROWS + 1 cycles (pipeline drain).
//
//   VSTORE (mreg → VMEM)
//     Streams MREG_ROWS rows from one matrix register into VMEM.
//     Latency: MREG_ROWS + 1 cycles (pipeline drain).
//
//   Scalar load
//     Reads one VMEM line, extracts a 32-bit word.  2-cycle latency.
//
//   Scalar store (read-modify-write)
//     Reads the target VMEM line, merges the written bytes using static
//     byte muxing, then writes the full line back.  2-cycle latency.
//     No masked writes — avoids SyncReadMem simulation issues.
//
// Priority (when idle): scalar store > scalar load > matrix command.
//
// Bank reporting:
//   activeMregRead  — valid during VSTORE (mreg → VMEM reads the bank).
//   activeMregWrite — valid during VLOAD  (VMEM → mreg writes the bank).
//   Scalar loads/stores access VMEM only, not mreg banks; no report needed.
// ============================================================================

package atlas.lsu

import chisel3._
import chisel3.util._

import atlas.common._
import atlas.scalar.ScalarISA

// ============================================================================
// Scalar command bundle
// ============================================================================

/** Scalar load/store command to VMEM (32-bit word granularity). */
class LsuScalarCmd(vmemP: VmemParams) extends Bundle {
  val isStore  = Bool()                     // true → store, false → load.
  val byteAddr = UInt(vmemP.byteAddrBits.W) // Byte address within VMEM.
  val wdata    = UInt(32.W)                 // Write data (store only).
  val wmask    = UInt(4.W)                  // Per-byte write mask (store only).
}

// ============================================================================
// LSU module
// ============================================================================

/** Load-Store Unit — serialised VMEM access for matrix and scalar paths.
  *
  * @param vmemP   VMEM geometry parameters.
  * @param mregP   Matrix register file parameters (spec: NUM_MREG, MREG_ROWS, …).
  */
class LSU(vmemP: VmemParams, mregP: MregParams) extends Module {
  import ScalarISA._

  // Convenience constants.
  private val lineOffBits = log2Ceil(vmemP.lineBytes)
  private val mregRows    = mregP.mregRows

  val io = IO(new Bundle {
    /** Matrix-move command from the scalar core (VLOAD / VSTORE). */
    val cmd        = Flipped(Valid(new atlas.scalar.LsuCmd))
    /** Scalar load/store command from the scalar core. */
    val scalarCmd  = Flipped(Valid(new LsuScalarCmd(vmemP)))
    /** Scalar load response (32-bit word). */
    val scalarResp = Valid(UInt(32.W))

    // ── VMEM line ports (all VMEM access flows through here) ────
    val vmemRead     = Valid(new VmemLineReadPort(vmemP))
    val vmemReadData = Flipped(Valid(UInt(vmemP.lineWidthBits.W)))
    val vmemWrite    = Valid(new VmemLineWritePort(vmemP))

    // ── Matrix register file (mreg) ports ───────────────────────
    val mregReadReq  = Valid(new MregReadReq(mregP))
    val mregReadResp = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWriteReq = Valid(new MregWriteReq(mregP))

    /** High while any operation is in progress. */
    val busy = Output(Bool())

    // ── Active mreg bank reports for direction-aware TRF tracking ──
    val activeMregRead  = Valid(UInt(mregP.mregIdBits.W))
    val activeMregWrite = Valid(UInt(mregP.mregIdBits.W))
  })

  // ==========================================================================
  // FSM states
  // ==========================================================================

  val (sIdle :: sMatrixRun :: sMatrixDrain ::
       sScalarLoad :: sScalarStoreRmw :: Nil) = Enum(5)

  val state = RegInit(sIdle)

  // ==========================================================================
  // Matrix-move registers
  // ==========================================================================

  val op       = Reg(UInt(2.W))                              // VLOAD or VSTORE.
  val mregId   = Reg(UInt(mregP.mregIdBits.W))               // Target matrix register.
  val vmemBase = Reg(UInt(vmemP.lineAddrBits.W))              // Starting VMEM line.
  val counter  = RegInit(0.U(log2Ceil(mregRows + 1).W))      // Row/line counter.
  val isLoad   = op === LSU_VLOAD

  // ==========================================================================
  // Scalar-load registers
  // ==========================================================================

  val scalarWordIdx = Reg(UInt(log2Ceil(vmemP.wordsPerLine).W))

  // ==========================================================================
  // Scalar-store RMW registers
  // ==========================================================================

  val storeLineAddr = Reg(UInt(vmemP.lineAddrBits.W))
  val storeWordIdx  = Reg(UInt(log2Ceil(vmemP.wordsPerLine).W))
  val storeData     = Reg(UInt(32.W))
  val storeMask     = Reg(UInt(4.W))

  // ==========================================================================
  // Port defaults
  // ==========================================================================

  io.vmemRead.valid      := false.B
  io.vmemRead.bits.addr  := vmemBase + counter
  io.vmemWrite.valid     := false.B
  io.vmemWrite.bits.addr := 0.U
  io.vmemWrite.bits.data := 0.U

  io.mregReadReq.valid      := false.B
  io.mregReadReq.bits.mregId := mregId
  io.mregReadReq.bits.row   := counter
  io.mregWriteReq.valid      := false.B
  io.mregWriteReq.bits.mregId := mregId
  io.mregWriteReq.bits.row   := 0.U
  io.mregWriteReq.bits.data  := 0.U

  io.scalarResp.valid := false.B
  io.scalarResp.bits  := 0.U

  io.busy := state =/= sIdle

  // ==========================================================================
  // Active mreg bank reports
  // ==========================================================================

  val matrixActive = (state === sMatrixRun || state === sMatrixDrain)
  io.activeMregRead.valid  := matrixActive && !isLoad   // VSTORE reads mreg
  io.activeMregRead.bits   := mregId
  io.activeMregWrite.valid := matrixActive && isLoad    // VLOAD writes mreg
  io.activeMregWrite.bits  := mregId

  // ==========================================================================
  // Matrix pipeline delay tracking
  // ==========================================================================
  //
  // Both VMEM reads and mreg reads have one-cycle latency (SyncReadMem).
  // We pipeline the counter and valid to align writebacks correctly.

  val prevCounter = RegNext(counter)
  val prevValid   = RegNext(state === sMatrixRun, false.B)

  // ==========================================================================
  // FSM
  // ==========================================================================

  switch(state) {

    // ── Idle: accept new commands ────────────────────────────────
    is(sIdle) {
      when(io.scalarCmd.valid && io.scalarCmd.bits.isStore) {
        // Scalar store: begin read-modify-write — read the target line first.
        val byteAddr = io.scalarCmd.bits.byteAddr
        io.vmemRead.valid     := true.B
        io.vmemRead.bits.addr := byteAddr(vmemP.byteAddrBits - 1, lineOffBits)
        storeLineAddr := byteAddr(vmemP.byteAddrBits - 1, lineOffBits)
        storeWordIdx  := byteAddr(lineOffBits - 1, 2)
        storeData     := io.scalarCmd.bits.wdata
        storeMask     := io.scalarCmd.bits.wmask
        state := sScalarStoreRmw

      }.elsewhen(io.scalarCmd.valid && !io.scalarCmd.bits.isStore) {
        // Scalar load: issue a single-line VMEM read.
        val byteAddr = io.scalarCmd.bits.byteAddr
        io.vmemRead.valid     := true.B
        io.vmemRead.bits.addr := byteAddr(vmemP.byteAddrBits - 1, lineOffBits)
        scalarWordIdx         := byteAddr(lineOffBits - 1, 2)
        state := sScalarLoad

      }.elsewhen(io.cmd.valid) {
        // Matrix move: capture command and begin streaming.
        state    := sMatrixRun
        op       := io.cmd.bits.op
        mregId   := io.cmd.bits.mregBank
        vmemBase := io.cmd.bits.vmemLineAddr
        counter  := 0.U
      }
    }

    // ── Scalar load: extract a 32-bit word from the returned line ─
    is(sScalarLoad) {
      val lineData  = io.vmemReadData.bits
      val lineWords = Wire(Vec(vmemP.wordsPerLine, UInt(32.W)))
      for (i <- 0 until vmemP.wordsPerLine) {
        lineWords(i) := lineData(32 * i + 31, 32 * i)
      }
      io.scalarResp.valid := true.B
      io.scalarResp.bits  := lineWords(scalarWordIdx)
      state := sIdle
    }

    // ── Scalar store: read-modify-write ──────────────────────────
    //
    // Merge the store bytes into the old line using *static* byte muxing.
    // For each of the lineBytes bytes, check whether it belongs to the
    // target word AND its mask bit is set.  If so, take from storeData;
    // otherwise keep the original byte.
    is(sScalarStoreRmw) {
      val oldLine = io.vmemReadData.bits

      // Pre-compute which word index matches (one comparator per word).
      val isTargetWord = VecInit(
        (0 until vmemP.wordsPerLine).map(w => storeWordIdx === w.U)
      )

      val newLineBytes = Wire(Vec(vmemP.lineBytes, UInt(8.W)))
      for (i <- 0 until vmemP.lineBytes) {
        val wordOfByte = i / 4           // Which 32-bit word this byte belongs to.
        val byteInWord = i % 4           // Position within that word.
        val replace    = isTargetWord(wordOfByte) && storeMask(byteInWord)
        val storeByte  = storeData(byteInWord * 8 + 7, byteInWord * 8)
        val oldByte    = oldLine(i * 8 + 7, i * 8)
        newLineBytes(i) := Mux(replace, storeByte, oldByte)
      }

      io.vmemWrite.valid     := true.B
      io.vmemWrite.bits.addr := storeLineAddr
      io.vmemWrite.bits.data := Cat(newLineBytes.reverse)
      state := sIdle
    }

    // ── Matrix run: stream one row/line per cycle ────────────────
    is(sMatrixRun) {
      when(isLoad) {
        // VLOAD: read from VMEM (write-back to mreg happens in the
        //        delayed pipeline below).
        io.vmemRead.valid     := true.B
        io.vmemRead.bits.addr := vmemBase + counter
      }.otherwise {
        // VSTORE: read from mreg (write to VMEM happens in the
        //         delayed pipeline below).
        io.mregReadReq.valid      := true.B
        io.mregReadReq.bits.mregId := mregId
        io.mregReadReq.bits.row   := counter
      }

      counter := counter + 1.U
      when(counter === (mregRows - 1).U) {
        state := sMatrixDrain
      }
    }

    // ── Matrix drain: wait one cycle for the last pipeline beat ──
    is(sMatrixDrain) {
      state := sIdle
    }
  }

  // ==========================================================================
  // Matrix write pipeline (one-cycle delayed from the read)
  // ==========================================================================
  //
  // During VLOAD:  VMEM data → mreg write  (aligned to prevCounter).
  // During VSTORE: mreg data → VMEM write  (aligned to prevCounter).

  when(prevValid || (state === sMatrixDrain)) {
    when(isLoad) {
      io.mregWriteReq.valid      := io.vmemReadData.valid
      io.mregWriteReq.bits.mregId := mregId
      io.mregWriteReq.bits.row   := prevCounter
      io.mregWriteReq.bits.data  := io.vmemReadData.bits
    }.otherwise {
      io.vmemWrite.valid     := io.mregReadResp.valid
      io.vmemWrite.bits.addr := vmemBase + prevCounter
      io.vmemWrite.bits.data := io.mregReadResp.bits
    }
  }
}

/*
LSU.scala — Load-Store Unit. Single gateway to VMEM for all data access.

Handles:
  - VLOAD:  VMEM -> TRF  (32 lines, 33 cycles)
  - VSTORE: TRF -> VMEM  (32 lines, 33 cycles)
  - Scalar load:  read one VMEM line, extract 32-bit word (2 cycles)
  - Scalar store: read-modify-write one VMEM line (2 cycles)
    Reads full line, merges target bytes using static muxing, writes
    full line back. No masked writes — avoids SyncReadMem simulation issues.

All VMEM access is serialized through the LSU.
*/

package atlas.tile

import chisel3._
import chisel3.util._

import atlas.common._
import atlas.scalar.ScalarISA

class LsuScalarCmd(spP: ScratchpadParams) extends Bundle {
  val isStore  = Bool()
  val byteAddr = UInt(spP.byteAddrBits.W)
  val wdata    = UInt(32.W)
  val wmask    = UInt(4.W)
}

class LSU(spP: ScratchpadParams, rfP: RegFileParams) extends Module {
  import ScalarISA._

  val io = IO(new Bundle {
    val cmd        = Flipped(Valid(new atlas.scalar.LsuCmd))
    val scalarCmd  = Flipped(Valid(new LsuScalarCmd(spP)))
    val scalarResp = Valid(UInt(32.W))

    // VMEM line ports (ALL VMEM access uses these)
    val vmemRead     = Valid(new ScratchpadLineReadPort(spP))
    val vmemReadData = Flipped(Valid(UInt(spP.lineWidth.W)))
    val vmemWrite    = Valid(new ScratchpadLineWritePort(spP))

    // TRF ports
    val trfRead     = Valid(new RegFileReadInput(rfP))
    val trfReadData = Flipped(Valid(UInt(rfP.SRAM_WIDTH.W)))
    val trfWrite    = Valid(new RegFileWriteInput(rfP))

    val busy = Output(Bool())
  })

  val sIdle :: sTensorRun :: sTensorDrain :: sScalarLoad :: sScalarStoreRMW :: Nil = Enum(5)
  val state = RegInit(sIdle)

  // Tensor state
  val op       = Reg(UInt(2.W))
  val trfBank  = Reg(UInt(rfP.NUM_BANKS_BITS.W))
  val vmemBase = Reg(UInt(spP.lineAddrBits.W))
  val counter  = RegInit(0.U(log2Ceil(rfP.SRAM_DEPTH + 1).W))
  val isLoad   = op === LSU_VLOAD

  // Scalar load state
  val scalarWordIdx = Reg(UInt(log2Ceil(spP.wordsPerLine).W))

  // Store RMW state
  val storeLineAddr = Reg(UInt(spP.lineAddrBits.W))
  val storeWordIdx  = Reg(UInt(log2Ceil(spP.wordsPerLine).W))
  val storeData     = Reg(UInt(32.W))
  val storeMask     = Reg(UInt(4.W))

  val depth = rfP.SRAM_DEPTH.U
  private val lineOffBits = log2Ceil(spP.lineBytes)

  // ── Defaults ────────────────────────────────────────────────────
  io.vmemRead.valid      := false.B
  io.vmemRead.bits.addr  := vmemBase + counter
  io.vmemWrite.valid     := false.B
  io.vmemWrite.bits.addr := 0.U
  io.vmemWrite.bits.data := 0.U

  io.trfRead.valid          := false.B
  io.trfRead.bits.whichBank := trfBank
  io.trfRead.bits.rRow      := counter
  io.trfWrite.valid          := false.B
  io.trfWrite.bits.whichBank := trfBank
  io.trfWrite.bits.wRow      := 0.U
  io.trfWrite.bits.wData     := 0.U

  io.scalarResp.valid := false.B
  io.scalarResp.bits  := 0.U

  io.busy := state =/= sIdle

  // Tensor pipeline delay
  val prevCounter = RegNext(counter)
  val prevValid   = RegNext(state === sTensorRun, false.B)

  // ── State transitions ───────────────────────────────────────────
  switch(state) {
    is(sIdle) {
      when(io.scalarCmd.valid && io.scalarCmd.bits.isStore) {
        // Scalar store: start RMW — read the target line
        val byteAddr = io.scalarCmd.bits.byteAddr
        io.vmemRead.valid     := true.B
        io.vmemRead.bits.addr := byteAddr(spP.byteAddrBits - 1, lineOffBits)
        storeLineAddr := byteAddr(spP.byteAddrBits - 1, lineOffBits)
        storeWordIdx  := byteAddr(lineOffBits - 1, 2)
        storeData     := io.scalarCmd.bits.wdata
        storeMask     := io.scalarCmd.bits.wmask
        state := sScalarStoreRMW
      }.elsewhen(io.scalarCmd.valid && !io.scalarCmd.bits.isStore) {
        // Scalar load: issue line read
        val byteAddr = io.scalarCmd.bits.byteAddr
        io.vmemRead.valid     := true.B
        io.vmemRead.bits.addr := byteAddr(spP.byteAddrBits - 1, lineOffBits)
        scalarWordIdx         := byteAddr(lineOffBits - 1, 2)
        state := sScalarLoad
      }.elsewhen(io.cmd.valid) {
        state    := sTensorRun
        op       := io.cmd.bits.op
        trfBank  := io.cmd.bits.trfBank
        vmemBase := io.cmd.bits.vmemLineAddr
        counter  := 0.U
      }
    }

    is(sScalarLoad) {
      val lineData  = io.vmemReadData.bits
      val lineWords = Wire(Vec(spP.wordsPerLine, UInt(32.W)))
      for (i <- 0 until spP.wordsPerLine) {
        lineWords(i) := lineData(32 * i + 31, 32 * i)
      }
      io.scalarResp.valid := true.B
      io.scalarResp.bits  := lineWords(scalarWordIdx)
      state := sIdle
    }

    is(sScalarStoreRMW) {
      // ── Read-modify-write: merge store bytes into old line ────
      // Uses STATIC byte-by-byte muxing — no dynamic Vec indexing.
      // For each of the 32 bytes in the line, check if it belongs to
      // the target word AND has its mask bit set. If so, take from
      // storeData; otherwise keep from oldLine.
      val oldLine = io.vmemReadData.bits

      // Pre-compute which word index matches (8 comparators)
      val isTargetWord = VecInit(
        (0 until spP.wordsPerLine).map(w => storeWordIdx === w.U)
      )

      val newLineBytes = Wire(Vec(spP.lineBytes, UInt(8.W)))
      for (i <- 0 until spP.lineBytes) {
        val wordOfByte  = i / 4              // Scala Int: which word (0-7)
        val byteInWord  = i % 4              // Scala Int: which byte in word (0-3)
        val replace     = isTargetWord(wordOfByte) && storeMask(byteInWord)
        val storeByte   = storeData(byteInWord * 8 + 7, byteInWord * 8)
        val oldByte     = oldLine(i * 8 + 7, i * 8)
        newLineBytes(i) := Mux(replace, storeByte, oldByte)
      }

      io.vmemWrite.valid     := true.B
      io.vmemWrite.bits.addr := storeLineAddr
      io.vmemWrite.bits.data := Cat(newLineBytes.reverse)

      state := sIdle
    }

    is(sTensorRun) {
      when(isLoad) {
        io.vmemRead.valid     := true.B
        io.vmemRead.bits.addr := vmemBase + counter
      }.otherwise {
        io.trfRead.valid          := true.B
        io.trfRead.bits.whichBank := trfBank
        io.trfRead.bits.rRow      := counter
      }
      counter := counter + 1.U
      when(counter === (depth - 1.U)) {
        state := sTensorDrain
      }
    }

    is(sTensorDrain) {
      state := sIdle
    }
  }

  // ── Tensor write pipeline (1-cycle delayed) ─────────────────────
  when(prevValid || (state === sTensorDrain)) {
    when(isLoad) {
      io.trfWrite.valid          := io.vmemReadData.valid
      io.trfWrite.bits.whichBank := trfBank
      io.trfWrite.bits.wRow      := prevCounter
      io.trfWrite.bits.wData     := io.vmemReadData.bits
    }.otherwise {
      io.vmemWrite.valid     := io.trfReadData.valid
      io.vmemWrite.bits.addr := vmemBase + prevCounter
      io.vmemWrite.bits.data := io.trfReadData.bits
    }
  }
}

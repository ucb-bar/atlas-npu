// ============================================================================
// AtlasCore.scala — Non-diplomatic top-level core.
//
// Data-path topology:
//   System bus ←TL→ DMA ←line port→ VMEM
//   System bus ←TL→ VMEM (host direct access, lowest priority)
//   VMEM ←line read/write→ LSU ←row port→ MREG (mreg file)
//   LSU ←scalar cmd/resp→ ScalarCore  (all VMEM access is serialised)
//   Host ←TL→ IMEM,  Host ←TL→ CSR
//
// MREG bank hazard tracking:
//   All engines report which mreg banks they are actively reading / writing.
//   A MregBankTracker aggregates these into readBusy / writeBusy bitvectors
//   consumed by the scalar core for direction-aware stall decisions.
//   Pending MXU/XLU commands (captured but not yet accepted by the engine)
//   also contribute their banks so no tracking gap exists.
//
// DMA busy bridging:
//   A per-channel `dmaLaunched` flag bridges the gap between the command
//   handshake (`command.fire`) and the DMA engine asserting `channelBusy`.
//   The flag is set on fire and cleared only when `channelBusy` rises,
//   so `dma_busy` fed to the scalar core has no momentary false dip
//   regardless of engine startup latency.
// ============================================================================

package atlas.tile

import chisel3._
import chisel3.util._

import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters}

import atlas.common._
import atlas.scalar.{ScalarCore, ScalarISA, EngineStatus, AtlasMemMap,
                      CSRFile, LsuCmd, InstrMem, ImemFetchPort, DmaCmd,
                      MxuCmd => ScalarMxuCmd, VpuCmd, XluCmd,
                      MregBankTracker}
import atlas.ipt.InnerProductTreesTop
import atlas.sa.SystolicArrayTop
import atlas.mxu.{MxuOp, MxuCmd}
import atlas.dma.{DmaEngine, DmaCommand, DmaDirection}
import atlas.xlu.{XluEngine, XluCommand}
import atlas.lsu.LSU
import atlas.mreg.MregFile
import atlas.vmem.VMEM

// ============================================================================
// Stub engines (VPU is not yet implemented)
// ============================================================================

/** Placeholder VPU configuration bundle. */
class VpuConfig extends Bundle {
  val op      = UInt(8.W)
  val srcBank = UInt(8.W)
  val dstBank = UInt(8.W)
  val len     = UInt(16.W)
}

/** No-op VPU stub — accepts commands and does nothing.
  *
  * @param mregP  MREG geometry for the exposed register-file ports.
  */
class VpuEngine(mregP: MregParams) extends Module {
  val io = IO(new Bundle {
    val cmd       = Flipped(Decoupled(new VpuConfig))
    val mregRead0  = Valid(new MregReadReq(mregP))
    val mregRead1  = Valid(new MregReadReq(mregP))
    val mregReadData0 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregReadData1 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWrite0 = Valid(new MregWriteReq(mregP))
    val mregWrite1 = Valid(new MregWriteReq(mregP))
    val busy      = Output(Bool())
  })
  io.cmd.ready       := true.B
  io.mregRead0.valid  := false.B; io.mregRead0.bits  := 0.U.asTypeOf(io.mregRead0.bits)
  io.mregRead1.valid  := false.B; io.mregRead1.bits  := 0.U.asTypeOf(io.mregRead1.bits)
  io.mregWrite0.valid := false.B; io.mregWrite0.bits := 0.U.asTypeOf(io.mregWrite0.bits)
  io.mregWrite1.valid := false.B; io.mregWrite1.bits := 0.U.asTypeOf(io.mregWrite1.bits)
  io.busy            := false.B
}

// ============================================================================
// AtlasCore — top-level non-diplomatic module
// ============================================================================

/** Instantiates every functional unit and wires them together.
  *
  * Externally exposes three TileLink ports (IMEM, CSR, DMA), a `halted`
  * flag, and a debug bundle.
  *
  * @param tp      Full set of Atlas design parameters.
  * @param imemBP  TileLink bundle parameters for the instruction-memory port.
  * @param csrBP   TileLink bundle parameters for the CSR port.
  * @param dmaBP   TileLink bundle parameters for the DMA port.
  * @param vmemBP  TileLink bundle parameters for the VMEM slave port.
  */
class AtlasCore(
  tp:     AtlasParams        = AtlasParams(),
  imemBP: TLBundleParameters,
  csrBP:  TLBundleParameters,
  dmaBP:  TLBundleParameters,
  vmemBP: TLBundleParameters
) extends Module {

  private val iptP  = tp.ipt
  private val mregP = tp.mreg
  private val dmaP  = tp.dma
  private val vmemP = tp.vmem
  private val saP   = tp.sa
  import ScalarISA._

  // ---------- IO ----------

  val io = IO(new Bundle {
    val imemTL = Flipped(new TLBundle(imemBP))
    val csrTL  = Flipped(new TLBundle(csrBP))
    val dmaTL  = new TLBundle(dmaBP)
    val vmemTL = Flipped(new TLBundle(vmemBP))
    val halted = Output(Bool())
    val dbg = Output(new Bundle {
      val mxu0DataBusy = Bool()
      val mxu0CompBusy = Bool()
      val mxu1DataBusy = Bool()
      val mxu1CompBusy = Bool()
      val xluBusy      = Bool()
      val dmaBusy      = Vec(dmaP.numChannels, Bool())
      val lsuBusy      = Bool()
      val scaleRegs    = Vec(ScalarISA.NUM_SCALE_REGS, UInt(8.W))
      val dbg0         = UInt(32.W)
      val dbg1         = UInt(32.W)
    })
  })

  // ==========================================================================
  // Sub-module instantiation
  // ==========================================================================

  val imem    = Module(new InstrMem(imemBP))
  val csrfile = Module(new CSRFile(csrBP))
  val vmem    = Module(new VMEM(vmemP, vmemBP))
  val scalar  = Module(new ScalarCore(vmemP))
  val dma     = Module(new DmaEngine(tp, dmaBP))
  val lsu     = Module(new LSU(vmemP, mregP))
  val mxu0    = Module(new SystolicArrayTop(saP, mregP))
  val mxu1    = Module(new InnerProductTreesTop(iptP, mregP))
  val vpu     = Module(new VpuEngine(mregP))
  val xlu     = Module(new XluEngine(mregP))
  val mreg    = Module(new MregFile(mregP))

  // ==========================================================================
  // TileLink port connections
  // ==========================================================================

  imem.io.fetch <> scalar.io.imemFetch
  imem.io.tl    <> io.imemTL
  csrfile.io.tl <> io.csrTL
  io.dmaTL      <> dma.io.tl
  io.halted     := scalar.io.halted
  vmem.io.tl    <> io.vmemTL

  // ==========================================================================
  // CSR interface
  // ==========================================================================

  csrfile.io.csr <> scalar.io.csrPort
  scalar.io.softReset := csrfile.io.softReset

  // ==========================================================================
  // DMA ↔ VMEM (line-granularity ports)
  // ==========================================================================

  vmem.io.dmaRead     <> dma.io.vmemRead
  dma.io.vmemReadData := vmem.io.dmaReadData
  vmem.io.dmaWrite    <> dma.io.vmemWrite

  // ==========================================================================
  // LSU ↔ VMEM (line-granularity read / write)
  // ==========================================================================

  vmem.io.lsuRead     <> lsu.io.vmemRead
  lsu.io.vmemReadData := vmem.io.lsuReadData
  vmem.io.lsuWrite    <> lsu.io.vmemWrite

  // ==========================================================================
  // LSU ↔ MREG (mreg row-level access)
  // ==========================================================================

  mreg.io.lsuReadReq   <> lsu.io.mregReadReq
  lsu.io.mregReadResp  := mreg.io.lsuReadResp
  mreg.io.lsuWriteReq  <> lsu.io.mregWriteReq

  // ==========================================================================
  // ScalarCore ↔ LSU (scalar memory + matrix-move commands)
  // ==========================================================================

  lsu.io.scalarCmd        <> scalar.io.scalarMemCmd
  scalar.io.scalarMemResp := lsu.io.scalarResp
  lsu.io.cmd              := scalar.io.lsuCmd
  scalar.io.lsu_busy      := lsu.io.busy

  // ==========================================================================
  // Utility: build an EngineStatus from a single busy signal
  // ==========================================================================

  def makeStatus(busy: Bool): EngineStatus = {
    val s = Wire(new EngineStatus)
    s.busy  := busy
    s.done  := !busy
    s.error := false.B
    s
  }

  // ==========================================================================
  // DMA command dispatch (ScalarCore → DmaEngine)
  // ==========================================================================

  val dmaPending    = RegInit(false.B)
  val dmaCommandReg = Reg(new DmaCommand(vmemP, dmaP))

  // Per-channel launched flags: set on command.fire, cleared when channelBusy
  // rises.  Bridges any latency gap so dma_busy never has a false dip.
  val dmaLaunched = RegInit(VecInit(Seq.fill(dmaP.numChannels)(false.B)))

  for (ch <- 0 until dmaP.numChannels) {
    when(dma.io.channelBusy(ch)) {
      dmaLaunched(ch) := false.B
    }
  }
  when(dma.io.command.fire) {
    dmaLaunched(dmaCommandReg.channelId) := true.B  // last-connect wins over clear
  }

  for (ch <- 0 until dmaP.numChannels) {
    scalar.io.dma_busy(ch) := dma.io.channelBusy(ch) ||
      (dmaPending && dmaCommandReg.channelId === ch.U) ||
      dmaLaunched(ch)
  }

  private val wordOffBits = vmemP.wordOffBits

  when(scalar.io.dmaCmd.valid) {
    assert(!dmaPending, "DMA command issued while a prior command is still pending")

    val cmd = scalar.io.dmaCmd.bits
    dmaPending := true.B

    dmaCommandReg.opType       := Mux(cmd.op === DMA_LD,
                                       DmaDirection.LoadToVmem,
                                       DmaDirection.StoreFromVmem)
    dmaCommandReg.channelId    := cmd.channel
    dmaCommandReg.vmemLineAddr := cmd.vmemAddr(vmemP.wordAddrBits - 1, wordOffBits)
    dmaCommandReg.dramAddress  := cmd.addr
    dmaCommandReg.transferSize := cmd.size(dmaP.transferSizeBits - 1, 0)
  }

  dma.io.command.valid := dmaPending
  dma.io.command.bits  := dmaCommandReg
  when(dma.io.command.fire) { dmaPending := false.B }

  // ==========================================================================
  // MXU-0 command dispatch (ScalarCore → SystolicArray)
  // ==========================================================================

  val mxu0Pending = RegInit(false.B)
  val mxu0CmdReg  = Reg(new MxuCmd(mregP.mregIdBits))

  when(scalar.io.mxu0Cmd.valid) {
    assert(!mxu0Pending, "MXU0 command issued while a prior command is still pending")
    mxu0Pending := true.B

    val sc = scalar.io.mxu0Cmd.bits
    mxu0CmdReg.op         := MxuOp.safe((sc.op - 1.U)(2, 0))._1
    mxu0CmdReg.mregId     := sc.mregBank
    mxu0CmdReg.accSel     := sc.accSel
    mxu0CmdReg.weightSlot := sc.weightSlot
    mxu0CmdReg.scaleE8M0  := sc.scaleE8M0
  }

  mxu0.io.cmd.valid := mxu0Pending
  mxu0.io.cmd.bits  := mxu0CmdReg
  when(mxu0.io.cmd.fire) { mxu0Pending := false.B }

  // ==========================================================================
  // MXU-1 command dispatch (ScalarCore → InnerProductTrees)
  // ==========================================================================

  val mxu1Pending = RegInit(false.B)
  val mxu1CmdReg  = Reg(new MxuCmd(mregP.mregIdBits))

  when(scalar.io.mxu1Cmd.valid) {
    assert(!mxu1Pending, "MXU1 command issued while a prior command is still pending")
    mxu1Pending := true.B

    val sc = scalar.io.mxu1Cmd.bits
    mxu1CmdReg.op         := MxuOp.safe((sc.op - 1.U)(2, 0))._1
    mxu1CmdReg.mregId     := sc.mregBank
    mxu1CmdReg.accSel     := sc.accSel
    mxu1CmdReg.weightSlot := sc.weightSlot
    mxu1CmdReg.scaleE8M0  := sc.scaleE8M0
  }

  mxu1.io.cmd.valid := mxu1Pending
  mxu1.io.cmd.bits  := mxu1CmdReg
  when(mxu1.io.cmd.fire) { mxu1Pending := false.B }

  // ==========================================================================
  // XLU command dispatch (ScalarCore → XluEngine)
  // ==========================================================================

  val xluPending = RegInit(false.B)
  val xluCmdReg  = Reg(new XluCommand(mregP.mregIdBits))

  when(scalar.io.xluCmd.valid) {
    assert(!xluPending, "XLU command issued while a prior command is still pending")
    xluPending := true.B

    val sc = scalar.io.xluCmd.bits
    xluCmdReg.op        := sc.op
    xluCmdReg.srcMregId := sc.srcBank
    xluCmdReg.dstMregId := sc.dstBank
  }

  xlu.io.cmd.valid := xluPending
  xlu.io.cmd.bits  := xluCmdReg
  when(xluPending && !xlu.io.busy) {
    xluPending := false.B  // XLU was idle and accepted the command
  }

  // ==========================================================================
  // Engine command backpressure to scalar core
  // ==========================================================================

  scalar.io.mxu0_cmd_ready := !mxu0Pending
  scalar.io.mxu1_cmd_ready := !mxu1Pending
  scalar.io.xlu_cmd_ready  := !xluPending

  // ==========================================================================
  // Engine status feedback to the scalar core (VPU only)
  // ==========================================================================

  scalar.io.vpu_status := makeStatus(vpu.io.busy)

  // ==========================================================================
  // MREG bank tracker — direction-aware hazard detection
  // ==========================================================================
  //
  // Collects read/write bank reports from all engines plus pending MXU/XLU
  // commands and aggregates them into readBusy / writeBusy bitvectors.

  private val numBanks   = 64
  private val bankBits   = log2Ceil(numBanks)
  val mregTracker = Module(new MregBankTracker(numBanks, numReaders = 10, numWriters = 10))

  // Helper: decode a pending MxuCmd to determine which banks it reads/writes.
  private def pendingMxuBanks(
      pending: Bool,
      cmd:     MxuCmd,
      reads:   Vec[Valid[UInt]],  // length 2
      writes:  Vec[Valid[UInt]]   // length 2
  ): Unit = {
    reads(0).valid  := false.B; reads(0).bits  := 0.U
    reads(1).valid  := false.B; reads(1).bits  := 0.U
    writes(0).valid := false.B; writes(0).bits := 0.U
    writes(1).valid := false.B; writes(1).bits := 0.U

    when(pending) {
      val isRead = (cmd.op === MxuOp.PushWeight  || cmd.op === MxuOp.PushAccFP8 ||
                    cmd.op === MxuOp.PushAccBF16  || cmd.op === MxuOp.Matmul ||
                    cmd.op === MxuOp.MatmulAcc)
      val isWrite = (cmd.op === MxuOp.PopAccFP8 || cmd.op === MxuOp.PopAccBF16)
      val isBF16Read  = (cmd.op === MxuOp.PushAccBF16)
      val isBF16Write = (cmd.op === MxuOp.PopAccBF16)

      when(isRead) {
        reads(0).valid := true.B
        reads(0).bits  := cmd.mregId
        when(isBF16Read) {
          reads(1).valid := true.B
          reads(1).bits  := cmd.mregId + 1.U
        }
      }
      when(isWrite) {
        writes(0).valid := true.B
        writes(0).bits  := cmd.mregId
        when(isBF16Write) {
          writes(1).valid := true.B
          writes(1).bits  := cmd.mregId + 1.U
        }
      }
    }
  }

  val mxu0PendReads  = Wire(Vec(2, Valid(UInt(bankBits.W))))
  val mxu0PendWrites = Wire(Vec(2, Valid(UInt(bankBits.W))))
  pendingMxuBanks(mxu0Pending, mxu0CmdReg, mxu0PendReads, mxu0PendWrites)

  val mxu1PendReads  = Wire(Vec(2, Valid(UInt(bankBits.W))))
  val mxu1PendWrites = Wire(Vec(2, Valid(UInt(bankBits.W))))
  pendingMxuBanks(mxu1Pending, mxu1CmdReg, mxu1PendReads, mxu1PendWrites)

  // Reader port assignment (10 ports):
  //   0-1: MXU0 active reads
  //   2-3: MXU1 active reads
  //   4:   XLU active read
  //   5:   XLU pending read (srcBank of command not yet accepted)
  //   6-7: MXU0 pending reads
  //   8-9: MXU1 pending reads
  mregTracker.io.readers(0) := mxu0.io.activeReads(0)
  mregTracker.io.readers(1) := mxu0.io.activeReads(1)
  mregTracker.io.readers(2) := mxu1.io.activeReads(0)
  mregTracker.io.readers(3) := mxu1.io.activeReads(1)
  mregTracker.io.readers(4) := xlu.io.activeMregRead
  mregTracker.io.readers(5).valid := xluPending
  mregTracker.io.readers(5).bits  := xluCmdReg.srcMregId
  mregTracker.io.readers(6) := mxu0PendReads(0)
  mregTracker.io.readers(7) := mxu0PendReads(1)
  mregTracker.io.readers(8) := mxu1PendReads(0)
  mregTracker.io.readers(9) := mxu1PendReads(1)

  // Writer port assignment (10 ports):
  //   0-1: MXU0 active writes
  //   2-3: MXU1 active writes
  //   4:   XLU active write
  //   5:   XLU pending write (dstBank of command not yet accepted)
  //   6-7: MXU0 pending writes
  //   8-9: MXU1 pending writes
  mregTracker.io.writers(0) := mxu0.io.activeWrites(0)
  mregTracker.io.writers(1) := mxu0.io.activeWrites(1)
  mregTracker.io.writers(2) := mxu1.io.activeWrites(0)
  mregTracker.io.writers(3) := mxu1.io.activeWrites(1)
  mregTracker.io.writers(4) := xlu.io.activeMregWrite
  mregTracker.io.writers(5).valid := xluPending
  mregTracker.io.writers(5).bits  := xluCmdReg.dstMregId
  mregTracker.io.writers(6) := mxu0PendWrites(0)
  mregTracker.io.writers(7) := mxu0PendWrites(1)
  mregTracker.io.writers(8) := mxu1PendWrites(0)
  mregTracker.io.writers(9) := mxu1PendWrites(1)

  scalar.io.mregReadBusy  := mregTracker.io.readBusy
  scalar.io.mregWriteBusy := mregTracker.io.writeBusy

  // ==========================================================================
  // VPU command forwarding
  // ==========================================================================

  vpu.io.cmd.valid := scalar.io.vpuCmd.valid
  vpu.io.cmd.bits  := 0.U.asTypeOf(new VpuConfig)

  // ==========================================================================
  // Matrix Register File (mreg) wiring
  // ==========================================================================

  // ── MXU-1 ──
  mreg.io.mxu1ReadReq0    := mxu1.io.mregReadReq0
  mxu1.io.mregReadResp0   := mreg.io.mxu1ReadResp0
  mreg.io.mxu1ReadReq1    := mxu1.io.mregReadReq1
  mxu1.io.mregReadResp1   := mreg.io.mxu1ReadResp1
  mreg.io.mxu1WriteReq0   := mxu1.io.mregWriteReq0
  mreg.io.mxu1WriteReq1   := mxu1.io.mregWriteReq1

  // ── MXU-0 ──
  mreg.io.mxu0ReadReq0    := mxu0.io.mregReadReq0
  mreg.io.mxu0ReadReq1    := mxu0.io.mregReadReq1
  mxu0.io.mregReadResp0   := mreg.io.mxu0ReadResp0
  mxu0.io.mregReadResp1   := mreg.io.mxu0ReadResp1
  mreg.io.mxu0WriteReq0   := mxu0.io.mregWriteReq0
  mreg.io.mxu0WriteReq1   := mxu0.io.mregWriteReq1

  // ── VPU ──
  mreg.io.vpuReadReq0     := vpu.io.mregRead0
  mreg.io.vpuReadReq1     := vpu.io.mregRead1
  vpu.io.mregReadData0     := mreg.io.vpuReadResp0
  vpu.io.mregReadData1     := mreg.io.vpuReadResp1
  mreg.io.vpuWriteReq0    := vpu.io.mregWrite0
  mreg.io.vpuWriteReq1    := vpu.io.mregWrite1

  // ── XLU ──
  mreg.io.xluReadReq         := xlu.io.mregReadReq
  xlu.io.mregReadResp.valid := mreg.io.xluReadResp.valid
  xlu.io.mregReadResp.bits  := mreg.io.xluReadResp.bits
  mreg.io.xluWriteReq        := xlu.io.mregWriteReq

  // ==========================================================================
  // Debug outputs
  // ==========================================================================

  io.dbg.mxu0DataBusy := mxu0.io.dataBusy
  io.dbg.mxu0CompBusy := mxu0.io.computeBusy
  io.dbg.mxu1DataBusy := mxu1.io.dataBusy
  io.dbg.mxu1CompBusy := mxu1.io.computeBusy
  io.dbg.xluBusy      := xlu.io.busy
  io.dbg.dmaBusy      := dma.io.channelBusy
  io.dbg.lsuBusy      := lsu.io.busy
  io.dbg.scaleRegs    := scalar.io.scaleRegs
  io.dbg.dbg0         := csrfile.io.dbg0
  io.dbg.dbg1         := csrfile.io.dbg1
}

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
// Software-scheduled command dispatch:
//   All engine commands (MXU, XLU, DMA, VPU) are dispatched directly from
//   the scalar core to the engines with no intermediate pending registers
//   or backpressure.  Software guarantees that engines are ready to accept
//   commands when they are issued.  Structural hazard assertions live in
//   the engines/sequencers themselves.
//
// MREG bank hazard tracking:
//   All engines report which mreg banks they are actively reading / writing.
//   A MregBankTracker aggregates these into readBusy / writeBusy bitvectors
//   consumed by the scalar core for direction-aware hazard assertions.
//
// DMA busy bridging:
//   A per-channel `dmaLaunched` flag bridges the gap between the command
//   issue and the DMA engine asserting `channelBusy`.  The flag is set on
//   issue and cleared only when `channelBusy` rises, so `dma_busy` fed to
//   the scalar core has no momentary false dip.
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

/** No-op VPU stub — accepts commands and does nothing.
  *
  * @param mregP  MREG geometry for the exposed register-file ports.
  */
class VpuEngine(mregP: MregParams) extends Module {
  val io = IO(new Bundle {
    val cmd       = Flipped(Valid(new VpuCmd))
    val mregRead0  = Valid(new MregReadReq(mregP))
    val mregRead1  = Valid(new MregReadReq(mregP))
    val mregReadData0 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregReadData1 = Flipped(Valid(UInt(mregP.mregRowBits.W)))
    val mregWrite0 = Valid(new MregWriteReq(mregP))
    val mregWrite1 = Valid(new MregWriteReq(mregP))
    val busy      = Output(Bool())
  })
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
  // DMA command dispatch (ScalarCore → DmaEngine, combinational)
  // ==========================================================================

  private val wordOffBits = vmemP.wordOffBits

  val dmaCmdWire = Wire(new DmaCommand(vmemP, dmaP))
  val scDma      = scalar.io.dmaCmd.bits
  dmaCmdWire.opType       := Mux(scDma.op === DMA_LD,
                                  DmaDirection.LoadToVmem,
                                  DmaDirection.StoreFromVmem)
  dmaCmdWire.channelId    := scDma.channel
  dmaCmdWire.vmemLineAddr := scDma.vmemAddr(vmemP.wordAddrBits - 1, wordOffBits)
  dmaCmdWire.dramAddress  := scDma.addr
  dmaCmdWire.transferSize := scDma.size(dmaP.transferSizeBits - 1, 0)

  dma.io.command.valid := scalar.io.dmaCmd.valid
  dma.io.command.bits  := dmaCmdWire

  // Per-channel launched flags: set on command issue, cleared when channelBusy
  // rises.  Bridges any latency gap so dma_busy never has a false dip.
  val dmaLaunched = RegInit(VecInit(Seq.fill(dmaP.numChannels)(false.B)))

  for (ch <- 0 until dmaP.numChannels) {
    when(dma.io.channelBusy(ch)) {
      dmaLaunched(ch) := false.B
    }
  }
  when(scalar.io.dmaCmd.valid) {
    dmaLaunched(scDma.channel) := true.B  // last-connect wins over clear
  }

  for (ch <- 0 until dmaP.numChannels) {
    scalar.io.dma_busy(ch) := dma.io.channelBusy(ch) || dmaLaunched(ch)
  }

  // ==========================================================================
  // MXU-0 command dispatch (ScalarCore → SystolicArray, combinational)
  // ==========================================================================

  val mxu0CmdWire = Wire(new MxuCmd(mregP.mregIdBits))
  val sc0 = scalar.io.mxu0Cmd.bits
  mxu0CmdWire.op         := MxuOp.safe((sc0.op - 1.U)(2, 0))._1
  mxu0CmdWire.mregId     := sc0.mregBank
  mxu0CmdWire.accSel     := sc0.accSel
  mxu0CmdWire.weightSlot := sc0.weightSlot
  mxu0CmdWire.scaleE8M0  := sc0.scaleE8M0

  mxu0.io.cmd.valid := scalar.io.mxu0Cmd.valid
  mxu0.io.cmd.bits  := mxu0CmdWire

  // ==========================================================================
  // MXU-1 command dispatch (ScalarCore → InnerProductTrees, combinational)
  // ==========================================================================

  val mxu1CmdWire = Wire(new MxuCmd(mregP.mregIdBits))
  val sc1 = scalar.io.mxu1Cmd.bits
  mxu1CmdWire.op         := MxuOp.safe((sc1.op - 1.U)(2, 0))._1
  mxu1CmdWire.mregId     := sc1.mregBank
  mxu1CmdWire.accSel     := sc1.accSel
  mxu1CmdWire.weightSlot := sc1.weightSlot
  mxu1CmdWire.scaleE8M0  := sc1.scaleE8M0

  mxu1.io.cmd.valid := scalar.io.mxu1Cmd.valid
  mxu1.io.cmd.bits  := mxu1CmdWire

  // ==========================================================================
  // XLU command dispatch (ScalarCore → XluEngine, combinational)
  // ==========================================================================

  val xluCmdWire = Wire(new XluCommand(mregP.mregIdBits))
  val scX = scalar.io.xluCmd.bits
  xluCmdWire.op        := scX.op
  xluCmdWire.srcMregId := scX.srcBank
  xluCmdWire.dstMregId := scX.dstBank

  xlu.io.cmd.valid := scalar.io.xluCmd.valid
  xlu.io.cmd.bits  := xluCmdWire

  // ==========================================================================
  // Engine status feedback to the scalar core (VPU only)
  // ==========================================================================

  scalar.io.vpu_status := makeStatus(vpu.io.busy)

  // ==========================================================================
  // MREG bank tracker — direction-aware hazard detection
  // ==========================================================================
  //
  // Collects read/write bank reports from all active engines.
  // No pending-command tracking is needed: commands are accepted
  // combinationally, and the engine FSMs update activeReads/activeWrites
  // on the next clock edge — which aligns with the next scalar instruction
  // due to the 2-stage pipeline.

  private val numBanks = 64
  private val bankBits = log2Ceil(numBanks)
  val mregTracker = Module(new MregBankTracker(numBanks, numReaders = 10, numWriters = 10))

  // Reader port assignment (10 ports):
  //   0-1: MXU0 active reads
  //   2-3: MXU1 active reads
  //   4:   XLU active read
  //   5-9: unused (tied off)
  mregTracker.io.readers(0) := mxu0.io.activeReads(0)
  mregTracker.io.readers(1) := mxu0.io.activeReads(1)
  mregTracker.io.readers(2) := mxu1.io.activeReads(0)
  mregTracker.io.readers(3) := mxu1.io.activeReads(1)
  mregTracker.io.readers(4) := xlu.io.activeMregRead
  for (i <- 5 until 10) {
    mregTracker.io.readers(i).valid := false.B
    mregTracker.io.readers(i).bits  := 0.U
  }

  // Writer port assignment (10 ports):
  //   0-1: MXU0 active writes
  //   2-3: MXU1 active writes
  //   4:   XLU active write
  //   5-9: unused (tied off)
  mregTracker.io.writers(0) := mxu0.io.activeWrites(0)
  mregTracker.io.writers(1) := mxu0.io.activeWrites(1)
  mregTracker.io.writers(2) := mxu1.io.activeWrites(0)
  mregTracker.io.writers(3) := mxu1.io.activeWrites(1)
  mregTracker.io.writers(4) := xlu.io.activeMregWrite
  for (i <- 5 until 10) {
    mregTracker.io.writers(i).valid := false.B
    mregTracker.io.writers(i).bits  := 0.U
  }

  scalar.io.mregReadBusy  := mregTracker.io.readBusy
  scalar.io.mregWriteBusy := mregTracker.io.writeBusy

  // ==========================================================================
  // VPU command forwarding
  // ==========================================================================

  vpu.io.cmd := scalar.io.vpuCmd

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

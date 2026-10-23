// ============================================================================
// AtlasCore.scala — Non-diplomatic top-level core (VMEM, concurrent LSU).
//
// Data-path topology:
//   Vmem.lsuScalar{Read,Write}  ←→  LSU scalar path
//   Vmem.lsuVec{Read,Write}     ←→  LSU vector path
//   Vmem.dma{Read,Write}+grant  ←→  DMA engine
//   Vmem.tl                     ←→  io.vmemTL (Saturn / SBUS)
//   LSU ←row port→ MREG
//   LSU ←scalar cmd/resp→ ScalarCore
// ============================================================================

package atlas.tile

import chisel3._
import chisel3.util._

import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters}

import atlas.common._
import atlas.scalar.{ScalarCore, ScalarISA, VpuStatus, AtlasMemMap,
                      CSRFile, LsuCmd, InstrMem, ImemFetchPort, DmaCmd,
                      MxuCmd => ScalarMxuCmd, VpuCmd, XluCmd,
                      MregBankTracker}
import atlas.ipt.InnerProductTreesTop
import atlas.sa.SystolicArrayTop
import atlas.mxu.{MxuOp, MxuCmd}
import atlas.vector.VectorEngineTop
import atlas.dma.{DmaEngine, DmaCommand, DmaDirection}
import atlas.xlu.{XluEngine, XluCommand}
import atlas.lsu.LSU
import atlas.mreg.MregFile
import atlas.vmem.Vmem
import pmu.PMURegistry

class AtlasCore(
  tp:     AtlasParams        = AtlasParams(),
  imemBP: TLBundleParameters,
  csrBP:  TLBundleParameters,
  dmaBP:  TLBundleParameters,
  vmemBP: TLBundleParameters,
  pmuRegistry: Option[PMURegistry] = None
) extends Module {

  private val iptP  = tp.ipt
  private val vpuP  = tp.vpu
  private val mregP = tp.mreg
  private val dmaP  = tp.dma
  private val vmemP = tp.vmem
  private val saP   = tp.sa
  import ScalarISA._

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
  val vmem    = Module(new Vmem(vmemP, vmemBP))
  val scalar  = Module(new ScalarCore(vmemP))
  val dma     = Module(new DmaEngine(tp, dmaBP))
  val lsu     = Module(new LSU(vmemP, mregP))
  val mxu0    = Module(new SystolicArrayTop(saP, mregP))
  val mxu1    = Module(new InnerProductTreesTop(iptP, mregP))
  val vpu     = Module(new VectorEngineTop(vpuP, mregP))
  val xlu     = Module(new XluEngine(mregP))
  val mreg    = Module(new MregFile(mregP))

  // ==========================================================================
  // TileLink ports
  // ==========================================================================

  imem.io.fetch       <> scalar.io.imemFetch
  imem.io.fetchActive := !scalar.io.halted
  imem.io.tl          <> io.imemTL
  csrfile.io.tl <> io.csrTL
  io.dmaTL      <> dma.io.tl
  io.halted     := scalar.io.halted
  vmem.io.tl    <> io.vmemTL

  // ==========================================================================
  // CSR
  // ==========================================================================

  csrfile.io.csr <> scalar.io.csrPort
  pmuRegistry.foreach { registry =>
    def event(name: String, increment: UInt, unit: String, description: String): Unit =
      registry.register(name, increment, unit, description)

    event("core.run_cycles", csrfile.io.execRun.asUInt, "cycles",
      "Cycles while Atlas execution is enabled")
    event("dma.active_cycles", dma.io.channelBusy.asUInt.orR.asUInt, "cycles",
      "Cycles with at least one DMA channel active")
    event("dma.commands", scalar.io.dmaCmd.valid.asUInt, "commands",
      "DMA commands issued by the scalar core")
    event("dma.beats_issued", dma.io.perfIssuedBeat.asUInt, "beats",
      "VMEM beats issued by DMA")
    event("dma.beats_completed", dma.io.perfCompletedBeat.asUInt, "beats",
      "VMEM beats completed by DMA")
    event("dma.vmem_stall_cycles", dma.io.perfVmemStall.asUInt, "cycles",
      "Cycles DMA is stalled by VMEM")
    event("dma.saturated_cycles", dma.io.perfSaturated.asUInt, "cycles",
      "Cycles DMA is at its configured in-flight limit")
    event("dma.limit_blocked_cycles", dma.io.perfLimitBlocked.asUInt, "cycles",
      "Cycles an otherwise ready DMA request is blocked by the in-flight limit")
    event("dma.outstanding_cycles", dma.io.perfOutstanding.asUInt, "cycles",
      "Cycles with at least one DMA beat outstanding")
    event("dma.response_wait_beat_cycles", dma.io.perfResponseWaitBeatCycles, "beat-cycles",
      "Outstanding DMA beats summed across cycles while awaiting responses")
    event("dma.fence_wait_cycles", scalar.io.dmaFenceWait.asUInt, "cycles",
      "Cycles the scalar core waits for a DMA fence")
    event("vpu.active_cycles", vpu.io.busy.asUInt, "cycles",
      "Cycles the vector processing unit is active")
    event("vpu.commands", scalar.io.vpuCmd.valid.asUInt, "commands",
      "VPU commands issued by the scalar core")
    event("mxu0.compute_cycles", mxu0.io.computeBusy.asUInt, "cycles",
      "Cycles MXU 0 performs computation")
    event("mxu0.data_cycles", mxu0.io.dataBusy.asUInt, "cycles",
      "Cycles MXU 0 is active in its data path")
    event("mxu0.commands", scalar.io.mxu0Cmd.valid.asUInt, "commands",
      "MXU 0 commands issued by the scalar core")
    event("mxu1.compute_cycles", mxu1.io.computeBusy.asUInt, "cycles",
      "Cycles MXU 1 performs computation")
    event("mxu1.data_cycles", mxu1.io.dataBusy.asUInt, "cycles",
      "Cycles MXU 1 is active in its data path")
    event("mxu1.commands", scalar.io.mxu1Cmd.valid.asUInt, "commands",
      "MXU 1 commands issued by the scalar core")
    event("xlu.commands", xlu.io.cmd.valid.asUInt, "commands",
      "XLU transpose commands issued by the scalar core")
    event("xlu.active_cycles", xlu.io.busy.asUInt, "cycles",
      "Cycles the XLU is processing a transpose")
    event("xlu.mreg_write_rows", xlu.io.mregWriteReq.valid.asUInt, "rows",
      "Transposed rows written by the XLU to MREG")
    event("lsu.scalar_loads", lsu.io.vmemScalarRead.valid.asUInt, "operations",
      "Scalar loads issued to VMEM")
    event("lsu.scalar_stores", lsu.io.vmemScalarWrite.valid.asUInt, "operations",
      "Scalar stores issued to VMEM")
    event("lsu.scalar_load_completions", lsu.io.scalarResp.valid.asUInt, "operations",
      "Scalar load results returned by the LSU")
    event("lsu.vload_commands",
      (lsu.io.cmd.valid && lsu.io.cmd.bits.op === LSU_VLOAD).asUInt, "commands",
      "VLOAD commands issued to the LSU")
    event("lsu.vstore_commands",
      (lsu.io.cmd.valid && lsu.io.cmd.bits.op === LSU_VSTORE).asUInt, "commands",
      "VSTORE commands issued to the LSU")
    event("lsu.vload_lines_issued", lsu.io.vmemVecRead.valid.asUInt, "lines",
      "VMEM lines requested by VLOAD")
    event("lsu.vload_rows_written", lsu.io.mregWriteReq.valid.asUInt, "rows",
      "MREG rows written by VLOAD")
    event("lsu.vstore_rows_read", lsu.io.mregReadReq.valid.asUInt, "rows",
      "MREG rows requested by VSTORE")
    event("lsu.vstore_lines_written", lsu.io.vmemVecWrite.valid.asUInt, "lines",
      "VMEM lines written by VSTORE")
    event("lsu.scalar_busy_cycles", lsu.io.scalarBusy.asUInt, "cycles",
      "Cycles a scalar load is in flight")
    event("lsu.vload_active_cycles", lsu.io.vloadBusy.asUInt, "cycles",
      "Cycles the VLOAD path is active")
    event("lsu.vstore_active_cycles", lsu.io.vstoreBusy.asUInt, "cycles",
      "Cycles the VSTORE path is active")

    for (bank <- 0 until mregP.numMregBanks) {
      event(f"mreg.bank_$bank%02d.read_cycles", mreg.io.perfReadAccess(bank).asUInt,
        "cycles", s"Cycles MREG bank $bank accepts a read")
      event(f"mreg.bank_$bank%02d.write_cycles", mreg.io.perfWriteAccess(bank).asUInt,
        "cycles", s"Cycles MREG bank $bank accepts a write")
      event(f"mreg.bank_$bank%02d.read_contended_cycles", mreg.io.perfReadContended(bank).asUInt,
        "cycles", s"Cycles MREG bank $bank has contended read requests")
      event(f"mreg.bank_$bank%02d.write_contended_cycles", mreg.io.perfWriteContended(bank).asUInt,
        "cycles", s"Cycles MREG bank $bank has contended write requests")
    }
    for (bank <- 0 until vmemP.numBanks) {
      event(f"vmem.bank_$bank%02d.read_cycles", vmem.io.perfReadAccess(bank).asUInt,
        "cycles", s"Cycles VMEM bank $bank accepts a read")
      event(f"vmem.bank_$bank%02d.write_cycles", vmem.io.perfWriteAccess(bank).asUInt,
        "cycles", s"Cycles VMEM bank $bank accepts a write")
      event(f"vmem.bank_$bank%02d.contended_cycles", vmem.io.perfContended(bank).asUInt,
        "cycles", s"Cycles VMEM bank $bank has contending requests")
    }
  }
  scalar.io.execRun      := csrfile.io.execRun
  scalar.io.execRunWrite := csrfile.io.execRunWrite

  // ==========================================================================
  // Vmem ↔ LSU scalar ports
  // ==========================================================================

  vmem.io.lsuScalarRead     <> lsu.io.vmemScalarRead
  lsu.io.vmemScalarReadData := vmem.io.lsuScalarReadData
  vmem.io.lsuScalarWrite    <> lsu.io.vmemScalarWrite

  // ==========================================================================
  // Vmem ↔ LSU vector ports
  // ==========================================================================

  vmem.io.lsuVecRead     <> lsu.io.vmemVecRead
  lsu.io.vmemVecReadData := vmem.io.lsuVecReadData
  vmem.io.lsuVecWrite    <> lsu.io.vmemVecWrite

  // ==========================================================================
  // Vmem ↔ DMA (with backpressure)
  // ==========================================================================

  vmem.io.dmaRead       <> dma.io.vmemRead
  dma.io.vmemReadGrant  := vmem.io.dmaReadGrant
  dma.io.vmemReadData   := vmem.io.dmaReadData
  vmem.io.dmaWrite      <> dma.io.vmemWrite
  dma.io.vmemWriteGrant := vmem.io.dmaWriteGrant

  // ==========================================================================
  // LSU ↔ MREG
  // ==========================================================================

  mreg.io.lsuReadReq   <> lsu.io.mregReadReq
  lsu.io.mregReadResp  := mreg.io.lsuReadResp
  mreg.io.lsuWriteReq  <> lsu.io.mregWriteReq

  // ==========================================================================
  // ScalarCore ↔ LSU
  // ==========================================================================

  lsu.io.scalarCmd        <> scalar.io.scalarMemCmd
  scalar.io.scalarMemResp := lsu.io.scalarResp
  lsu.io.cmd              := scalar.io.lsuCmd

  scalar.io.lsu_scalar_busy := lsu.io.scalarBusy
  scalar.io.lsu_vload_busy  := lsu.io.vloadBusy
  scalar.io.lsu_vstore_busy := lsu.io.vstoreBusy

  // ==========================================================================
  // Utility
  // ==========================================================================

  def makeVpuStatus(busy: Bool, issueBusy: UInt): VpuStatus = {
    val s = Wire(new VpuStatus)
    s.busy      := busy
    s.done      := !busy
    s.error     := false.B
    s.issueBusy := issueBusy
    s
  }

  // ==========================================================================
  // DMA command dispatch
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

  val dmaLaunched = RegInit(VecInit(Seq.fill(dmaP.numChannels)(false.B)))
  for (ch <- 0 until dmaP.numChannels) {
    when(dma.io.channelBusy(ch)) { dmaLaunched(ch) := false.B }
  }
  when(scalar.io.dmaCmd.valid) { dmaLaunched(scDma.channel) := true.B }
  for (ch <- 0 until dmaP.numChannels) {
    scalar.io.dma_busy(ch) := dma.io.channelBusy(ch) || dmaLaunched(ch)
  }

  // ==========================================================================
  // MXU-0 / MXU-1 / VPU / XLU command dispatch
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

  val mxu1CmdWire = Wire(new MxuCmd(mregP.mregIdBits))
  val sc1 = scalar.io.mxu1Cmd.bits
  mxu1CmdWire.op         := MxuOp.safe((sc1.op - 1.U)(2, 0))._1
  mxu1CmdWire.mregId     := sc1.mregBank
  mxu1CmdWire.accSel     := sc1.accSel
  mxu1CmdWire.weightSlot := sc1.weightSlot
  mxu1CmdWire.scaleE8M0  := sc1.scaleE8M0
  mxu1.io.cmd.valid := scalar.io.mxu1Cmd.valid
  mxu1.io.cmd.bits  := mxu1CmdWire

  vpu.io.cmd.valid := scalar.io.vpuCmd.valid
  vpu.io.cmd.bits  := scalar.io.vpuCmd.bits

  val xluCmdWire = Wire(new XluCommand(mregP.mregIdBits))
  val scX = scalar.io.xluCmd.bits
  xluCmdWire.op        := scX.op
  xluCmdWire.srcMregId := scX.srcBank
  xluCmdWire.dstMregId := scX.dstBank
  xlu.io.cmd.valid := scalar.io.xluCmd.valid
  xlu.io.cmd.bits  := xluCmdWire

  scalar.io.vpu_status := makeVpuStatus(vpu.io.busy, vpu.io.issueBusy)

  // ==========================================================================
  // MREG bank tracker
  // ==========================================================================

  private val numBanks = 64
  val mregTracker = Module(new MregBankTracker(numBanks, numReaders = 10, numWriters = 10))

  mregTracker.io.readers(0) := mxu0.io.activeReads(0)
  mregTracker.io.readers(1) := mxu0.io.activeReads(1)
  mregTracker.io.readers(2) := mxu1.io.activeReads(0)
  mregTracker.io.readers(3) := mxu1.io.activeReads(1)
  mregTracker.io.readers(4) := xlu.io.activeMregRead
  mregTracker.io.readers(5) := vpu.io.activeReads(0)
  mregTracker.io.readers(6) := vpu.io.activeReads(1)
  mregTracker.io.readers(7) := vpu.io.activeReads(2)
  mregTracker.io.readers(8) := vpu.io.activeReads(3)
  mregTracker.io.readers(9) := lsu.io.activeMregRead

  mregTracker.io.writers(0) := mxu0.io.activeWrites(0)
  mregTracker.io.writers(1) := mxu0.io.activeWrites(1)
  mregTracker.io.writers(2) := mxu1.io.activeWrites(0)
  mregTracker.io.writers(3) := mxu1.io.activeWrites(1)
  mregTracker.io.writers(4) := xlu.io.activeMregWrite
  mregTracker.io.writers(5) := vpu.io.activeWrites(0)
  mregTracker.io.writers(6) := vpu.io.activeWrites(1)
  mregTracker.io.writers(7) := vpu.io.activeWrites(2)
  mregTracker.io.writers(8) := vpu.io.activeWrites(3)
  mregTracker.io.writers(9) := lsu.io.activeMregWrite

  scalar.io.mregReadBusy  := mregTracker.io.readBusy
  scalar.io.mregWriteBusy := mregTracker.io.writeBusy

  // ==========================================================================
  // MREG wiring
  // ==========================================================================

  mreg.io.mxu1ReadReq0  := mxu1.io.mregReadReq0
  mxu1.io.mregReadResp0 := mreg.io.mxu1ReadResp0
  mreg.io.mxu1ReadReq1  := mxu1.io.mregReadReq1
  mxu1.io.mregReadResp1 := mreg.io.mxu1ReadResp1
  mreg.io.mxu1WriteReq0 := mxu1.io.mregWriteReq0
  mreg.io.mxu1WriteReq1 := mxu1.io.mregWriteReq1

  mreg.io.mxu0ReadReq0  := mxu0.io.mregReadReq0
  mreg.io.mxu0ReadReq1  := mxu0.io.mregReadReq1
  mxu0.io.mregReadResp0 := mreg.io.mxu0ReadResp0
  mxu0.io.mregReadResp1 := mreg.io.mxu0ReadResp1
  mreg.io.mxu0WriteReq0 := mxu0.io.mregWriteReq0
  mreg.io.mxu0WriteReq1 := mxu0.io.mregWriteReq1

  mreg.io.vpuReadReq0  := vpu.io.mregReadReq0
  vpu.io.mregReadResp0 := mreg.io.vpuReadResp0
  mreg.io.vpuReadReq1  := vpu.io.mregReadReq1
  vpu.io.mregReadResp1 := mreg.io.vpuReadResp1
  mreg.io.vpuWriteReq0 := vpu.io.mregWriteReq0
  mreg.io.vpuWriteReq1 := vpu.io.mregWriteReq1

  mreg.io.xluReadReq         := xlu.io.mregReadReq
  xlu.io.mregReadResp.valid := mreg.io.xluReadResp.valid
  xlu.io.mregReadResp.bits  := mreg.io.xluReadResp.bits
  mreg.io.xluWriteReq        := xlu.io.mregWriteReq

  // ==========================================================================
  // Debug
  // ==========================================================================

  io.dbg.mxu0DataBusy := mxu0.io.dataBusy
  io.dbg.mxu0CompBusy := mxu0.io.computeBusy
  io.dbg.mxu1DataBusy := mxu1.io.dataBusy
  io.dbg.mxu1CompBusy := mxu1.io.computeBusy
  io.dbg.xluBusy      := xlu.io.busy
  io.dbg.dmaBusy      := dma.io.channelBusy
  io.dbg.lsuBusy      := lsu.io.scalarBusy || lsu.io.vecBusy
  io.dbg.scaleRegs    := scalar.io.scaleRegs
  io.dbg.dbg0         := csrfile.io.dbg0
  io.dbg.dbg1         := csrfile.io.dbg1
}

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
import atlas.mreg.{MregFile, MregScaleFile}
import atlas.nvfp4.{Nvfp4Mxu, NvOp, NvCmd}
import atlas.vmem.Vmem

class AtlasCore(
  tp:     AtlasParams        = AtlasParams(),
  imemBP: TLBundleParameters,
  csrBP:  TLBundleParameters,
  dmaBP:  TLBundleParameters,
  vmemBP: TLBundleParameters
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
    // Constant: [0] hardware interlock, [1] systolic-array MXU, [2] inner-product-tree MXU,
    // [3] NVFP4 MXU and scale sidecar.
    val features = Output(UInt(4.W))
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
  val scalar  = Module(new ScalarCore(vmemP, hasMxu0 = tp.hasSystolicArray, hasNvfp4 = tp.hasNvfp4))
  val dma     = Module(new DmaEngine(tp, dmaBP))
  val lsu     = Module(new LSU(vmemP, mregP))
  // MXU slot 0: the systolic array, or the NVFP4 MXU (which then uses slot 0's
  // MregFile ports), or nothing.
  val mxu0    = Option.when(tp.hasSystolicArray)(Module(new SystolicArrayTop(saP, mregP)))
  val nv      = tp.nvfp4.map(np => Module(new Nvfp4Mxu(np, mregP)))
  val mscale  = tp.nvfp4.map(_ => Module(new MregScaleFile(mregP)))
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
  mxu0.foreach { m =>
    m.io.cmd.valid := scalar.io.mxu0Cmd.valid
    m.io.cmd.bits  := mxu0CmdWire
  }

  val nvCmdWire = Wire(new NvCmd(mregP.mregIdBits))
  val scn = scalar.io.nvCmd.bits
  nvCmdWire.op         := NvOp.safe((scn.op - ScalarISA.MXU_NV_PUSH_WEIGHT)(2, 0))._1
  nvCmdWire.mregId     := scn.mregBank
  nvCmdWire.accSel     := scn.accSel
  nvCmdWire.weightSlot := scn.weightSlot
  nv.foreach { n =>
    n.io.cmd.valid := scalar.io.nvCmd.valid
    n.io.cmd.bits  := nvCmdWire
  }

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

  // Interlock: readiness of each unit for the command decoded in S1.
  scalar.io.mxu0Ready            := mxu0.map(_.io.cmdReady).getOrElse(false.B)
  scalar.io.mxu1Ready            := mxu1.io.cmdReady
  scalar.io.nvReady              := nv.map(_.io.cmdReady).getOrElse(false.B)
  scalar.io.xlu_busy             := xlu.io.busy
  scalar.io.dma_can_enqueue      := dma.io.canEnqueue
  scalar.io.lsu_vload_vmem_bank  := lsu.io.vloadVmemBank
  scalar.io.lsu_vstore_vmem_bank := lsu.io.vstoreVmemBank

  // ==========================================================================
  // MREG bank tracker
  // ==========================================================================

  private val numBanks = 64
  // Reporters 0, 1 (and 10, 11 for the NVFP4 MXU's 4-register transfers) are MXU slot 0.
  val mregTracker = Module(new MregBankTracker(numBanks, numReaders = 12, numWriters = 12))
  private val noBank = 0.U.asTypeOf(Valid(UInt(mregP.mregIdBits.W)))
  private def slot0Reads: Seq[Valid[UInt]] =
    mxu0.map(m => Seq(m.io.activeReads(0), m.io.activeReads(1), noBank, noBank))
      .orElse(nv.map(n => n.io.activeReads.toSeq)).getOrElse(Seq.fill(4)(noBank))
  private def slot0Writes: Seq[Valid[UInt]] =
    mxu0.map(m => Seq(m.io.activeWrites(0), m.io.activeWrites(1), noBank, noBank))
      .orElse(nv.map(n => n.io.activeWrites.toSeq)).getOrElse(Seq.fill(4)(noBank))

  mregTracker.io.readers(0)  := slot0Reads(0)
  mregTracker.io.readers(1)  := slot0Reads(1)
  mregTracker.io.readers(10) := slot0Reads(2)
  mregTracker.io.readers(11) := slot0Reads(3)
  mregTracker.io.readers(2) := mxu1.io.activeReads(0)
  mregTracker.io.readers(3) := mxu1.io.activeReads(1)
  mregTracker.io.readers(4) := xlu.io.activeMregRead
  mregTracker.io.readers(5) := vpu.io.activeReads(0)
  mregTracker.io.readers(6) := vpu.io.activeReads(1)
  mregTracker.io.readers(7) := vpu.io.activeReads(2)
  mregTracker.io.readers(8) := vpu.io.activeReads(3)
  mregTracker.io.readers(9) := lsu.io.activeMregRead

  mregTracker.io.writers(0)  := slot0Writes(0)
  mregTracker.io.writers(1)  := slot0Writes(1)
  mregTracker.io.writers(10) := slot0Writes(2)
  mregTracker.io.writers(11) := slot0Writes(3)
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

  private def off[T <: Data](v: Valid[T]): Unit = { v.valid := false.B; v.bits := 0.U.asTypeOf(v.bits) }
  mxu0.foreach { m =>
    mreg.io.mxu0ReadReq0  := m.io.mregReadReq0
    mreg.io.mxu0ReadReq1  := m.io.mregReadReq1
    m.io.mregReadResp0    := mreg.io.mxu0ReadResp0
    m.io.mregReadResp1    := mreg.io.mxu0ReadResp1
    mreg.io.mxu0WriteReq0 := m.io.mregWriteReq0
    mreg.io.mxu0WriteReq1 := m.io.mregWriteReq1
  }
  nv.foreach { n =>
    mreg.io.mxu0ReadReq0  := n.io.mregReadReq0
    mreg.io.mxu0ReadReq1  := n.io.mregReadReq1
    n.io.mregReadResp0    := mreg.io.mxu0ReadResp0
    n.io.mregReadResp1    := mreg.io.mxu0ReadResp1
    mreg.io.mxu0WriteReq0 := n.io.mregWriteReq0
    mreg.io.mxu0WriteReq1 := n.io.mregWriteReq1
  }
  if (mxu0.isEmpty && nv.isEmpty) {
    Seq(mreg.io.mxu0ReadReq0, mreg.io.mxu0ReadReq1).foreach(off(_))
    Seq(mreg.io.mxu0WriteReq0, mreg.io.mxu0WriteReq1).foreach(off(_))
  }

  // ── NVFP4 scale sidecar: LSU (vload.scale / vstore.scale) and NVFP4 MXU reads ──
  mscale match {
    case Some(sf) =>
      sf.io.lineWrite := lsu.io.scaleWrite
      sf.io.lineRead  := lsu.io.scaleLineRead
      lsu.io.scaleLineResp := sf.io.lineResp
      for (i <- 0 until 2) {
        sf.io.rowRead(i) := nv.get.io.scaleReadReq(i)
        nv.get.io.scaleReadResp(i) := sf.io.rowResp(i)
      }
    case None =>
      lsu.io.scaleLineResp := 0.U
  }

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

  // Slot 0 busy bits report the systolic array or the NVFP4 MXU.
  io.dbg.mxu0DataBusy := mxu0.map(_.io.dataBusy).getOrElse(false.B)
  io.dbg.mxu0CompBusy := mxu0.map(_.io.computeBusy).orElse(nv.map(_.io.busy)).getOrElse(false.B)
  io.features := Cat(tp.hasNvfp4.B, true.B, tp.hasSystolicArray.B, true.B)

  io.dbg.mxu1DataBusy := mxu1.io.dataBusy
  io.dbg.mxu1CompBusy := mxu1.io.computeBusy
  io.dbg.xluBusy      := xlu.io.busy
  io.dbg.dmaBusy      := dma.io.channelBusy
  io.dbg.lsuBusy      := lsu.io.scalarBusy || lsu.io.vecBusy
  io.dbg.scaleRegs    := scalar.io.scaleRegs
  io.dbg.dbg0         := csrfile.io.dbg0
  io.dbg.dbg1         := csrfile.io.dbg1
}

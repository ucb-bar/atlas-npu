/*
AtlasCore.scala — Non-diplomatic core.

Data path:
  System bus <-TL-> DMA <-line port-> VMEM
  VMEM <-line read/write-> LSU <-row port-> TRF
  LSU <-scalar cmd/resp-> ScalarCore (all VMEM access serialized)
  Host <-TL-> IMEM, Host <-TL-> CSR
*/

package atlas.tile

import chisel3._
import chisel3.util._

import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters}

import atlas.common._
import atlas.scalar.{ScalarCore, ScalarISA, EngineStatus, AtlasMemMap,
                      CSRFile, LsuCmd, InstrMem, ImemFetchPort, DmaCmd,
                      MxuCmd, VpuCmd, XluCmd}
import atlas.ipt.{InnerProductTreesSequencer, Mxu0Cmd, Mxu0Op}
import atlas.dma.{DMASplitInterface, dma_instruction, dma_instruction_type}
import atlas.xlu.xlu_engine

// ── Stubs ─────────────────────────────────────────────────────

class VpuConfig extends Bundle {
  val op      = UInt(8.W)
  val srcBank = UInt(8.W)
  val dstBank = UInt(8.W)
  val len     = UInt(16.W)
}

class VpuEngine(rfP: RegFileParams) extends Module {
  val io = IO(new Bundle {
    val cmd       = Flipped(Decoupled(new VpuConfig))
    val trfRead0  = Valid(new RegFileReadInput(rfP))
    val trfRead1  = Valid(new RegFileReadInput(rfP))
    val trfReadData0 = Input(Valid(UInt(rfP.SRAM_WIDTH.W)))
    val trfReadData1 = Input(Valid(UInt(rfP.SRAM_WIDTH.W)))
    val trfWrite0 = Valid(new RegFileWriteInput(rfP))
    val trfWrite1 = Valid(new RegFileWriteInput(rfP))
    val busy      = Output(Bool())
  })
  io.cmd.ready := true.B
  io.trfRead0.valid := false.B;  io.trfRead0.bits := 0.U.asTypeOf(io.trfRead0.bits)
  io.trfRead1.valid := false.B;  io.trfRead1.bits := 0.U.asTypeOf(io.trfRead1.bits)
  io.trfWrite0.valid := false.B; io.trfWrite0.bits := 0.U.asTypeOf(io.trfWrite0.bits)
  io.trfWrite1.valid := false.B; io.trfWrite1.bits := 0.U.asTypeOf(io.trfWrite1.bits)
  io.busy := false.B
}

class Mxu1Stub(rfP: RegFileParams) extends Module {
  val io = IO(new Bundle {
    val cmd   = Flipped(Valid(new MxuCmd))
    val trfRead0  = Valid(new RegFileReadInput(rfP))
    val trfRead1  = Valid(new RegFileReadInput(rfP))
    val trfReadData0 = Input(Valid(UInt(rfP.SRAM_WIDTH.W)))
    val trfReadData1 = Input(Valid(UInt(rfP.SRAM_WIDTH.W)))
    val trfWrite0 = Valid(new RegFileWriteInput(rfP))
    val trfWrite1 = Valid(new RegFileWriteInput(rfP))
    val busy      = Output(Bool())
  })
  io.trfRead0.valid := false.B;  io.trfRead0.bits := 0.U.asTypeOf(io.trfRead0.bits)
  io.trfRead1.valid := false.B;  io.trfRead1.bits := 0.U.asTypeOf(io.trfRead1.bits)
  io.trfWrite0.valid := false.B; io.trfWrite0.bits := 0.U.asTypeOf(io.trfWrite0.bits)
  io.trfWrite1.valid := false.B; io.trfWrite1.bits := 0.U.asTypeOf(io.trfWrite1.bits)
  io.busy := false.B
}

// ── AtlasCore ────────────────────────────────────────────────────────

class AtlasCore(
  tp:     AtlasParams        = AtlasParams(),
  imemBP: TLBundleParameters,
  csrBP:  TLBundleParameters,
  dmaBP:  TLBundleParameters
) extends Module {

  private val iptP = tp.ipt
  private val rfP  = tp.regfile
  private val dmaP = tp.dma
  private val spP  = tp.scratchpad
  import ScalarISA._

  val io = IO(new Bundle {
    val imemTL = Flipped(new TLBundle(imemBP))
    val csrTL  = Flipped(new TLBundle(csrBP))
    val dmaTL  = new TLBundle(dmaBP)
    val halted = Output(Bool())
    val dbg = Output(new Bundle {
      val mxu0DataBusy = Bool()
      val mxu0CompBusy = Bool()
      val xluBusy      = Bool()
      val dmaBusy      = Vec(dmaP.numInterfaces, Bool())
      val lsuBusy      = Bool()
      val scaleRegs    = Vec(ScalarISA.NUM_SCALE_REGS, UInt(8.W))
      val dbg0         = UInt(32.W)
      val dbg1         = UInt(32.W)
    })
  })

  // ── Sub-modules ───────────────────────────────────────────────────
  val imem    = Module(new InstrMem(imemBP))
  val csrfile = Module(new CSRFile(csrBP))
  val vmem    = Module(new ScratchpadMem(spP))
  val scalar  = Module(new ScalarCore(spP))
  val dma     = Module(new DMASplitInterface(tp, dmaBP))
  val lsu     = Module(new LSU(spP, rfP))
  val mxu0Seq = Module(new InnerProductTreesSequencer(iptP, rfP))
  val mxu1    = Module(new Mxu1Stub(rfP))
  val vpu     = Module(new VpuEngine(rfP))
  val xlu     = Module(new xlu_engine(rfP))
  val trf     = Module(new TensorRegFile(rfP))

  // ── TileLink ──────────────────────────────────────────────────────
  imem.io.fetch <> scalar.io.imemFetch
  imem.io.tl    <> io.imemTL
  csrfile.io.tl <> io.csrTL
  io.dmaTL      <> dma.io.tl
  io.halted     := scalar.io.halted

  // ── CSR ──────────────────────────────────────────────────────────
  csrfile.io.csr <> scalar.io.csrPort

  // ── DMA <-> VMEM (line ports) ───────────────────────────────────
  vmem.io.dmaRead     <> dma.io.vmemRead
  dma.io.vmemReadData := vmem.io.dmaReadData
  vmem.io.dmaWrite    <> dma.io.vmemWrite

  // ── LSU <-> VMEM (line read + line write only) ──────────────────
  vmem.io.lsuRead     <> lsu.io.vmemRead
  lsu.io.vmemReadData := vmem.io.lsuReadData
  vmem.io.lsuWrite    <> lsu.io.vmemWrite

  // ── LSU <-> TRF ────────────────────────────────────────────────
  trf.io.dmareadinput  := lsu.io.trfRead
  lsu.io.trfReadData   := trf.io.dmareadoutput
  trf.io.dmawriteinput := lsu.io.trfWrite

  // ── ScalarCore <-> LSU (scalar mem + tensor commands) ───────────
  lsu.io.scalarCmd        <> scalar.io.scalarMemCmd
  scalar.io.scalarMemResp := lsu.io.scalarResp
  lsu.io.cmd              := scalar.io.lsuCmd
  scalar.io.lsu_busy      := lsu.io.busy

  // ── Helpers ──────────────────────────────────────────────────────
  def makeStatus(busy: Bool): EngineStatus = {
    val s = Wire(new EngineStatus)
    s.busy := busy; s.done := !busy; s.error := false.B; s
  }

  // ── DMA command dispatch ─────────────────────────────────────────
  val dmaPending = RegInit(false.B)
  val dmaBitsReg = Reg(new dma_instruction(spP, dmaP))

  for (i <- 0 until dmaP.numInterfaces) {
    scalar.io.dma_busy(i) := dma.io.busy(i) ||
      (dmaPending && dmaBitsReg.channelId === i.U)
  }

  private val wordOffBits = spP.wordOffBits

  when(scalar.io.dmaCmd.valid) {
    assert(!dmaPending, "DMA command while prior pending")
    val cmd = scalar.io.dmaCmd.bits
    dmaPending := true.B
    dmaBitsReg.type_of_instruction := Mux(cmd.op === DMA_LD,
      dma_instruction_type.LOAD_TO_VMEM, dma_instruction_type.STORE_FROM_VMEM)
    dmaBitsReg.channelId    := cmd.channel
    dmaBitsReg.vmemLineAddr := cmd.vmemAddr(spP.wordAddrBits - 1, wordOffBits)
    dmaBitsReg.address      := cmd.addr
    dmaBitsReg.size         := cmd.size(dmaP.maxSizeBits - 1, 0)
  }

  dma.io.inst_received.valid := dmaPending
  dma.io.inst_received.bits  := dmaBitsReg
  when(dma.io.inst_received.fire) { dmaPending := false.B }

  // ── MXU0 command dispatch ────────────────────────────────────────
  val mxu0Pending = RegInit(false.B)
  val mxu0CmdReg  = Reg(new Mxu0Cmd(rfP))

  when(scalar.io.mxu0Cmd.valid) {
    assert(!mxu0Pending, "MXU0 command while prior pending")
    mxu0Pending := true.B
    val sc = scalar.io.mxu0Cmd.bits
    mxu0CmdReg.op         := Mxu0Op.safe((sc.op - 1.U)(2, 0))._1
    mxu0CmdReg.trfBank    := sc.trfBank
    mxu0CmdReg.accSel     := sc.accSel
    mxu0CmdReg.weightSlot := sc.weightSlot
    mxu0CmdReg.scaleE8M0  := sc.scaleE8M0
  }

  mxu0Seq.io.cmd.valid := mxu0Pending
  mxu0Seq.io.cmd.bits  := mxu0CmdReg
  when(mxu0Seq.io.cmd.fire) { mxu0Pending := false.B }

  scalar.io.mxu0_status := makeStatus(
    mxu0Seq.io.dataBusy || mxu0Seq.io.computeBusy || mxu0Pending)
  scalar.io.mxu1_status := makeStatus(mxu1.io.busy)
  scalar.io.vpu_status  := makeStatus(vpu.io.busy)
  scalar.io.xlu_status  := makeStatus(xlu.io.busy)

  mxu1.io.cmd := scalar.io.mxu1Cmd

  vpu.io.cmd.valid := scalar.io.vpuCmd.valid
  vpu.io.cmd.bits  := 0.U.asTypeOf(new VpuConfig)

  val xc = scalar.io.xluCmd.bits
  xlu.io.instruction_received.valid := scalar.io.xluCmd.valid
  xlu.io.instruction_received.bits.op                    := xc.op
  xlu.io.instruction_received.bits.whichBank_source      := xc.srcBank
  xlu.io.instruction_received.bits.whichBank_destination := xc.dstBank

  // ── TensorRegFile wiring ─────────────────────────────────────────
  trf.io.mxu0readinput0      := mxu0Seq.io.trfReadPort0In
  mxu0Seq.io.trfReadPort0Out := trf.io.mxu0readoutput0
  trf.io.mxu0readinput1      := mxu0Seq.io.trfReadPort1In
  mxu0Seq.io.trfReadPort1Out := trf.io.mxu0readoutput1
  trf.io.mxu0writeinput0     := mxu0Seq.io.trfWritePort0
  trf.io.mxu0writeinput1     := mxu0Seq.io.trfWritePort1

  trf.io.mxu1readinput0  := mxu1.io.trfRead0
  trf.io.mxu1readinput1  := mxu1.io.trfRead1
  mxu1.io.trfReadData0   := trf.io.mxu1readoutput0
  mxu1.io.trfReadData1   := trf.io.mxu1readoutput1
  trf.io.mxu1writeinput0 := mxu1.io.trfWrite0
  trf.io.mxu1writeinput1 := mxu1.io.trfWrite1

  trf.io.vpureadinput0  := vpu.io.trfRead0
  trf.io.vpureadinput1  := vpu.io.trfRead1
  vpu.io.trfReadData0   := trf.io.vpureadoutput0
  vpu.io.trfReadData1   := trf.io.vpureadoutput1
  trf.io.vpuwriteinput0 := vpu.io.trfWrite0
  trf.io.vpuwriteinput1 := vpu.io.trfWrite1

  trf.io.xlureadinput          := xlu.io.xluReadToMreg
  xlu.io.xluReadFromMreg.valid := trf.io.xlureadoutput.valid
  xlu.io.xluReadFromMreg.bits  := trf.io.xlureadoutput.bits
  trf.io.xluwriteinput         := xlu.io.xluWriteToMreg

  // ── Debug ────────────────────────────────────────────────────────
  io.dbg.mxu0DataBusy := mxu0Seq.io.dataBusy
  io.dbg.mxu0CompBusy := mxu0Seq.io.computeBusy
  io.dbg.xluBusy      := xlu.io.busy
  io.dbg.dmaBusy      := dma.io.busy
  io.dbg.lsuBusy      := lsu.io.busy
  io.dbg.scaleRegs    := scalar.io.scaleRegs
  io.dbg.dbg0         := csrfile.io.dbg0
  io.dbg.dbg1         := csrfile.io.dbg1
}

/*
ScalarCore.scala — 2-stage pipeline scalar controller.

All VMEM access goes through the LSU:
  - Scalar loads/stores: via scalarMemCmd/scalarMemResp
  - Tensor transfers: via lsuCmd
  - DMA: separate path (not through LSU)
*/

package atlas.scalar

import chisel3._
import chisel3.util._
import ScalarISA._
import atlas.common.ScratchpadParams
import atlas.tile.LsuScalarCmd

object AtlasMemMap {
  val IMEM_BASE      = 0x00020000L
  val IMEM_SIZE      = 0x10000
  val IMEM_WORDS     = IMEM_SIZE / 4
  val IMEM_ADDR_BITS = log2Ceil(IMEM_WORDS)
  val CSR_BASE       = 0x00030000L
  val CSR_SIZE       = 0x20
}

class CSRInternalPort extends Bundle {
  val addr        = Input(UInt(12.W))
  val op          = Input(UInt(3.W))
  val wdata       = Input(UInt(32.W))
  val rdata       = Output(UInt(32.W))
  val valid       = Input(Bool())
  val halted      = Input(Bool())
  val set_illegal = Input(Bool())
  val illegal_pc  = Input(UInt(32.W))
  val inst_retire = Input(Bool())
}

class ImemFetchPort extends Bundle {
  val addr  = Output(UInt(AtlasMemMap.IMEM_ADDR_BITS.W))
  val rdata = Input(UInt(32.W))
}

class ScaleRegFile extends Module {
  val io = IO(new Bundle {
    val writeEn   = Input(Bool())
    val writeIdx  = Input(UInt(5.W))
    val writeData = Input(UInt(8.W))
    val regs      = Output(Vec(NUM_SCALE_REGS, UInt(8.W)))
  })
  val reg = RegInit(VecInit(Seq.fill(NUM_SCALE_REGS)(0.U(8.W))))
  when(io.writeEn) { reg(io.writeIdx) := io.writeData }
  io.regs := reg
}

class EngineStatus extends Bundle {
  val busy  = Bool()
  val done  = Bool()
  val error = Bool()
}

class DmaCmd extends Bundle {
  val op       = UInt(3.W)
  val vmemAddr = UInt(32.W)
  val addr     = UInt(32.W)
  val size     = UInt(32.W)
  val channel  = UInt(3.W)
}

class MxuCmd extends Bundle {
  val op         = UInt(4.W)
  val trfBank    = UInt(6.W)
  val accSel     = Bool()
  val weightSlot = Bool()
  val scaleE8M0  = UInt(8.W)
}

class VpuCmd extends Bundle {
  val op  = UInt(5.W)
  val vd  = UInt(6.W)
  val vs1 = UInt(6.W)
  val vs2 = UInt(6.W)
  val imm = UInt(16.W)
}

class XluCmd extends Bundle {
  val op      = UInt(2.W)
  val dstBank = UInt(6.W)
  val srcBank = UInt(6.W)
}

class LsuCmd extends Bundle {
  val op           = UInt(2.W)
  val trfBank      = UInt(6.W)
  val vmemLineAddr = UInt(13.W)
}

// ── IO ───────────────────────────────────────────────────────────────

class ScalarCoreIO(spP: ScratchpadParams) extends Bundle {
  val imemFetch     = new ImemFetchPort
  val dmaCmd        = Valid(new DmaCmd)
  val mxu0Cmd       = Valid(new MxuCmd)
  val mxu1Cmd       = Valid(new MxuCmd)
  val vpuCmd        = Valid(new VpuCmd)
  val xluCmd        = Valid(new XluCmd)
  val lsuCmd        = Valid(new LsuCmd)
  // Scalar mem via LSU
  val scalarMemCmd  = Valid(new LsuScalarCmd(spP))
  val scalarMemResp = Flipped(Valid(UInt(32.W)))
  val dma_busy      = Input(Vec(8, Bool()))
  val mxu0_status   = Input(new EngineStatus)
  val mxu1_status   = Input(new EngineStatus)
  val vpu_status    = Input(new EngineStatus)
  val xlu_status    = Input(new EngineStatus)
  val lsu_busy      = Input(Bool())
  val csrPort       = Flipped(new CSRInternalPort)
  val scaleRegs     = Output(Vec(NUM_SCALE_REGS, UInt(8.W)))
  val dmaBaseReg    = Output(UInt(32.W))
  val halted        = Output(Bool())
}

// ── ScalarCore ───────────────────────────────────────────────────────

class ScalarCore(spP: ScratchpadParams) extends Module {
  val io = IO(new ScalarCoreIO(spP))

  val pc_ctrl = Module(new PcControl(0))
  val decoder = Module(new ScalarDecoder)
  val regfile = Module(new ScalarRegFile)
  val alu     = Module(new ScalarALU)
  val branch  = Module(new BranchUnit)
  val scaleRF = Module(new ScaleRegFile)

  val halted     = RegInit(false.B)
  val dmaBaseReg = RegInit(0.U(32.W))
  io.dmaBaseReg := dmaBaseReg

  val s1_hold_active = RegInit(false.B)
  val s1_instr_hold  = RegInit(0.U(32.W))
  val memLoadPending = RegInit(false.B)
  val memLoadCmd     = Reg(UInt(4.W))
  val memLoadRd      = Reg(UInt(5.W))
  val memLoadByteOff = Reg(UInt(2.W))

  // ── Stage 0 — Fetch ──────────────────────────────────────────────
  io.imemFetch.addr := pc_ctrl.io.pc(AtlasMemMap.IMEM_ADDR_BITS - 1, 0)

  // ── Stage 1 — Decode + Execute + Writeback ───────────────────────
  val s1_pc    = pc_ctrl.io.s1_pc
  val s1_valid = pc_ctrl.io.s1_valid
  val s1_instr = Mux(s1_hold_active, s1_instr_hold, io.imemFetch.rdata)

  decoder.io.instr := s1_instr
  val dec = decoder.io.decoded

  regfile.io.rs1_addr := dec.rs1
  regfile.io.rs2_addr := dec.rs2
  regfile.io.rd_addr  := dec.rd
  val rs1_data = regfile.io.rs1_data
  val rs2_data = regfile.io.rs2_data
  val rd_data  = regfile.io.rd_data

  io.csrPort.addr := dec.csr_addr

  // ── Stalls ───────────────────────────────────────────────────────
  val wait_stall = s1_valid && (dec.dma_cmd === DMA_WAIT) && io.dma_busy(dec.funct3)

  val trf_writer_busy = io.xlu_status.busy || io.vpu_status.busy
  val trf_hazard = s1_valid && trf_writer_busy && (
    (dec.mxu_cmd === MXU_PUSH_WEIGHT)  || (dec.mxu_cmd === MXU_PUSH_ACC_FP8) ||
    (dec.mxu_cmd === MXU_PUSH_ACC_BF16)|| (dec.mxu_cmd === MXU_POP_FP8) ||
    (dec.mxu_cmd === MXU_POP_BF16)     || dec.is_lsu)

  val engine_hazard = s1_valid && (
    (dec.mxu_cmd =/= MXU_NONE && dec.mxu_sel === MXUSEL_0 && io.mxu0_status.busy) ||
    (dec.mxu_cmd =/= MXU_NONE && dec.mxu_sel === MXUSEL_1 && io.mxu1_status.busy) ||
    (dec.vpu_cmd =/= VPU_NONE && io.vpu_status.busy) ||
    (dec.xlu_cmd =/= XLU_NONE && io.xlu_status.busy) ||
    (dec.is_lsu && io.lsu_busy))

  // Scalar load: stall on first encounter (issue to LSU), release when pending
  val mem_load_stall = s1_valid && dec.is_mem_load && !memLoadPending
  // Scalar store: stall if LSU is busy with a tensor op
  val mem_store_stall = s1_valid && dec.is_mem_store && io.lsu_busy
  val stall = wait_stall || engine_hazard || trf_hazard || mem_load_stall || mem_store_stall

  // ── ALU ──────────────────────────────────────────────────────────
  val alu_a = MuxLookup(dec.op1_sel, rs1_data)(Seq(
    OP1_RS1 -> rs1_data, OP1_PC -> s1_pc, OP1_ZERO -> 0.U))
  val alu_b = MuxLookup(dec.op2_sel, rs2_data)(Seq(
    OP2_RS2 -> rs2_data, OP2_IMM -> dec.imm.asUInt,
    OP2_ONE -> 1.U, OP2_ZERO -> 0.U))
  alu.io.a := alu_a; alu.io.b := alu_b; alu.io.op := dec.alu_fn
  val alu_result = alu.io.out

  // ── Branch ───────────────────────────────────────────────────────
  branch.io.branch_type := dec.br_type
  branch.io.rs1 := rs1_data; branch.io.rs2 := rs2_data
  val branch_target = (s1_pc.asSInt + (dec.imm >> 1)).asUInt
  val jal_target    = (s1_pc.asSInt + (dec.imm >> 1)).asUInt
  val jalr_target   = (rs1_data.asSInt + dec.imm).asUInt
  val is_branch_taken = s1_valid && (dec.br_type =/= BR_NONE) && branch.io.taken
  val is_jal  = s1_valid && dec.is_jal
  val is_jalr = s1_valid && dec.is_jalr
  val redirect = is_branch_taken || is_jal || is_jalr
  val redirect_target = Mux(is_jalr, jalr_target, Mux(is_jal, jal_target, branch_target))
  val link_value = s1_pc + 1.U

  // ── Illegal / halt ───────────────────────────────────────────────
  val is_illegal = s1_valid && dec.illegal
  val branch_in_delay_slot = pc_ctrl.io.in_delay_slot &&
    (dec.br_type =/= BR_NONE || dec.is_jal || dec.is_jalr)
  val illegal_detected = is_illegal || (s1_valid && branch_in_delay_slot)
  val halt_now = halted || illegal_detected
  when(illegal_detected && !halted) { halted := true.B }
  val s1_fire = s1_valid && !stall && !halt_now

  // ── Stall hold ───────────────────────────────────────────────────
  when(reset.asBool || halt_now) {
    s1_hold_active := false.B; s1_instr_hold := 0.U
  }.elsewhen(stall) {
    when(!s1_hold_active) { s1_instr_hold := io.imemFetch.rdata }
    s1_hold_active := true.B
  }.otherwise { s1_hold_active := false.B }

  // ── Scalar mem via LSU ───────────────────────────────────────────
  // Issue scalar load to LSU during stall cycle when LSU is idle
  val canIssueLsuScalar = !io.lsu_busy
  val issueScalarLoad   = s1_valid && dec.is_mem_load && !memLoadPending &&
                          canIssueLsuScalar && !halt_now
  // Issue scalar store on s1_fire (already past mem_store_stall check)
  val issueScalarStore  = s1_fire && dec.is_mem_store

  // Build store data + mask
  val storeByteOff = alu_result(1, 0)
  val storeData = Wire(UInt(32.W)); storeData := rs2_data
  val storeMask = Wire(UInt(4.W));  storeMask := 0.U
  switch(dec.mem_cmd) {
    is(MEM_SB) {
      storeData := rs2_data(7, 0) << (storeByteOff ## 0.U(3.W))
      storeMask := 1.U(4.W) << storeByteOff
    }
    is(MEM_SH) {
      storeData := rs2_data(15, 0) << (storeByteOff(1) ## 0.U(4.W))
      storeMask := 3.U(4.W) << (storeByteOff(1) ## 0.U(1.W))
    }
    is(MEM_SW) {
      storeData := rs2_data
      storeMask := "b1111".U(4.W)
    }
  }

  io.scalarMemCmd.valid         := issueScalarLoad || issueScalarStore
  io.scalarMemCmd.bits.isStore  := issueScalarStore
  io.scalarMemCmd.bits.byteAddr := alu_result(spP.byteAddrBits - 1, 0)
  io.scalarMemCmd.bits.wdata    := storeData
  io.scalarMemCmd.bits.wmask    := storeMask

  // Track load pending
  when(issueScalarLoad) {
    memLoadPending := true.B
    memLoadCmd     := dec.mem_cmd
    memLoadRd      := dec.rd
    memLoadByteOff := alu_result(1, 0)
  }
  when(memLoadPending && s1_fire) { memLoadPending := false.B }

  // ── Scalar load result extraction ───────────────────────────────
  val memWord = io.scalarMemResp.bits
  val memLoadResult = Wire(UInt(32.W))
  memLoadResult := memWord
  switch(memLoadCmd) {
    is(MEM_LB) {
      val b = (memWord >> (memLoadByteOff ## 0.U(3.W)))(7, 0)
      memLoadResult := Cat(Fill(24, b(7)), b)
    }
    is(MEM_LBU) {
      val b = (memWord >> (memLoadByteOff ## 0.U(3.W)))(7, 0)
      memLoadResult := Cat(0.U(24.W), b)
    }
    is(MEM_LH) {
      val h = (memWord >> (memLoadByteOff(1) ## 0.U(4.W)))(15, 0)
      memLoadResult := Cat(Fill(16, h(15)), h)
    }
    is(MEM_LHU) {
      val h = (memWord >> (memLoadByteOff(1) ## 0.U(4.W)))(15, 0)
      memLoadResult := Cat(0.U(16.W), h)
    }
    is(MEM_LW)   { memLoadResult := memWord }
    is(MEM_SELD) { memLoadResult := memWord }
  }

  // ── CSR ──────────────────────────────────────────────────────────
  io.csrPort.op    := Mux(s1_valid && dec.is_csr, dec.csr_cmd, CSR_NONE)
  io.csrPort.valid := s1_fire && dec.is_csr
  io.csrPort.wdata := Mux(
    dec.csr_cmd === CSR_RWI || dec.csr_cmd === CSR_RSI || dec.csr_cmd === CSR_RCI,
    dec.imm.asUInt, rs1_data)
  io.csrPort.halted      := halt_now
  io.csrPort.set_illegal := illegal_detected && !halted
  io.csrPort.illegal_pc  := s1_pc
  io.csrPort.inst_retire := s1_fire

  // ── Scale register writes ────────────────────────────────────────
  val seldWriteEn = memLoadPending && s1_fire && memLoadCmd === MEM_SELD
  scaleRF.io.writeEn   := (s1_fire && dec.is_seli) || seldWriteEn
  scaleRF.io.writeIdx  := Mux(seldWriteEn, memLoadRd, dec.rd)
  scaleRF.io.writeData := Mux(seldWriteEn, memLoadResult(7, 0), dec.imm(7, 0))

  // ── Writeback ────────────────────────────────────────────────────
  val isMemLoadWB = memLoadPending && memLoadCmd =/= MEM_SELD
  val wb_data = Mux(isMemLoadWB, memLoadResult,
                Mux(dec.is_csr, io.csrPort.rdata,
                Mux(dec.is_jal || dec.is_jalr, link_value, alu_result)))
  regfile.io.wr_en   := (s1_fire && dec.rd_wen) || (s1_fire && isMemLoadWB)
  regfile.io.wr_addr := Mux(isMemLoadWB, memLoadRd, dec.rd)
  regfile.io.wr_data := wb_data

  // ── PC control ───────────────────────────────────────────────────
  pc_ctrl.io.redirect        := redirect && !stall && !halt_now
  pc_ctrl.io.redirect_target := redirect_target
  pc_ctrl.io.stall           := stall
  pc_ctrl.io.halted          := halt_now

  // ── Engine launch flags ──────────────────────────────────────────
  val is_dma_launch  = s1_fire && (dec.dma_cmd === DMA_LD || dec.dma_cmd === DMA_ST)
  val is_dma_config  = s1_fire && (dec.dma_cmd === DMA_CONFIG)
  val is_mxu0_launch = s1_fire && (dec.mxu_cmd =/= MXU_NONE) && (dec.mxu_sel === MXUSEL_0)
  val is_mxu1_launch = s1_fire && (dec.mxu_cmd =/= MXU_NONE) && (dec.mxu_sel === MXUSEL_1)
  val is_vpu_launch  = s1_fire && (dec.vpu_cmd =/= VPU_NONE)
  val is_xlu_launch  = s1_fire && (dec.xlu_cmd =/= XLU_NONE)
  val is_lsu_launch  = s1_fire && dec.is_lsu

  when(is_dma_config) { dmaBaseReg := rs1_data }

  // ── DMA command ──────────────────────────────────────────────────
  val isDmaLoad = dec.dma_cmd === DMA_LD
  io.dmaCmd.valid         := is_dma_launch
  io.dmaCmd.bits.op       := dec.dma_cmd
  io.dmaCmd.bits.vmemAddr := Mux(isDmaLoad, rd_data, rs1_data)
  io.dmaCmd.bits.addr     := Mux(isDmaLoad, rs1_data + dmaBaseReg, rd_data + dmaBaseReg)
  io.dmaCmd.bits.size     := rs2_data
  io.dmaCmd.bits.channel  := dec.funct3

  // ── MXU0/1 commands ──────────────────────────────────────────────
  io.mxu0Cmd.valid := is_mxu0_launch; io.mxu0Cmd.bits := buildMxuCmd(dec)
  io.mxu1Cmd.valid := is_mxu1_launch; io.mxu1Cmd.bits := buildMxuCmd(dec)

  private def buildMxuCmd(d: DecodedInstr): MxuCmd = {
    val cmd = Wire(new MxuCmd)
    cmd.op := d.mxu_cmd; cmd.trfBank := d.vs1
    cmd.accSel := d.vd(0); cmd.weightSlot := d.vd(0); cmd.scaleE8M0 := 0.U
    switch(d.mxu_cmd) {
      is(MXU_PUSH_WEIGHT)  { cmd.trfBank := d.vs1; cmd.weightSlot := d.vd(0) }
      is(MXU_PUSH_ACC_FP8) { cmd.trfBank := d.vs1; cmd.accSel := d.vd(0) }
      is(MXU_PUSH_ACC_BF16){ cmd.trfBank := d.vs1; cmd.accSel := d.vd(0) }
      is(MXU_POP_FP8) {
        cmd.trfBank := d.vd; cmd.accSel := d.vs2(0)
        cmd.scaleE8M0 := scaleRF.io.regs(d.vs1(4, 0))
      }
      is(MXU_POP_BF16)   { cmd.trfBank := d.vd; cmd.accSel := d.vs2(0) }
      is(MXU_MATMUL)      { cmd.trfBank := d.vs1; cmd.accSel := d.vd(0); cmd.weightSlot := d.vs2(0) }
      is(MXU_MATMUL_ACC)  { cmd.trfBank := d.vs1; cmd.accSel := d.vd(0); cmd.weightSlot := d.vs2(0) }
    }
    cmd
  }

  io.vpuCmd.valid     := is_vpu_launch
  io.vpuCmd.bits.op   := dec.vpu_cmd
  io.vpuCmd.bits.vd   := dec.vd
  io.vpuCmd.bits.vs1  := dec.vs1
  io.vpuCmd.bits.vs2  := dec.vs2
  io.vpuCmd.bits.imm  := dec.vi_imm

  io.xluCmd.valid        := is_xlu_launch
  io.xluCmd.bits.op      := dec.xlu_cmd
  io.xluCmd.bits.dstBank := dec.vd
  io.xluCmd.bits.srcBank := dec.vs1

  private val lineOffBits = log2Ceil(spP.wordsPerLine)
  io.lsuCmd.valid             := is_lsu_launch
  io.lsuCmd.bits.op           := dec.lsu_cmd
  io.lsuCmd.bits.trfBank      := dec.vd
  io.lsuCmd.bits.vmemLineAddr := alu_result(spP.lineAddrBits + lineOffBits - 1, lineOffBits)

  io.scaleRegs := scaleRF.io.regs
  io.halted    := halt_now
}

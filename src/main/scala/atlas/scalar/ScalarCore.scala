/*
ScalarCore.scala — 2-stage pipeline scalar controller (software-scheduled).

All VMEM access goes through the LSU:
  - Scalar loads/stores: via scalarMemCmd/scalarMemResp
  - Tensor transfers: via lsuCmd
  - DMA: separate path (not through LSU)

Pipeline overview
  Stage 0: Fetch from IMEM
  Stage 1: Decode / Execute / Writeback

Software-scheduled hazard model:
  The frontend never stalls for resource hazards.  The compiler / programmer
  is responsible for scheduling instructions so that no bank conflicts,
  engine-busy violations, or command-backpressure violations occur.
  Hardware assertions fire on any such violation to aid debugging.

  The only conditions that stall the pipeline are:
    DELAY    — stalls for `imm` cycles.
    DMA.WAIT — stalls until the targeted DMA channel is no longer busy.

Scalar stores go directly to the LSU (no store buffer).  Software must
ensure the LSU is idle before issuing a scalar load or store.
*/

package atlas.scalar

import chisel3._
import chisel3.util._
import ScalarISA._
import atlas.common.VmemParams
import atlas.lsu.LsuScalarCmd

object AtlasMemMap {
  val IMEM_BASE      = 0x0002_0000L
  val IMEM_SIZE      = 0x10000
  val IMEM_WORDS     = IMEM_SIZE / 4
  val IMEM_ADDR_BITS = log2Ceil(IMEM_WORDS)
  val CSR_BASE       = 0x0003_0000L
  val CSR_SIZE       = 0x20
  val VMEM_BASE      = 0x2000_0000L
  val DRAM_BASE      = 0x8000_0000L
}

object StallType {
  val WIDTH     = 3
  val NONE      = 0.U(WIDTH.W)
  val DMA_WAIT  = 1.U(WIDTH.W)
  val ENGINE    = 2.U(WIDTH.W)
  val MREG_BANK = 3.U(WIDTH.W)
  val MEM_LOAD  = 4.U(WIDTH.W)
  val MEM_STORE = 5.U(WIDTH.W)
  val DELAY     = 6.U(WIDTH.W)
  val CMD_READY = 7.U(WIDTH.W)
}

/** Scalar controller coordinating fetch, decode, execute, and engine launch.
  *
  * @param spP  Shared VMEM / scalar-memory geometry parameters.
  */
class ScalarCore(spP: VmemParams) extends Module {
  val io = IO(new ScalarCoreIO(spP))

  val pc_ctrl    = Module(new PcControl(0))
  val decoder    = Module(new ScalarDecoder)
  val regfile    = Module(new ScalarRegFile)
  val alu        = Module(new ScalarALU)
  val branch     = Module(new BranchUnit)
  val scaleRF    = Module(new ScalingFactorRegFile)

  val halted     = RegInit(true.B)
  val dmaBaseReg = RegInit(0.U(32.W))
  io.dmaBaseReg := dmaBaseReg

  val s1_hold_active = RegInit(false.B)
  val s1_instr_hold  = RegInit(0.U(32.W))
  val memLoadPending = RegInit(false.B)
  val memLoadCmd     = Reg(UInt(4.W))
  val memLoadRd      = Reg(UInt(5.W))
  val memLoadByteOff = Reg(UInt(2.W))

  val inStall     = RegInit(false.B)
  val stallTypeR  = RegInit(StallType.NONE)

  val delayCounter = RegInit(0.U(12.W))

  // ==============================================================
  // Helper functions
  // ==============================================================

  private def instrMregReadHazard(dec: DecodedInstr): Bool = {
    val wb = io.mregWriteBusy
    val h  = WireDefault(false.B)
    when(dec.mxu_cmd === MXU_PUSH_WEIGHT || dec.mxu_cmd === MXU_PUSH_ACC_FP8 ||
         dec.mxu_cmd === MXU_MATMUL      || dec.mxu_cmd === MXU_MATMUL_ACC) {
      h := wb(dec.vs1)
    }
    when(dec.mxu_cmd === MXU_PUSH_ACC_BF16) {
      h := wb(dec.vs1) || wb(dec.vs1 + 1.U)
    }
    when(dec.is_lsu && dec.lsu_cmd === LSU_VSTORE) {
      h := wb(dec.vd)
    }
    when(dec.xlu_cmd =/= XLU_NONE) {
      h := wb(dec.vs1)
    }
    h
  }

  private def instrMregWriteHazard(dec: DecodedInstr): Bool = {
    val rb = io.mregReadBusy
    val wb = io.mregWriteBusy
    val h  = WireDefault(false.B)
    when(dec.mxu_cmd === MXU_POP_FP8) {
      h := rb(dec.vd) || wb(dec.vd)
    }
    when(dec.mxu_cmd === MXU_POP_BF16) {
      h := rb(dec.vd)         || wb(dec.vd) ||
           rb(dec.vd + 1.U)   || wb(dec.vd + 1.U)
    }
    when(dec.is_lsu && dec.lsu_cmd === LSU_VLOAD) {
      h := wb(dec.vd)
    }
    when(dec.xlu_cmd =/= XLU_NONE) {
      h := rb(dec.vd) || wb(dec.vd)
    }
    h
  }

  private def buildMxuCmd(d: DecodedInstr): MxuCmd = {
    val cmd = Wire(new MxuCmd)
    cmd.op := d.mxu_cmd
    cmd.mregBank := d.vs1
    cmd.accSel := d.vd(0)
    cmd.weightSlot := d.vd(0)
    cmd.scaleE8M0 := 0.U

    switch(d.mxu_cmd) {
      is(MXU_PUSH_WEIGHT)   { cmd.mregBank := d.vs1; cmd.weightSlot := d.vd(0) }
      is(MXU_PUSH_ACC_FP8)  { cmd.mregBank := d.vs1; cmd.accSel := d.vd(0) }
      is(MXU_PUSH_ACC_BF16) { cmd.mregBank := d.vs1; cmd.accSel := d.vd(0) }
      is(MXU_POP_FP8) {
        cmd.mregBank := d.vd
        cmd.accSel := d.vs2(0)
        cmd.scaleE8M0 := scaleRF.io.regs(d.vs1(4, 0))
      }
      is(MXU_POP_BF16) {
        cmd.mregBank := d.vd
        cmd.accSel := d.vs2(0)
      }
      is(MXU_MATMUL) {
        cmd.mregBank := d.vs1
        cmd.accSel := d.vd(0)
        cmd.weightSlot := d.vs2(0)
      }
      is(MXU_MATMUL_ACC) {
        cmd.mregBank := d.vs1
        cmd.accSel := d.vd(0)
        cmd.weightSlot := d.vs2(0)
      }
    }
    cmd
  }

  // ==============================================================
  // Stage 0: Fetch
  // ==============================================================

  io.imemFetch.addr := pc_ctrl.io.pc(AtlasMemMap.IMEM_ADDR_BITS - 1, 0)

  // ==============================================================
  // Stage 1: Decode / register read
  // ==============================================================

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

  // ==============================================================
  // Stall conditions (only DELAY and DMA.WAIT)
  // ==============================================================

  val delay_stall = delayCounter =/= 0.U

  val dma_wait_stall = s1_valid && (dec.dma_cmd === DMA_WAIT) && io.dma_busy(dec.funct3)

  // ==============================================================
  // Halt logic
  // ==============================================================

  val is_illegal = s1_valid && dec.illegal
  val branch_in_delay_slot =
    pc_ctrl.io.in_delay_slot && (dec.br_type =/= BR_NONE || dec.is_jal || dec.is_jalr)

  val illegal_detected = is_illegal || (s1_valid && branch_in_delay_slot)
  val ecall_ebreak     = s1_valid && (dec.is_ecall || dec.is_ebreak)

  val halt_now = halted || illegal_detected || ecall_ebreak
  when((illegal_detected || ecall_ebreak) && !halted) { halted := true.B }

  // ==============================================================
  // Full stall computation
  // ==============================================================

  val fullStall = delay_stall || dma_wait_stall

  val stall = fullStall

  when(reset.asBool || halted) {
    inStall    := false.B
    stallTypeR := StallType.NONE
  }.elsewhen(!inStall && fullStall) {
    inStall := true.B
    when(delay_stall) {
      stallTypeR := StallType.DELAY
    }.otherwise {
      stallTypeR := StallType.DMA_WAIT
    }
  }.elsewhen(inStall && !fullStall) {
    inStall    := false.B
    stallTypeR := StallType.NONE
  }

  // ==============================================================
  // Execute units: ALU + branch
  // ==============================================================

  val s1_fire = s1_valid && !stall && !halt_now

  val alu_a = MuxLookup(dec.op1_sel, rs1_data)(Seq(
    OP1_RS1 -> rs1_data,
    OP1_PC  -> s1_pc,
    OP1_ZERO -> 0.U
  ))
  val alu_b = MuxLookup(dec.op2_sel, rs2_data)(Seq(
    OP2_RS2  -> rs2_data,
    OP2_IMM  -> dec.imm.asUInt,
    OP2_ONE  -> 1.U,
    OP2_ZERO -> 0.U
  ))

  alu.io.a  := alu_a
  alu.io.b  := alu_b
  alu.io.op := dec.alu_fn
  val alu_result = alu.io.out

  branch.io.branch_type := dec.br_type
  branch.io.rs1 := rs1_data
  branch.io.rs2 := rs2_data

  val branch_target = (s1_pc.asSInt + (dec.imm >> 1)).asUInt
  val jal_target    = (s1_pc.asSInt + (dec.imm >> 1)).asUInt
  val jalr_target   = (rs1_data.asSInt + dec.imm).asUInt

  val is_branch_taken = s1_valid && (dec.br_type =/= BR_NONE) && branch.io.taken
  val is_jal  = s1_valid && dec.is_jal
  val is_jalr = s1_valid && dec.is_jalr

  val redirect = is_branch_taken || is_jal || is_jalr
  val redirect_target =
    Mux(is_jalr, jalr_target, Mux(is_jal, jal_target, branch_target))
  val link_value = s1_pc + 1.U

  // ==============================================================
  // Software-scheduling assertions (printf + assert for visibility)
  //
  // Engine cmd_ready assertions are now in the sequencers/engines
  // themselves, where the structural hazards are directly observable.
  // ==============================================================

  when(s1_fire && instrMregReadHazard(dec)) {
    printf("ASSERT FAIL [pc=0x%x]: MREG RAW hazard: instruction reads bank with pending write\n", s1_pc)
  }
  assert(!(s1_fire && instrMregReadHazard(dec)))

  when(s1_fire && instrMregWriteHazard(dec)) {
    printf("ASSERT FAIL [pc=0x%x]: MREG WAR/WAW hazard: instruction writes bank with pending read/write\n", s1_pc)
  }
  assert(!(s1_fire && instrMregWriteHazard(dec)))

  when(s1_fire && dec.vpu_cmd =/= VPU_NONE && io.vpu_status.busy) {
    printf("ASSERT FAIL [pc=0x%x]: VPU command issued while VPU is busy\n", s1_pc)
  }
  assert(!(s1_fire && dec.vpu_cmd =/= VPU_NONE && io.vpu_status.busy))

  when(s1_fire && dec.is_lsu && io.lsu_busy) {
    printf("ASSERT FAIL [pc=0x%x]: LSU command issued while LSU is busy\n", s1_pc)
  }
  assert(!(s1_fire && dec.is_lsu && io.lsu_busy))

  when(s1_fire && dec.is_mem_load && memLoadPending) {
    printf("ASSERT FAIL [pc=0x%x]: scalar load issued while prior load response pending\n", s1_pc)
  }
  assert(!(s1_fire && dec.is_mem_load && memLoadPending))

  when(s1_fire && dec.is_mem_load && io.lsu_busy) {
    printf("ASSERT FAIL [pc=0x%x]: scalar load issued while LSU is busy\n", s1_pc)
  }
  assert(!(s1_fire && dec.is_mem_load && io.lsu_busy))

  when(s1_fire && dec.is_mem_store && io.lsu_busy) {
    printf("ASSERT FAIL [pc=0x%x]: scalar store issued while LSU is busy\n", s1_pc)
  }
  assert(!(s1_fire && dec.is_mem_store && io.lsu_busy))

  // ==============================================================
  // Delay counter
  // ==============================================================

  when(reset.asBool || halt_now) {
    delayCounter := 0.U
  }.elsewhen(s1_fire && dec.is_delay) {
    delayCounter := dec.imm(11, 0).asUInt
  }.elsewhen(delayCounter =/= 0.U) {
    delayCounter := delayCounter - 1.U
  }

  // ==============================================================
  // Instruction hold
  // ==============================================================

  when(reset.asBool || halt_now) {
    s1_hold_active := false.B
    s1_instr_hold  := 0.U
  }.elsewhen(stall && !s1_hold_active) {
    s1_instr_hold  := io.imemFetch.rdata
    s1_hold_active := true.B
  }.elsewhen(!stall) {
    s1_hold_active := false.B
  }

  // ==============================================================
  // Scalar memory path via LSU (no store buffer — direct issue)
  // ==============================================================

  val issueScalarLoad =
    s1_fire && dec.is_mem_load && !memLoadPending

  val issueScalarStore = s1_fire && dec.is_mem_store

  val storeByteOff = alu_result(1, 0)
  val storeData = Wire(UInt(32.W))
  val storeMask = Wire(UInt(4.W))
  storeData := rs2_data
  storeMask := 0.U

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

  when(issueScalarStore) {
    io.scalarMemCmd.valid         := true.B
    io.scalarMemCmd.bits.isStore  := true.B
    io.scalarMemCmd.bits.byteAddr := alu_result(spP.byteAddrBits - 1, 0)
    io.scalarMemCmd.bits.wdata    := storeData
    io.scalarMemCmd.bits.wmask    := storeMask
  }.elsewhen(issueScalarLoad) {
    io.scalarMemCmd.valid         := true.B
    io.scalarMemCmd.bits.isStore  := false.B
    io.scalarMemCmd.bits.byteAddr := alu_result(spP.byteAddrBits - 1, 0)
    io.scalarMemCmd.bits.wdata    := 0.U
    io.scalarMemCmd.bits.wmask    := 0.U
  }.otherwise {
    io.scalarMemCmd.valid         := false.B
    io.scalarMemCmd.bits.isStore  := false.B
    io.scalarMemCmd.bits.byteAddr := 0.U
    io.scalarMemCmd.bits.wdata    := 0.U
    io.scalarMemCmd.bits.wmask    := 0.U
  }

  when(issueScalarLoad) {
    memLoadPending := true.B
    memLoadCmd     := dec.mem_cmd
    memLoadRd      := dec.rd
    memLoadByteOff := alu_result(1, 0)
  }
  when(memLoadPending && s1_fire) {
    memLoadPending := false.B
  }

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

  // ==============================================================
  // CSR + scale registers + writeback
  // ==============================================================

  io.csrPort.op    := Mux(s1_valid && dec.is_csr, dec.csr_cmd, CSR_NONE)
  io.csrPort.valid := s1_fire && dec.is_csr
  io.csrPort.wdata := Mux(
    dec.csr_cmd === CSR_RWI || dec.csr_cmd === CSR_RSI || dec.csr_cmd === CSR_RCI,
    dec.imm.asUInt,
    rs1_data
  )
  io.csrPort.halted      := halt_now
  io.csrPort.set_illegal := illegal_detected && !halted
  io.csrPort.set_ecall   := s1_valid && dec.is_ecall && !halted
  io.csrPort.set_ebreak  := s1_valid && dec.is_ebreak && !halted
  io.csrPort.illegal_pc  := s1_pc
  io.csrPort.inst_retire := s1_fire

  val seldWriteEn = memLoadPending && s1_fire && memLoadCmd === MEM_SELD
  scaleRF.io.writeEn   := (s1_fire && dec.is_seli) || seldWriteEn
  scaleRF.io.writeIdx  := Mux(seldWriteEn, memLoadRd, dec.rd)
  scaleRF.io.writeData := Mux(seldWriteEn, memLoadResult(7, 0), dec.imm(7, 0))

  val isMemLoadWB = memLoadPending && memLoadCmd =/= MEM_SELD
  val wb_data = Mux(
    isMemLoadWB, memLoadResult,
    Mux(dec.is_csr, io.csrPort.rdata,
    Mux(dec.is_jal || dec.is_jalr, link_value, alu_result))
  )

  regfile.io.wr_en   := (s1_fire && dec.rd_wen) || (s1_fire && isMemLoadWB)
  regfile.io.wr_addr := Mux(isMemLoadWB, memLoadRd, dec.rd)
  regfile.io.wr_data := wb_data

  // ==============================================================
  // PC control
  // ==============================================================

  pc_ctrl.io.redirect        := redirect && !stall && !halt_now
  pc_ctrl.io.redirect_target := redirect_target
  pc_ctrl.io.stall           := stall
  pc_ctrl.io.halted          := halt_now
  pc_ctrl.io.softReset       := io.softReset

  // ==============================================================
  // Engine launch flags
  // ==============================================================

  val is_dma_launch  = s1_fire && (dec.dma_cmd === DMA_LD || dec.dma_cmd === DMA_ST)
  val is_dma_config  = s1_fire && (dec.dma_cmd === DMA_CONFIG)
  val is_mxu0_launch = s1_fire && (dec.mxu_cmd =/= MXU_NONE) && (dec.mxu_sel === MXUSEL_0)
  val is_mxu1_launch = s1_fire && (dec.mxu_cmd =/= MXU_NONE) && (dec.mxu_sel === MXUSEL_1)
  val is_vpu_launch  = s1_fire && (dec.vpu_cmd =/= VPU_NONE)
  val is_xlu_launch  = s1_fire && (dec.xlu_cmd =/= XLU_NONE)
  val is_lsu_launch  = s1_fire && dec.is_lsu

  when(is_dma_config) {
    dmaBaseReg := rs1_data
  }

  // ==============================================================
  // Engine command generation
  // ==============================================================

  val isDmaLoad = dec.dma_cmd === DMA_LD
  io.dmaCmd.valid         := is_dma_launch
  io.dmaCmd.bits.op       := dec.dma_cmd
  io.dmaCmd.bits.vmemAddr := Mux(isDmaLoad, rd_data, rs1_data)
  io.dmaCmd.bits.addr     := Mux(isDmaLoad, rs1_data + dmaBaseReg, rd_data + dmaBaseReg)
  io.dmaCmd.bits.size     := rs2_data
  io.dmaCmd.bits.channel  := dec.funct3

  io.mxu0Cmd.valid := is_mxu0_launch
  io.mxu0Cmd.bits  := buildMxuCmd(dec)

  io.mxu1Cmd.valid := is_mxu1_launch
  io.mxu1Cmd.bits  := buildMxuCmd(dec)

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
  io.lsuCmd.bits.mregBank     := dec.vd
  io.lsuCmd.bits.vmemLineAddr := alu_result(spP.lineAddrBits + lineOffBits - 1, lineOffBits)

  io.scaleRegs := scaleRF.io.regs
  io.halted    := halt_now

  when(io.softReset) {
    halted          := false.B
    dmaBaseReg      := 0.U
    s1_hold_active  := false.B
    s1_instr_hold   := 0.U
    memLoadPending  := false.B
  }
}

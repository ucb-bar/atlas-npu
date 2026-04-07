/*
ScalarCore.scala — 2-stage pipeline scalar controller.

All VMEM access goes through the LSU:
  - Scalar loads/stores: via scalarMemCmd/scalarMemResp
  - Tensor transfers: via lsuCmd
  - DMA: separate path (not through LSU)

Pipeline overview
  Stage 0: Fetch from IMEM
  Stage 1: Decode / Execute / Writeback

Hazard model:
  MXU0/MXU1/XLU — backpressure via cmd_ready; no engine-level stall.
  VPU/LSU — fire-and-forget; stall while engine busy.
  MREG banks — direction-aware:
    Read  bank X → stall only if writeBusy(X)             (RAW)
    Write bank X → stall if readBusy(X) || writeBusy(X)   (WAR / WAW)

Scalar store buffer:
  A 1-entry posted-store buffer decouples scalar stores from the pipeline.
  Stores capture into the buffer and retire immediately.  The buffer drains
  to LSU asynchronously when the LSU is idle.

Posted DMA wait:
  DMA.WAIT fires immediately, recording the channel in a pending-wait mask.
  The pending bit clears when dma_busy(channel) goes false.  All VMEM-
  accessing instructions stall while any pending wait is active: VLOAD and
  VSTORE via engineHazard, scalar loads via issueScalarLoad, scalar store
  buffer drain via drainStoreBuf, and DMA launches via engineHazard.
  Non-VMEM instructions (ALU, branches, CSR, MXU/XLU commands, DMA.CONFIG)
  execute freely during the wait.

Halt drain:
  ECALL and EBREAK stall the pipeline until all async work completes
  (pending DMA waits + store buffer), guaranteeing that all side effects
  are visible before the core halts.  Illegal instructions halt immediately.
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
  val DMA_WAIT  = 1.U(WIDTH.W)  // reserved (posted model — no longer stalls)
  val ENGINE    = 2.U(WIDTH.W)
  val MREG_BANK = 3.U(WIDTH.W)
  val MEM_LOAD  = 4.U(WIDTH.W)
  val MEM_STORE = 5.U(WIDTH.W)
  val DELAY     = 6.U(WIDTH.W)
  val CMD_READY = 7.U(WIDTH.W)
}

object EngineId {
  val VPU  = 2.U(3.W)
  val LSU  = 4.U(3.W)
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
  val stallParamR = Reg(UInt(6.W))

  val delayCounter = RegInit(0.U(12.W))

  // Posted-store buffer: captures scalar stores for asynchronous LSU drain.
  val storeBufValid = RegInit(false.B)
  val storeBufCmd   = Reg(new LsuScalarCmd(spP))

  // Posted DMA wait mask: per-channel pending bits set by DMA.WAIT, cleared
  // when the corresponding dma_busy flag goes low.
  val dmaWaitPending = RegInit(VecInit(Seq.fill(8)(false.B)))
  val anyDmaWaitPending = dmaWaitPending.reduce(_ || _)

  // ==============================================================
  // DMA wait tracking: clear pending bits when channels complete
  // ==============================================================

  for (ch <- 0 until 8) {
    when(!io.dma_busy(ch)) {
      dmaWaitPending(ch) := false.B
    }
  }

  // ==============================================================
  // Helper functions
  // ==============================================================

  /** True when the decoded instruction reads a MREG bank that has a
    * pending asynchronous write (RAW hazard). */
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

  /** True when the decoded instruction writes a MREG bank that is either
    * being read or written by another engine (WAR / WAW hazard).
    *
    * Exception: VLOAD only checks writeBusy (WAW), not readBusy (WAR).
    * The LSU writes rows sequentially and always has a head start over
    * any MXU that reads the same bank, so the row-level overlap is
    * structurally safe.  The WAR check is kept for non-LSU writers
    * (POP, XLU) where no such structural guarantee exists. */
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

  private def engineHazard(dec: DecodedInstr): Bool = {
    (dec.vpu_cmd =/= VPU_NONE && io.vpu_status.busy) ||
    (dec.is_lsu && (io.lsu_busy || storeBufValid || anyDmaWaitPending)) ||
    ((dec.dma_cmd === DMA_LD || dec.dma_cmd === DMA_ST) && anyDmaWaitPending)
  }

  private def engineHazardId(dec: DecodedInstr): UInt = {
    MuxCase(EngineId.VPU, Seq(
      (dec.vpu_cmd =/= VPU_NONE) -> EngineId.VPU,
      (dec.is_lsu)               -> EngineId.LSU,
      (dec.dma_cmd === DMA_LD || dec.dma_cmd === DMA_ST) -> EngineId.LSU
    ))
  }

  private def fastStallStillActive(stallType: UInt, stallParam: UInt): Bool = {
    val active = WireDefault(false.B)
    switch(stallType) {
      is(StallType.ENGINE) {
        active := MuxLookup(stallParam(2, 0), false.B)(Seq(
          EngineId.VPU  -> io.vpu_status.busy,
          EngineId.LSU  -> (io.lsu_busy || storeBufValid || anyDmaWaitPending)
        ))
      }
      is(StallType.MREG_BANK) {
        active := instrMregReadHazard(dec) || instrMregWriteHazard(dec)
      }
      is(StallType.MEM_LOAD) {
        active := !memLoadPending
      }
      is(StallType.MEM_STORE) {
        active := storeBufValid
      }
      is(StallType.DELAY) {
        active := delayCounter =/= 0.U
      }
      is(StallType.CMD_READY) {
        active := MuxLookup(stallParam(1, 0), false.B)(Seq(
          0.U -> !io.mxu0_cmd_ready,
          1.U -> !io.mxu1_cmd_ready,
          2.U -> !io.xlu_cmd_ready
        ))
      }
    }
    active
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
  // Hazard + stall detection
  // ==============================================================

  val mreg_read_haz  = s1_valid && instrMregReadHazard(dec)
  val mreg_write_haz = s1_valid && instrMregWriteHazard(dec)
  val mreg_hazard    = mreg_read_haz || mreg_write_haz

  val engine_hazard = s1_valid && engineHazard(dec)

  val cmd_ready_stall = s1_valid && (
    (dec.mxu_cmd =/= MXU_NONE && dec.mxu_sel === MXUSEL_0 && !io.mxu0_cmd_ready) ||
    (dec.mxu_cmd =/= MXU_NONE && dec.mxu_sel === MXUSEL_1 && !io.mxu1_cmd_ready) ||
    (dec.xlu_cmd =/= XLU_NONE && !io.xlu_cmd_ready)
  )

  val mem_load_stall  = s1_valid && dec.is_mem_load && !memLoadPending
  val mem_store_stall = s1_valid && dec.is_mem_store && storeBufValid

  val delay_stall = delayCounter =/= 0.U

  val fastStallActive = fastStallStillActive(stallTypeR, stallParamR)

  // ==============================================================
  // Halt logic: ECALL/EBREAK drain all async work before halting
  // ==============================================================

  val is_illegal = s1_valid && dec.illegal
  val branch_in_delay_slot =
    pc_ctrl.io.in_delay_slot && (dec.br_type =/= BR_NONE || dec.is_jal || dec.is_jalr)

  val illegal_detected = is_illegal || (s1_valid && branch_in_delay_slot)
  val ecall_ebreak     = s1_valid && (dec.is_ecall || dec.is_ebreak)

  // Async work that must drain before ECALL/EBREAK can halt the core.
  val asyncBusy = anyDmaWaitPending || storeBufValid

  // ECALL/EBREAK stalls the pipeline while async work is in progress.
  // Illegal instructions halt immediately (they indicate a bug).
  val pending_halt = ecall_ebreak && asyncBusy

  val halt_now = halted || illegal_detected || (ecall_ebreak && !asyncBusy)
  when((illegal_detected || (ecall_ebreak && !asyncBusy)) && !halted) { halted := true.B }

  // ==============================================================
  // Full stall computation
  // ==============================================================

  val fullStall =
    cmd_ready_stall || engine_hazard || mreg_hazard ||
    mem_load_stall || mem_store_stall || delay_stall ||
    pending_halt

  val stall = Mux(inStall && fastStallActive, true.B, fullStall)

  when(reset.asBool || halted) {
    inStall    := false.B
    stallTypeR := StallType.NONE
  }.elsewhen(!inStall && fullStall) {
    inStall := true.B
    when(cmd_ready_stall) {
      stallTypeR  := StallType.CMD_READY
      stallParamR := Mux(dec.xlu_cmd =/= XLU_NONE, 2.U, dec.mxu_sel)
    }.elsewhen(mreg_hazard) {
      stallTypeR := StallType.MREG_BANK
    }.elsewhen(engine_hazard) {
      stallTypeR  := StallType.ENGINE
      stallParamR := engineHazardId(dec)
    }.elsewhen(mem_load_stall) {
      stallTypeR := StallType.MEM_LOAD
    }.elsewhen(mem_store_stall) {
      stallTypeR := StallType.MEM_STORE
    }.elsewhen(delay_stall) {
      stallTypeR := StallType.DELAY
    }.elsewhen(pending_halt) {
      stallTypeR := StallType.DMA_WAIT
    }
  }.elsewhen(inStall && !fastStallActive) {
    inStall    := false.B
    stallTypeR := StallType.NONE
  }

  // ==============================================================
  // Execute units: ALU + branch
  // ==============================================================

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

  val s1_fire = s1_valid && !stall && !halt_now

  // ==============================================================
  // Posted DMA wait: DMA.WAIT fires and sets the channel pending bit
  // ==============================================================

  when(s1_fire && dec.dma_cmd === DMA_WAIT) {
    dmaWaitPending(dec.funct3) := true.B
  }

  // ==============================================================
  // Delay counter: hold pipeline for imm cycles after DELAY fires
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
  // Scalar memory path via LSU (with posted-store buffer)
  // ==============================================================

  val canIssueLsuScalar = !io.lsu_busy

  // Scalar loads go through LSU → VMEM, the same VMEM that DMA writes to.
  // Must wait for pending DMA waits to avoid reading partially-written data.
  val issueScalarLoad =
    s1_valid && dec.is_mem_load && !memLoadPending &&
    canIssueLsuScalar && !storeBufValid && !anyDmaWaitPending && !halt_now

  val issueScalarStore = s1_fire && dec.is_mem_store

  // Scalar stores go through LSU → VMEM.  Must wait for pending DMA waits
  // to avoid writing VMEM while DMA.STORE is reading from it.
  val drainStoreBuf = storeBufValid && !io.lsu_busy && !anyDmaWaitPending

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
    storeBufValid        := true.B
    storeBufCmd.isStore  := true.B
    storeBufCmd.byteAddr := alu_result(spP.byteAddrBits - 1, 0)
    storeBufCmd.wdata    := storeData
    storeBufCmd.wmask    := storeMask
  }

  when(drainStoreBuf) {
    storeBufValid := false.B
  }

  when(drainStoreBuf) {
    io.scalarMemCmd.valid := true.B
    io.scalarMemCmd.bits  := storeBufCmd
  }.otherwise {
    io.scalarMemCmd.valid         := issueScalarLoad
    io.scalarMemCmd.bits.isStore  := false.B
    io.scalarMemCmd.bits.byteAddr := alu_result(spP.byteAddrBits - 1, 0)
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
  io.csrPort.set_ecall   := s1_valid && dec.is_ecall && !asyncBusy && !halted
  io.csrPort.set_ebreak  := s1_valid && dec.is_ebreak && !asyncBusy && !halted
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

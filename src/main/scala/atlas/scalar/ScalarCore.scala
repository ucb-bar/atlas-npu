/*
ScalarCore.scala — 2-stage pipeline scalar controller (software-scheduled).

All VMEM access goes through the LSU:
  - Scalar loads/stores: via scalarMemCmd/scalarMemResp
  - Tensor transfers: via lsuCmd
  - DMA: separate path (not through LSU)

Pipeline overview
  Stage 0: Fetch from IMEM
  Stage 1: Decode / Execute / Writeback

Interlocked hazard model:
  The instruction in S1 stalls until it can issue without a structural or
  data hazard (see "Interlock" below): its unit cannot take the command, an
  MREG reservation or physical port is taken, an LSU path holds its VMEM
  bank, a DMA would overtake an LSU transfer, or a scalar load result or
  write port is pending. Programs need no DELAY spacing; the unit assertions
  remain as checks of the interlock.

  The pipeline also stalls for:
    DELAY    — stalls for `imm` cycles (still legal, no longer required).
    DMA.WAIT — stalls until the targeted DMA channel is no longer busy. This
               stays the architectural fence between a DMA transfer and the
               instructions that consume or overwrite its VMEM data.

Scalar stores are pipelined masked writes (no RMW).
Scalar loads have fixed 3-cycle latency.
VLOAD/VSTORE have fixed mregRows+2 cycle latency.
Scalar, VLOAD, and VSTORE LSU paths operate concurrently; software must
ensure no same-bank VMEM conflicts (asserted in Vmem).
*/

package atlas.scalar

import chisel3._
import chisel3.util._
import ScalarISA._
import atlas.common.VmemParams
import atlas.lsu.LsuScalarCmd

object AtlasMemMap {
  val IMEM_BASE      = 0x0002_0000L
  val IMEM_SIZE      = 0x0002_0000  // 128 KiB
  val IMEM_WORDS     = IMEM_SIZE / 4
  val IMEM_ADDR_BITS = log2Ceil(IMEM_WORDS)

  val CSR_BASE        = 0x0004_0000L
  val CSR_SIZE        = 0x20    // Total size of the actual CSR registers we’ve implemented
  val CSR_WINDOW_SIZE = 0x1000  // Size of the address range reserved for CSRs on the bus (4 KiB page). Only the first CSR_SIZE bytes are used; the rest is unused space
  val CSR_ADDR_BITS   = log2Ceil(CSR_WINDOW_SIZE)

  val VMEM_BASE      = 0x2000_0000L
  val VMEM_SIZE      = 0x0018_0000  // 1.5 MiB
  val VMEM_NUM_BANKS = 6

  val DRAM_BASE      = 0x8000_0000L
}

/** @param hasNvfp4  the NVFP4 MXU and scale sidecar are present (.nv and *.scale legal)
  */
class ScalarCore(spP: VmemParams, hasNvfp4: Boolean = false) extends Module {
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
  val memLoadRespValid   = RegInit(false.B)
  val memLoadRespWord    = RegInit(0.U(32.W))
  val memLoadRespCmd     = RegInit(0.U(4.W))
  val memLoadRespRd      = RegInit(0.U(5.W))
  val memLoadRespByteOff = RegInit(0.U(2.W))
  val scalarMemCmdReg    = RegInit(0.U.asTypeOf(new LsuScalarCmd(spP)))
  val scalarMemCmdValid  = RegInit(false.B)

  val delayCounter = RegInit(0.U(12.W))

  // ==============================================================
  // Helper functions
  // ==============================================================

  private def vpuIsVli(cmd: UInt): Bool =
    cmd === VPU_LI_ALL || cmd === VPU_LI_ROW ||
    cmd === VPU_LI_COL || cmd === VPU_LI_ONE

  private def vpuIsVliColOrOne(cmd: UInt): Bool =
    cmd === VPU_LI_COL || cmd === VPU_LI_ONE

  private def vpuIsTwoInput(cmd: UInt): Bool =
    cmd === VPU_ADD || cmd === VPU_SUB || cmd === VPU_MUL ||
    cmd === VPU_PAIRMAX || cmd === VPU_PAIRMIN

  private def vpuUsesPackScale(cmd: UInt): Bool = cmd === VPU_FP8PACK

  private def vpuUsesUnpackScale(cmd: UInt): Bool = cmd === VPU_FP8UNPACK

  private def vpuIsPackUnpack(cmd: UInt): Bool =
    vpuUsesPackScale(cmd) || vpuUsesUnpackScale(cmd)

  private def vpuPrimaryReadBank(dec: DecodedInstr): UInt =
    Mux(vpuIsPackUnpack(dec.vpu_cmd), dec.vs2, dec.vs1)

  private def vpuReadsPrimaryPair(cmd: UInt): Bool =
    cmd =/= VPU_NONE && !vpuIsVli(cmd) && !vpuUsesUnpackScale(cmd)

  private def vpuReadsSecondaryPair(cmd: UInt): Bool =
    vpuIsTwoInput(cmd)

  private def vpuWritesPair(cmd: UInt): Bool =
    cmd =/= VPU_NONE && !vpuUsesPackScale(cmd) && !vpuIsVliColOrOne(cmd)

  private def vpuIssueBusy(cmd: UInt): Bool =
    io.vpu_status.issueBusy(cmd)

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
    when(dec.vpu_cmd =/= VPU_NONE) {
      val src1Bank = vpuPrimaryReadBank(dec)
      val src1Haz = wb(src1Bank) ||
        (vpuReadsPrimaryPair(dec.vpu_cmd) && wb(src1Bank + 1.U))
      val src2Haz = vpuReadsSecondaryPair(dec.vpu_cmd) &&
        (wb(dec.vs2) || wb(dec.vs2 + 1.U))
      h := Mux(vpuIsVli(dec.vpu_cmd), false.B, src1Haz || src2Haz)
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
      h := rb(dec.vd) || wb(dec.vd) || rb(dec.vd + 1.U) || wb(dec.vd + 1.U)
    }
    when(dec.is_lsu && dec.lsu_cmd === LSU_VLOAD) {
      h := rb(dec.vd) || wb(dec.vd)
    }
    when(dec.xlu_cmd =/= XLU_NONE) {
      h := rb(dec.vd) || wb(dec.vd)
    }
    when(dec.vpu_cmd =/= VPU_NONE) {
      h := rb(dec.vd) || wb(dec.vd) ||
           (vpuWritesPair(dec.vpu_cmd) && (rb(dec.vd + 1.U) || wb(dec.vd + 1.U)))
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
        cmd.mregBank := d.vd; cmd.accSel := d.vs2(0)
        cmd.scaleE8M0 := scaleRF.io.regs(d.vs1(4, 0))
      }
      is(MXU_POP_BF16) { cmd.mregBank := d.vd; cmd.accSel := d.vs2(0) }
      is(MXU_MATMUL)     { cmd.mregBank := d.vs1; cmd.accSel := d.vd(0); cmd.weightSlot := d.vs2(0) }
      is(MXU_MATMUL_ACC) { cmd.mregBank := d.vs1; cmd.accSel := d.vd(0); cmd.weightSlot := d.vs2(0) }
      is(MXU_NV_PUSH_WEIGHT)   { cmd.mregBank := d.vs1; cmd.weightSlot := d.vd(0) }
      is(MXU_NV_PUSH_ACC_BF16) { cmd.mregBank := d.vs1; cmd.accSel := d.vd(0) }
      is(MXU_NV_PUSH_ACC_FP32) { cmd.mregBank := d.vs1; cmd.accSel := d.vd(0) }
      is(MXU_NV_POP_BF16)      { cmd.mregBank := d.vd;  cmd.accSel := d.vs2(0) }
      is(MXU_NV_POP_FP32)      { cmd.mregBank := d.vd;  cmd.accSel := d.vs2(0) }
      is(MXU_NV_MATMUL)        { cmd.mregBank := d.vs1; cmd.accSel := d.vd(0); cmd.weightSlot := d.vs2(0) }
      is(MXU_NV_MATMUL_ACC)    { cmd.mregBank := d.vs1; cmd.accSel := d.vd(0); cmd.weightSlot := d.vs2(0) }
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

  val delay_stall    = delayCounter =/= 0.U
  val dma_wait_stall = s1_valid && (dec.dma_cmd === DMA_WAIT) && io.dma_busy(dec.funct3)

  // ==============================================================
  // Halt logic
  // ==============================================================

  // Instructions of a unit this configuration does not have are illegal.
  val isNvCmd    = dec.mxu_cmd =/= MXU_NONE && dec.mxu_sel === MXUSEL_0
  val isScaleLsu = dec.is_lsu && (dec.lsu_cmd === LSU_VLOAD_SCALE || dec.lsu_cmd === LSU_VSTORE_SCALE)
  val unitAbsent = !hasNvfp4.B && (isNvCmd || isScaleLsu)
  // A two-source VPU instruction streams both source pairs in the same cycles. Two different
  // pairs in one physical bank (m_i and m_i+32 share bank i) can never be read together, so
  // the instruction is illegal; the same pair twice is fine (one read serves both).
  val vpuSourceBankClash = dec.vpu_cmd =/= VPU_NONE && vpuReadsSecondaryPair(dec.vpu_cmd) &&
    dec.vs1(4, 0) === dec.vs2(4, 0) && dec.vs1 =/= dec.vs2
  // A CSR instruction is illegal on an unmapped CSR, or when it writes a read-only one.
  val csrIllegal = dec.is_csr && io.csrPort.illegal
  val is_illegal = s1_valid && (dec.illegal || unitAbsent || vpuSourceBankClash || csrIllegal)
  val branch_in_delay_slot =
    pc_ctrl.io.in_delay_slot && (dec.br_type =/= BR_NONE || dec.is_jal || dec.is_jalr)
  val illegal_detected = is_illegal || (s1_valid && branch_in_delay_slot)
  val ecall_ebreak     = s1_valid && (dec.is_ecall || dec.is_ebreak)
  val hostStart        = io.execRunWrite && io.execRun
  val hostStop         = io.execRunWrite && !io.execRun

  val halt_now = halted || hostStop || illegal_detected || ecall_ebreak
  when((hostStop || illegal_detected || ecall_ebreak) && !halted) { halted := true.B }

  // ==============================================================
  // ALU + Branch
  // ==============================================================

  val alu_a = MuxLookup(dec.op1_sel, rs1_data)(Seq(
    OP1_RS1 -> rs1_data, OP1_PC -> s1_pc, OP1_ZERO -> 0.U))
  val alu_b = MuxLookup(dec.op2_sel, rs2_data)(Seq(
    OP2_RS2 -> rs2_data, OP2_IMM -> dec.imm.asUInt, OP2_ONE -> 1.U, OP2_ZERO -> 0.U))

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
  val redirect_target = Mux(is_jalr, jalr_target, Mux(is_jal, jal_target, branch_target))
  val link_value = s1_pc + 1.U

  // ==============================================================
  // Interlock
  //
  // The instruction in S1 waits until it can issue without a
  // structural or data hazard, so programs need no DELAY spacing.
  // Each term is a software-scheduling rule that a unit asserts on
  // (npu_spec §4: access windows, same-unit issue spacing, MXU
  // buffer rules). DMA.WAIT stays the architectural fence between
  // a DMA transfer and the instructions that use its data.
  // ==============================================================

  // ── The target unit can take the command ──
  // vload.scale / vstore.scale run on the VLOAD / VSTORE paths.
  val isVloadData = dec.is_lsu && dec.lsu_cmd === LSU_VLOAD
  val isVload   = isVloadData || (dec.is_lsu && dec.lsu_cmd === LSU_VLOAD_SCALE)
  val isVstore  = dec.is_lsu && (dec.lsu_cmd === LSU_VSTORE || dec.lsu_cmd === LSU_VSTORE_SCALE)
  val isDmaXfer = dec.dma_cmd === DMA_LD || dec.dma_cmd === DMA_ST

  val mxuNotReady = dec.mxu_cmd =/= MXU_NONE && Mux(isNvCmd, !io.nvReady, !io.mxu1Ready)
  val vpuNotReady = dec.vpu_cmd =/= VPU_NONE && vpuIssueBusy(dec.vpu_cmd)
  val xluNotReady = dec.xlu_cmd =/= XLU_NONE && io.xlu_busy
  val lsuNotReady = (isVload && io.lsu_vload_busy) || (isVstore && io.lsu_vstore_busy) ||
                    (dec.is_mem_load && (memLoadPending || io.lsu_scalar_busy))
  // A busy channel is illegal to reuse; slots are allocated round-robin and
  // retire out of order, so the next slot must also be free.
  val dmaNotReady = isDmaXfer && (io.dma_busy(dec.funct3) || !io.dma_can_enqueue)

  // ── MREG reservations and physical ports ──
  // Each register is a resource: no read of a register with a pending
  // write (RAW), no write over a pending read (WAR) or write (WAW). VLOAD
  // follows the same rules: it used to skip the WAR check on the grounds
  // that it writes each row after an earlier reader has read it, which a
  // VPU pair read breaks (it reads m_2k+1 only after all of m_2k).
  // Ports: m_i and m_{i+32} share one read port and one write port.
  private val nPhys = 32
  val physReadBusy  = io.mregReadBusy(nPhys - 1, 0)  | io.mregReadBusy(2 * nPhys - 1, nPhys)
  val physWriteBusy = io.mregWriteBusy(nPhys - 1, 0) | io.mregWriteBusy(2 * nPhys - 1, nPhys)
  private def phys(b: UInt): UInt = b(log2Ceil(nPhys) - 1, 0)

  val mxuReadsVs1 = dec.mxu_cmd === MXU_PUSH_WEIGHT || dec.mxu_cmd === MXU_PUSH_ACC_FP8 ||
                    dec.mxu_cmd === MXU_PUSH_ACC_BF16 ||
                    dec.mxu_cmd === MXU_MATMUL || dec.mxu_cmd === MXU_MATMUL_ACC
  val vpuSrc1 = vpuPrimaryReadBank(dec)
  // NVFP4: push/matmul read m[vs1] and its scales; acc pushes read 2 or 4
  // registers, pops write 2 or 4.
  val nvReadsVs1 = dec.mxu_cmd === MXU_NV_PUSH_WEIGHT || dec.mxu_cmd === MXU_NV_MATMUL ||
                   dec.mxu_cmd === MXU_NV_MATMUL_ACC ||
                   dec.mxu_cmd === MXU_NV_PUSH_ACC_BF16 || dec.mxu_cmd === MXU_NV_PUSH_ACC_FP32
  val nvReadsPair = dec.mxu_cmd === MXU_NV_PUSH_ACC_BF16 || dec.mxu_cmd === MXU_NV_PUSH_ACC_FP32
  val nvReadsQuad = dec.mxu_cmd === MXU_NV_PUSH_ACC_FP32
  val nvWritesPair = dec.mxu_cmd === MXU_NV_POP_BF16 || dec.mxu_cmd === MXU_NV_POP_FP32
  val nvWritesQuad = dec.mxu_cmd === MXU_NV_POP_FP32
  val mregReads: Seq[(Bool, UInt)] = Seq(
    (mxuReadsVs1,                                              dec.vs1),
    (dec.mxu_cmd === MXU_PUSH_ACC_BF16,                        dec.vs1 + 1.U),
    (nvReadsVs1,                                               dec.vs1),
    (nvReadsPair,                                              dec.vs1 + 1.U),
    (nvReadsQuad,                                              dec.vs1 + 2.U),
    (nvReadsQuad,                                              dec.vs1 + 3.U),
    (isVstore,                                                 dec.vd),
    (dec.xlu_cmd =/= XLU_NONE,                                 dec.vs1),
    (dec.vpu_cmd =/= VPU_NONE && !vpuIsVli(dec.vpu_cmd),       vpuSrc1),
    (vpuReadsPrimaryPair(dec.vpu_cmd),                         vpuSrc1 + 1.U),
    (vpuReadsSecondaryPair(dec.vpu_cmd),                       dec.vs2),
    (vpuReadsSecondaryPair(dec.vpu_cmd),                       dec.vs2 + 1.U))
  val mregWrites: Seq[(Bool, UInt)] = Seq(
    (dec.mxu_cmd === MXU_POP_FP8 || dec.mxu_cmd === MXU_POP_BF16, dec.vd),
    (dec.mxu_cmd === MXU_POP_BF16,                                dec.vd + 1.U),
    (nvWritesPair,                                                dec.vd),
    (nvWritesPair,                                                dec.vd + 1.U),
    (nvWritesQuad,                                                dec.vd + 2.U),
    (nvWritesQuad,                                                dec.vd + 3.U),
    (isVload,                                                     dec.vd),
    (dec.xlu_cmd =/= XLU_NONE,                                    dec.vd),
    (dec.vpu_cmd =/= VPU_NONE,                                    dec.vd),
    (vpuWritesPair(dec.vpu_cmd),                                  dec.vd + 1.U))
  val mregHazard =
    mregReads.map { case (v, b) => v && (io.mregWriteBusy(b) || physReadBusy(phys(b))) }.reduce(_ || _) ||
    mregWrites.map { case (v, b) => v && (physWriteBusy(phys(b)) || io.mregReadBusy(b)) }.reduce(_ || _)

  // ── VMEM banks ──
  // The LSU scalar and vector paths must never use one VMEM bank in the
  // same cycle (Vmem asserts). Scalar accesses reach VMEM the cycle after
  // issue; vloadBusy / vstoreBusy cover the VLOAD reads (T+1..T+32) and
  // VSTORE writes (T+3..T+34).
  private val vecWordOffBits = log2Ceil(spP.wordsPerLine)
  private def bankIn(p: Valid[UInt], b: UInt): Bool = p.valid && p.bits === b
  val vlBank = io.lsu_vload_vmem_bank
  val vsBank = io.lsu_vstore_vmem_bank
  val scalarVmemBank = spP.getBankIdx(alu_result(spP.byteAddrBits - 1, spP.lineOffBits))
  val vecVmemBank    = spP.getBankIdx(alu_result(spP.lineAddrBits + vecWordOffBits - 1, vecWordOffBits))
  val vmemHazard =
    ((dec.is_mem_load || dec.is_mem_store) &&
      (bankIn(vlBank, scalarVmemBank) || bankIn(vsBank, scalarVmemBank))) ||
    (isVload && bankIn(vsBank, vecVmemBank)) ||
    (isVstore && bankIn(vlBank, vecVmemBank))

  // DMA goes through bank grants, so it cannot collide with the LSU, but it
  // must not overtake an LSU transfer on the banks it touches: a DMA store
  // (VMEM read) waits for a VSTORE, a DMA load (VMEM write) for a VLOAD or
  // a VSTORE.
  val dmaLine0 = Mux(dec.dma_cmd === DMA_LD, rd_data, rs1_data)(spP.wordAddrBits - 1, spP.wordOffBits)
  val dmaLineN = dmaLine0 +& (rs2_data >> spP.lineOffBits) - 1.U
  val dmaBank0 = spP.getBankIdx(dmaLine0)
  val dmaBankN = spP.getBankIdx(dmaLineN)
  private def dmaTouches(p: Valid[UInt]): Bool = p.valid && dmaBank0 <= p.bits && p.bits <= dmaBankN
  val dmaVmemHazard = (dec.dma_cmd === DMA_ST && dmaTouches(vsBank)) ||
                      (dec.dma_cmd === DMA_LD && (dmaTouches(vlBank) || dmaTouches(vsBank)))

  // ── Scalar loads ──
  // A load writes rd at T+3 without bypass, and the response shares the one
  // register-file write port. Tensor instructions carry no scalar operands.
  val xLoadPending = memLoadPending && memLoadCmd =/= MEM_SELD
  val xLoadWb      = memLoadRespValid && memLoadRespCmd =/= MEM_SELD
  private def xPending(r: UInt): Bool = r =/= 0.U &&
    ((xLoadPending && memLoadRd === r) || (xLoadWb && memLoadRespRd === r))
  val tensorOp = dec.mxu_cmd =/= MXU_NONE || dec.vpu_cmd =/= VPU_NONE || dec.xlu_cmd =/= XLU_NONE
  val xRegHazard = (!tensorOp && (xPending(dec.rs1) || xPending(dec.rs2) || xPending(dec.rd))) ||
                   (memLoadRespValid && dec.rd_wen && dec.rd =/= 0.U)

  // seld writes a scale register at T+3; vpack/vunpack and vmatpop.fp8 read
  // one, and seli shares its write port.
  val seldPending = (memLoadPending && memLoadCmd === MEM_SELD) ||
                    (memLoadRespValid && memLoadRespCmd === MEM_SELD)
  val readsScale  = (dec.vpu_cmd =/= VPU_NONE && vpuIsPackUnpack(dec.vpu_cmd)) ||
                    dec.mxu_cmd === MXU_POP_FP8
  val scaleHazard = (seldPending && readsScale) || (dec.is_seli && (seldPending || memLoadRespValid))

  val interlock = s1_valid && (mxuNotReady || vpuNotReady || xluNotReady || lsuNotReady ||
    dmaNotReady || mregHazard || vmemHazard || dmaVmemHazard || xRegHazard || scaleHazard)

  // ==============================================================
  // Stall / fire
  // ==============================================================

  val stall   = delay_stall || dma_wait_stall || interlock
  val s1_fire = s1_valid && !stall && !halt_now

  // Performance events: one cause per stalled cycle, in this order.
  val unitStall      = mxuNotReady || vpuNotReady || xluNotReady || lsuNotReady || dmaNotReady
  val interlockStall = !halted && !delay_stall && !dma_wait_stall && interlock
  io.perf.running    := !halted
  io.perf.issue      := s1_fire
  io.perf.stallDelay := !halted && delay_stall
  io.perf.stallDma   := !halted && !delay_stall && dma_wait_stall
  io.perf.stallUnit  := interlockStall && unitStall
  io.perf.stallData  := interlockStall && !unitStall
  // The causes of the stallUnit and stallData cycles: every cause that holds counts.
  io.perf.stallUnitMxu0    := interlockStall && mxuNotReady && isNvCmd
  io.perf.stallUnitMxu1    := interlockStall && mxuNotReady && !isNvCmd
  io.perf.stallUnitVpu     := interlockStall && vpuNotReady
  io.perf.stallUnitXlu     := interlockStall && xluNotReady
  io.perf.stallUnitLsu     := interlockStall && lsuNotReady
  io.perf.stallUnitDma     := interlockStall && dmaNotReady
  io.perf.stallDataMreg    := interlockStall && mregHazard
  io.perf.stallDataVmem    := interlockStall && vmemHazard
  io.perf.stallDataDmaVmem := interlockStall && dmaVmemHazard
  io.perf.stallDataXreg    := interlockStall && xRegHazard
  io.perf.stallDataScale   := interlockStall && scaleHazard

  // ==============================================================
  // Software-scheduling assertions
  // ==============================================================

  assert(!(s1_fire && instrMregReadHazard(dec)),
    "ASSERT FAIL: MREG RAW hazard: instruction reads bank with pending write")
  assert(!(s1_fire && instrMregWriteHazard(dec)),
    "ASSERT FAIL: MREG WAR/WAW hazard: instruction writes bank with pending read/write")

  assert(!(s1_fire && dec.vpu_cmd =/= VPU_NONE && vpuIssueBusy(dec.vpu_cmd)),
    "ASSERT FAIL: VPU command issued while its issue-busy bit is set")

  val vpuPrimaryBank = vpuPrimaryReadBank(dec)
  assert(!(s1_fire && vpuReadsPrimaryPair(dec.vpu_cmd) && vpuPrimaryBank(0)),
    "ASSERT FAIL: VPU pair-read primary bank must be even")
  assert(!(s1_fire && vpuReadsSecondaryPair(dec.vpu_cmd) && dec.vs2(0)),
    "ASSERT FAIL: VPU pair-read secondary bank must be even")
  assert(!(s1_fire && vpuWritesPair(dec.vpu_cmd) && dec.vd(0)),
    "ASSERT FAIL: VPU pair-write destination bank must be even")

  val issueVload  = s1_fire && isVload
  val issueVstore = s1_fire && isVstore

  assert(!(issueVload && io.lsu_vload_busy),
    "ASSERT FAIL: VLOAD issued while VLOAD path is busy")
  assert(!(issueVstore && io.lsu_vstore_busy),
    "ASSERT FAIL: VSTORE issued while VSTORE path is busy")

  // Scalar load: must not be issued while a prior load is pending.
  assert(!(s1_fire && dec.is_mem_load && memLoadPending),
    "ASSERT FAIL: scalar load issued while prior load response pending")
  assert(!(s1_fire && dec.is_mem_load && io.lsu_scalar_busy),
    "ASSERT FAIL: scalar load issued while LSU scalar path is busy")

  // Scalar stores stay deterministic and do not consume scalar-load busy.
  // VMEM bank conflicts between scalar store and active VSTORE
  // are asserted in Vmem.

  // ==============================================================
  // Delay counter
  // ==============================================================

  when(reset.asBool || halt_now || hostStart) {
    delayCounter := 0.U
  }.elsewhen(s1_fire && dec.is_delay) {
    delayCounter := dec.imm(11, 0).asUInt
  }.elsewhen(delayCounter =/= 0.U) {
    delayCounter := delayCounter - 1.U
  }

  // ==============================================================
  // Instruction hold across frontend stalls
  // ==============================================================

  when(reset.asBool || halt_now || hostStart) {
    s1_hold_active := false.B
    s1_instr_hold  := 0.U
  }.elsewhen(stall && !s1_hold_active) {
    s1_instr_hold  := io.imemFetch.rdata
    s1_hold_active := true.B
  }.elsewhen(!stall) {
    s1_hold_active := false.B
  }

  // ==============================================================
  // Scalar memory path via LSU
  // ==============================================================

  val issueScalarLoad  = s1_fire && dec.is_mem_load
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

  // Register scalar LSU commands so the scalar RF/ALU cone ends locally.
  when(issueScalarStore) {
    scalarMemCmdValid           := true.B
    scalarMemCmdReg.isStore     := true.B
    scalarMemCmdReg.byteAddr    := alu_result(spP.byteAddrBits - 1, 0)
    scalarMemCmdReg.wdata       := storeData
    scalarMemCmdReg.wmask       := storeMask
  }.elsewhen(issueScalarLoad) {
    scalarMemCmdValid           := true.B
    scalarMemCmdReg.isStore     := false.B
    scalarMemCmdReg.byteAddr    := alu_result(spP.byteAddrBits - 1, 0)
    scalarMemCmdReg.wdata       := 0.U
    scalarMemCmdReg.wmask       := 0.U
  }.otherwise {
    scalarMemCmdValid := false.B
  }

  io.scalarMemCmd.valid := scalarMemCmdValid
  io.scalarMemCmd.bits  := scalarMemCmdReg

  when(issueScalarLoad) {
    memLoadPending := true.B
    memLoadCmd     := dec.mem_cmd
    memLoadRd      := dec.rd
    memLoadByteOff := alu_result(1, 0)
  }

  val memRespCapture = memLoadPending && io.scalarMemResp.valid
  when(memRespCapture) {
    memLoadPending       := false.B
    memLoadRespValid     := true.B
    memLoadRespWord      := io.scalarMemResp.bits
    memLoadRespCmd       := memLoadCmd
    memLoadRespRd        := memLoadRd
    memLoadRespByteOff   := memLoadByteOff
  }.elsewhen(memLoadRespValid) {
    memLoadRespValid := false.B
  }

  val memRespArrived = memLoadRespValid

  val memWord = memLoadRespWord
  val memLoadResult = Wire(UInt(32.W))
  memLoadResult := memWord
  switch(memLoadRespCmd) {
    is(MEM_LB) {
      val b = (memWord >> (memLoadRespByteOff ## 0.U(3.W)))(7, 0)
      memLoadResult := Cat(Fill(24, b(7)), b)
    }
    is(MEM_LBU) {
      val b = (memWord >> (memLoadRespByteOff ## 0.U(3.W)))(7, 0)
      memLoadResult := Cat(0.U(24.W), b)
    }
    is(MEM_LH) {
      val h = (memWord >> (memLoadRespByteOff(1) ## 0.U(4.W)))(15, 0)
      memLoadResult := Cat(Fill(16, h(15)), h)
    }
    is(MEM_LHU) {
      val h = (memWord >> (memLoadRespByteOff(1) ## 0.U(4.W)))(15, 0)
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
  // csrrs / csrrc with rs1 = x0 and csrrsi / csrrci with uimm = 0 only read (RISC-V).
  io.csrPort.write := dec.csr_cmd === CSR_RW || dec.csr_cmd === CSR_RWI || dec.rs1 =/= 0.U
  io.csrPort.wdata := Mux(
    dec.csr_cmd === CSR_RWI || dec.csr_cmd === CSR_RSI || dec.csr_cmd === CSR_RCI,
    dec.imm.asUInt, rs1_data)
  io.csrPort.halted      := halt_now
  io.csrPort.set_illegal := illegal_detected && !halted
  io.csrPort.set_ecall   := s1_valid && dec.is_ecall && !halted
  io.csrPort.set_ebreak  := s1_valid && dec.is_ebreak && !halted
  io.csrPort.illegal_pc  := s1_pc
  io.csrPort.inst_retire := s1_fire

  val s1WritesScalarRd = s1_fire && dec.rd_wen && dec.rd =/= 0.U

  assert(!(memRespArrived && s1WritesScalarRd),
    "ASSERT FAIL: scalar load response collides with scalar register writeback")
  assert(!(memRespArrived && s1_fire && dec.is_seli),
    "ASSERT FAIL: SELD response collides with SELI scale-register write")

  val seldWriteEn = memRespArrived && memLoadRespCmd === MEM_SELD
  scaleRF.io.writeEn   := (s1_fire && dec.is_seli) || seldWriteEn
  scaleRF.io.writeIdx  := Mux(seldWriteEn, memLoadRespRd, dec.rd)
  scaleRF.io.writeData := Mux(seldWriteEn, memLoadResult(7, 0), dec.imm(7, 0))

  val isMemLoadWB = memRespArrived && memLoadRespCmd =/= MEM_SELD
  val wb_data = Mux(isMemLoadWB, memLoadResult,
    Mux(dec.is_csr, io.csrPort.rdata,
    Mux(dec.is_jal || dec.is_jalr, link_value, alu_result)))

  regfile.io.wr_en   := s1WritesScalarRd || isMemLoadWB
  regfile.io.wr_addr := Mux(isMemLoadWB, memLoadRespRd, dec.rd)
  regfile.io.wr_data := wb_data

  // ==============================================================
  // PC control
  // ==============================================================

  pc_ctrl.io.redirect        := redirect && !stall && !halt_now
  pc_ctrl.io.redirect_target := redirect_target
  pc_ctrl.io.stall           := stall
  pc_ctrl.io.halted          := halt_now
  pc_ctrl.io.restart         := hostStart

  // ==============================================================
  // Engine launch flags
  // ==============================================================

  val is_dma_launch  = s1_fire && (dec.dma_cmd === DMA_LD || dec.dma_cmd === DMA_ST)
  val is_dma_config  = s1_fire && (dec.dma_cmd === DMA_CONFIG)
  val is_mxu1_launch = s1_fire && (dec.mxu_cmd =/= MXU_NONE) && (dec.mxu_sel === MXUSEL_1)
  val is_nv_launch   = s1_fire && isNvCmd
  val is_vpu_launch  = s1_fire && (dec.vpu_cmd =/= VPU_NONE)
  val is_xlu_launch  = s1_fire && (dec.xlu_cmd =/= XLU_NONE)
  val is_lsu_launch  = s1_fire && dec.is_lsu

  when(is_dma_config) { dmaBaseReg := rs1_data }

  // ==============================================================
  // Engine command generation
  // ==============================================================

  val isDmaLoad = dec.dma_cmd === DMA_LD
  io.dmaCmd.valid         := is_dma_launch
  io.dmaCmd.bits.op       := dec.dma_cmd
  io.dmaCmd.bits.vmemAddr := Mux(isDmaLoad, rd_data, rs1_data)
  io.dmaCmd.bits.addr     := Mux(isDmaLoad, Cat(dmaBaseReg, rs1_data), Cat(dmaBaseReg, rd_data))
  io.dmaCmd.bits.size     := rs2_data
  io.dmaCmd.bits.channel  := dec.funct3

  io.mxu1Cmd.valid := is_mxu1_launch
  io.mxu1Cmd.bits  := buildMxuCmd(dec)
  io.nvCmd.valid   := is_nv_launch
  io.nvCmd.bits    := buildMxuCmd(dec)

  io.vpuCmd.valid           := is_vpu_launch
  io.vpuCmd.bits.op         := dec.vpu_cmd
  io.vpuCmd.bits.scaleE8M0  := scaleRF.io.regs(dec.vs1(4, 0))
  io.vpuCmd.bits.vd         := dec.vd
  io.vpuCmd.bits.vs1        := dec.vs1
  io.vpuCmd.bits.vs2        := dec.vs2
  io.vpuCmd.bits.imm        := dec.vi_imm

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

  when(hostStart) {
    halted          := false.B
    dmaBaseReg      := 0.U
    s1_hold_active  := false.B
    s1_instr_hold   := 0.U
    memLoadPending  := false.B
    memLoadRespValid := false.B
    scalarMemCmdValid := false.B
  }
}

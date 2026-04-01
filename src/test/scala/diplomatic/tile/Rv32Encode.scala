package atlas.tile

object RV32Encode {
  private def rType(f7: Int, rs2: Int, rs1: Int, f3: Int, rd: Int, op: Int): Int =
    ((f7 & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) |
    ((f3 & 0x7) << 12)  | ((rd & 0x1F) << 7)   | (op & 0x7F)
  private def iType(imm: Int, rs1: Int, f3: Int, rd: Int, op: Int): Int =
    ((imm & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((f3 & 0x7) << 12) | ((rd & 0x1F) << 7) | (op & 0x7F)
  private def sType(imm: Int, rs2: Int, rs1: Int, f3: Int, op: Int): Int = {
    val i = imm & 0xFFF
    (((i >> 5) & 0x7F) << 25) | ((rs2 & 0x1F) << 20) | ((rs1 & 0x1F) << 15) |
    ((f3 & 0x7) << 12) | ((i & 0x1F) << 7) | (op & 0x7F)
  }
  private def bType(imm: Int, rs2: Int, rs1: Int, f3: Int, op: Int): Int = {
    val i = imm & 0x1FFF
    (((i >> 12) & 1) << 31) | (((i >> 5) & 0x3F) << 25) | ((rs2 & 0x1F) << 20) |
    ((rs1 & 0x1F) << 15) | ((f3 & 0x7) << 12) | (((i >> 1) & 0xF) << 8) |
    (((i >> 11) & 1) << 7) | (op & 0x7F)
  }
  private def uType(imm: Int, rd: Int, op: Int): Int =
    (imm << 12) | ((rd & 0x1F) << 7) | (op & 0x7F)
  private def jType(imm: Int, rd: Int, op: Int): Int = {
    val i = imm & 0x1FFFFF
    (((i >> 20) & 1) << 31) | (((i >> 1) & 0x3FF) << 21) | (((i >> 11) & 1) << 20) |
    (((i >> 12) & 0xFF) << 12) | ((rd & 0x1F) << 7) | (op & 0x7F)
  }
  private def vrType(f7: Int, vs2: Int, vs1: Int, vd: Int, op: Int): Int =
    ((f7 & 0x7F) << 25) | ((vs2 & 0x3F) << 19) | ((vs1 & 0x3F) << 13) |
    ((vd & 0x3F) << 7) | (op & 0x7F)
  private def vlsType(imm12: Int, rs1: Int, f2: Int, vd: Int, op: Int): Int =
    ((imm12 & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) |
    ((f2 & 0x3) << 13) | ((vd & 0x3F) << 7) | (op & 0x7F)

  // ── RV32I ─────────────────────────────────────────────────────
  def ADD(rd: Int, rs1: Int, rs2: Int): Int  = rType(0x00, rs2, rs1, 0, rd, 0x33)
  def SUB(rd: Int, rs1: Int, rs2: Int): Int  = rType(0x20, rs2, rs1, 0, rd, 0x33)
  def AND(rd: Int, rs1: Int, rs2: Int): Int  = rType(0x00, rs2, rs1, 7, rd, 0x33)
  def OR(rd: Int, rs1: Int, rs2: Int): Int   = rType(0x00, rs2, rs1, 6, rd, 0x33)
  def XOR(rd: Int, rs1: Int, rs2: Int): Int  = rType(0x00, rs2, rs1, 4, rd, 0x33)
  def SLL(rd: Int, rs1: Int, rs2: Int): Int  = rType(0x00, rs2, rs1, 1, rd, 0x33)
  def SRL(rd: Int, rs1: Int, rs2: Int): Int  = rType(0x00, rs2, rs1, 5, rd, 0x33)
  def SRA(rd: Int, rs1: Int, rs2: Int): Int  = rType(0x20, rs2, rs1, 5, rd, 0x33)
  def SLT(rd: Int, rs1: Int, rs2: Int): Int  = rType(0x00, rs2, rs1, 2, rd, 0x33)
  def SLTU(rd: Int, rs1: Int, rs2: Int): Int = rType(0x00, rs2, rs1, 3, rd, 0x33)
  def ADDI(rd: Int, rs1: Int, imm: Int): Int = iType(imm, rs1, 0, rd, 0x13)
  def ANDI(rd: Int, rs1: Int, imm: Int): Int = iType(imm, rs1, 7, rd, 0x13)
  def ORI(rd: Int, rs1: Int, imm: Int): Int  = iType(imm, rs1, 6, rd, 0x13)
  def XORI(rd: Int, rs1: Int, imm: Int): Int = iType(imm, rs1, 4, rd, 0x13)
  def SLTI(rd: Int, rs1: Int, imm: Int): Int = iType(imm, rs1, 2, rd, 0x13)
  def SLTIU(rd: Int, rs1: Int, imm: Int): Int= iType(imm, rs1, 3, rd, 0x13)
  def SLLI(rd: Int, rs1: Int, sh: Int): Int  = iType(sh & 0x1F, rs1, 1, rd, 0x13)
  def SRLI(rd: Int, rs1: Int, sh: Int): Int  = iType(sh & 0x1F, rs1, 5, rd, 0x13)
  def SRAI(rd: Int, rs1: Int, sh: Int): Int  = iType((1 << 10) | (sh & 0x1F), rs1, 5, rd, 0x13)
  def LUI(rd: Int, imm20: Int): Int   = uType(imm20, rd, 0x37)
  def AUIPC(rd: Int, imm20: Int): Int = uType(imm20, rd, 0x17)
  def BEQ(rs1: Int, rs2: Int, off: Int): Int  = bType(off * 2, rs2, rs1, 0, 0x63)
  def BNE(rs1: Int, rs2: Int, off: Int): Int  = bType(off * 2, rs2, rs1, 1, 0x63)
  def BLT(rs1: Int, rs2: Int, off: Int): Int  = bType(off * 2, rs2, rs1, 4, 0x63)
  def BGE(rs1: Int, rs2: Int, off: Int): Int  = bType(off * 2, rs2, rs1, 5, 0x63)
  def BLTU(rs1: Int, rs2: Int, off: Int): Int = bType(off * 2, rs2, rs1, 6, 0x63)
  def BGEU(rs1: Int, rs2: Int, off: Int): Int = bType(off * 2, rs2, rs1, 7, 0x63)
  def JAL(rd: Int, off: Int): Int             = jType(off * 2, rd, 0x6F)
  def JALR(rd: Int, rs1: Int, off: Int): Int  = iType(off, rs1, 0, rd, 0x67)
  def CSRRW(rd: Int, csr: Int, rs1: Int): Int = iType(csr, rs1, 1, rd, 0x73)
  def CSRRS(rd: Int, csr: Int, rs1: Int): Int = iType(csr, rs1, 2, rd, 0x73)
  def CSRRC(rd: Int, csr: Int, rs1: Int): Int = iType(csr, rs1, 3, rd, 0x73)
  def CSRRWI(rd: Int, csr: Int, z: Int): Int  = iType(csr, z & 0x1F, 5, rd, 0x73)
  def CSRRSI(rd: Int, csr: Int, z: Int): Int  = iType(csr, z & 0x1F, 6, rd, 0x73)
  def CSRRCI(rd: Int, csr: Int, z: Int): Int  = iType(csr, z & 0x1F, 7, rd, 0x73)
  def NOP: Int = ADDI(0, 0, 0)
  def FENCE: Int = iType(0, 0, 0, 0, 0x0F)

  // ── Scalar loads/stores ───────────────────────────────────────
  def LB(rd: Int, rs1: Int, imm: Int): Int   = iType(imm, rs1, 0, rd, 0x03)
  def LH(rd: Int, rs1: Int, imm: Int): Int   = iType(imm, rs1, 1, rd, 0x03)
  def LW(rd: Int, rs1: Int, imm: Int): Int   = iType(imm, rs1, 2, rd, 0x03)
  def LBU(rd: Int, rs1: Int, imm: Int): Int  = iType(imm, rs1, 4, rd, 0x03)
  def LHU(rd: Int, rs1: Int, imm: Int): Int  = iType(imm, rs1, 5, rd, 0x03)
  def SELD(rd: Int, rs1: Int, imm: Int): Int = iType(imm, rs1, 6, rd, 0x03)
  def SELI(rd: Int, imm: Int): Int           = iType(imm, 0, 7, rd, 0x03)
  def SB(rs2: Int, rs1: Int, imm: Int): Int  = sType(imm, rs2, rs1, 0, 0x23)
  def SH(rs2: Int, rs1: Int, imm: Int): Int  = sType(imm, rs2, rs1, 1, 0x23)
  def SW(rs2: Int, rs1: Int, imm: Int): Int  = sType(imm, rs2, rs1, 2, 0x23)

  // ── Tensor load/store (VLS format, opcode 0x07) ───────────────
  def VLOAD(vd: Int, rs1: Int, imm12: Int): Int  = vlsType(imm12, rs1, 0, vd, 0x07)
  def VSTORE(vs: Int, rs1: Int, imm12: Int): Int = vlsType(imm12, rs1, 1, vs, 0x07)

  // ── DMA R-type (opcode 0x7B) ──────────────────────────────────
  // LOAD:  rd=vmem_reg, rs1=dram_off_reg, rs2=size_reg, funct3=ch
  // STORE: rd=dram_off_reg, rs1=vmem_reg, rs2=size_reg, funct3=ch
  def DMA_LOAD(rd: Int, rs1: Int, rs2: Int, ch: Int): Int =
    rType(0x00, rs2, rs1, ch, rd, 0x7B)
  def DMA_STORE(rd: Int, rs1: Int, rs2: Int, ch: Int): Int =
    rType(0x01, rs2, rs1, ch, rd, 0x7B)

  // ── DMA I-type (opcode 0x7F) ──────────────────────────────────
  def DMA_CONFIG(rs1: Int, ch: Int): Int = iType(0x000, rs1, ch, 0, 0x7F)
  def DMA_WAIT(ch: Int): Int             = iType(0x020, 0, ch, 0, 0x7F)

  // ── MXU (opcode 0x77), VR format ──────────────────────────────
  def VMATPUSH_W_MXU0(wSlot: Int, vs: Int): Int       = vrType(0x00, 0, vs, wSlot, 0x77)
  def VMATPUSH_W_MXU1(wSlot: Int, vs: Int): Int       = vrType(0x01, 0, vs, wSlot, 0x77)
  def VMATPUSH_AFP8_MXU0(accSel: Int, vs: Int): Int   = vrType(0x02, 0, vs, accSel, 0x77)
  def VMATPUSH_AFP8_MXU1(accSel: Int, vs: Int): Int   = vrType(0x03, 0, vs, accSel, 0x77)
  def VMATPUSH_ABF16_MXU0(accSel: Int, vs: Int): Int  = vrType(0x04, 0, vs, accSel, 0x77)
  def VMATPUSH_ABF16_MXU1(accSel: Int, vs: Int): Int  = vrType(0x05, 0, vs, accSel, 0x77)
  def VMATPOP_FP8_MXU0(vd: Int, scaleReg: Int, accSel: Int): Int = vrType(0x06, accSel, scaleReg, vd, 0x77)
  def VMATPOP_FP8_MXU1(vd: Int, scaleReg: Int, accSel: Int): Int = vrType(0x07, accSel, scaleReg, vd, 0x77)
  def VMATPOP_BF16_MXU0(vd: Int, accSel: Int): Int    = vrType(0x08, accSel, 0, vd, 0x77)
  def VMATPOP_BF16_MXU1(vd: Int, accSel: Int): Int    = vrType(0x09, accSel, 0, vd, 0x77)
  def VMATMUL_MXU0(accSel: Int, vs1: Int, wSlot: Int): Int     = vrType(0x0A, wSlot, vs1, accSel, 0x77)
  def VMATMUL_MXU1(accSel: Int, vs1: Int, wSlot: Int): Int     = vrType(0x0B, wSlot, vs1, accSel, 0x77)
  def VMATMUL_ACC_MXU0(accSel: Int, vs1: Int, wSlot: Int): Int = vrType(0x0C, wSlot, vs1, accSel, 0x77)
  def VMATMUL_ACC_MXU1(accSel: Int, vs1: Int, wSlot: Int): Int = vrType(0x0D, wSlot, vs1, accSel, 0x77)

  // ── VPU (opcode 0x57), XLU (opcode 0x6B) ──────────────────────
  def VADD_BF16(vd: Int, vs1: Int, vs2: Int): Int = vrType(0x00, vs2, vs1, vd, 0x57)
  def VREDSUM_BF16(vd: Int, vs1: Int): Int        = vrType(0x01, 0, vs1, vd, 0x57)
  def VSUB_BF16(vd: Int, vs1: Int, vs2: Int): Int = vrType(0x02, vs2, vs1, vd, 0x57)
  def VMIN_BF16(vd: Int, vs1: Int, vs2: Int): Int = vrType(0x04, vs2, vs1, vd, 0x57)
  def VMAX_BF16(vd: Int, vs1: Int, vs2: Int): Int = vrType(0x06, vs2, vs1, vd, 0x57)
  def VMUL_BF16(vd: Int, vs1: Int, vs2: Int): Int = vrType(0x24, vs2, vs1, vd, 0x57)
  def VMOV(vd: Int, vs: Int): Int                  = vrType(0x40, 0, vs, vd, 0x57)
  def VRECIP_BF16(vd: Int, vs: Int): Int           = vrType(0x41, 0, vs, vd, 0x57)
  def VEXP(vd: Int, vs: Int): Int                  = vrType(0x42, 0, vs, vd, 0x57)
  def VRELU(vd: Int, vs: Int): Int                 = vrType(0x44, 0, vs, vd, 0x57)
  def VTRPOSE_XLU(vd: Int, vs1: Int): Int          = vrType(0x00, 0, vs1, vd, 0x6B)
  def VREDMAX_XLU(vd: Int, vs1: Int): Int          = vrType(0x01, 0, vs1, vd, 0x6B)
  def VREDSUM_XLU(vd: Int, vs1: Int): Int          = vrType(0x02, 0, vs1, vd, 0x6B)
}

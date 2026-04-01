package atlas.scalar

import chisel3._
import chisel3.util._

object ScalarISA {

  val Y = 1.U(1.W)
  val N = 0.U(1.W)

  // ── ALU (4-bit) ────────────────────────────────────────────────────
  val ALU_ADD    =  0.U(4.W)
  val ALU_SUB    =  1.U(4.W)
  val ALU_AND    =  2.U(4.W)
  val ALU_OR     =  3.U(4.W)
  val ALU_XOR    =  4.U(4.W)
  val ALU_SLL    =  5.U(4.W)
  val ALU_SRL    =  6.U(4.W)
  val ALU_SRA    =  7.U(4.W)
  val ALU_SLT    =  8.U(4.W)
  val ALU_SLTU   =  9.U(4.W)
  val ALU_PASS_B = 10.U(4.W)
  val ALU_NOP    = 15.U(4.W)
  val ALU_X      =  0.U(4.W)

  // ── Branch (3-bit) ─────────────────────────────────────────────────
  val BR_NONE = 0.U(3.W);  val BR_EQ  = 1.U(3.W);  val BR_NE  = 2.U(3.W)
  val BR_LT   = 3.U(3.W);  val BR_GE  = 4.U(3.W);  val BR_LTU = 5.U(3.W)
  val BR_GEU  = 6.U(3.W);  val BR_X   = 0.U(3.W)

  // ── Op1/Op2 select (2-bit) ─────────────────────────────────────────
  val OP1_RS1  = 0.U(2.W); val OP1_PC   = 1.U(2.W)
  val OP1_ZERO = 2.U(2.W); val OP1_X    = 0.U(2.W)
  val OP2_RS2  = 0.U(2.W); val OP2_IMM  = 1.U(2.W)
  val OP2_ONE  = 2.U(2.W); val OP2_ZERO = 3.U(2.W); val OP2_X = 0.U(2.W)

  // ── Immediate type (3-bit) ─────────────────────────────────────────
  val IMM_I = 0.U(3.W); val IMM_S   = 1.U(3.W); val IMM_B   = 2.U(3.W)
  val IMM_U = 3.U(3.W); val IMM_J   = 4.U(3.W); val IMM_Z   = 5.U(3.W)
  val IMM_VLS = 6.U(3.W); val IMM_X = 0.U(3.W)

  // ── CSR op (3-bit) ─────────────────────────────────────────────────
  val CSR_NONE = 0.U(3.W); val CSR_RW  = 1.U(3.W); val CSR_RS  = 2.U(3.W)
  val CSR_RC   = 3.U(3.W); val CSR_RWI = 4.U(3.W); val CSR_RSI = 5.U(3.W)
  val CSR_RCI  = 6.U(3.W); val CSR_X   = 0.U(3.W)

  // ── DMA command (3-bit) ────────────────────────────────────────────
  val DMA_NONE   = 0.U(3.W)
  val DMA_LD     = 1.U(3.W)
  val DMA_ST     = 2.U(3.W)
  val DMA_CONFIG = 3.U(3.W)
  val DMA_WAIT   = 4.U(3.W)
  val DMA_X      = 0.U(3.W)

  // ── Memory command (4-bit) — scalar loads/stores to VMEM ───────────
  val MEM_NONE = 0.U(4.W); val MEM_LB  = 1.U(4.W); val MEM_LH   = 2.U(4.W)
  val MEM_LW   = 3.U(4.W); val MEM_LBU = 4.U(4.W); val MEM_LHU  = 5.U(4.W)
  val MEM_SB   = 6.U(4.W); val MEM_SH  = 7.U(4.W); val MEM_SW   = 8.U(4.W)
  val MEM_SELD = 9.U(4.W); val MEM_X   = 0.U(4.W)

  // ── LSU command (2-bit) ────────────────────────────────────────────
  val LSU_NONE   = 0.U(2.W); val LSU_VLOAD  = 1.U(2.W)
  val LSU_VSTORE = 2.U(2.W); val LSU_X      = 0.U(2.W)

  // ── MXU command (4-bit) ────────────────────────────────────────────
  val MXU_NONE         = 0.U(4.W); val MXU_PUSH_WEIGHT  = 1.U(4.W)
  val MXU_PUSH_ACC_FP8 = 2.U(4.W); val MXU_PUSH_ACC_BF16= 3.U(4.W)
  val MXU_POP_FP8      = 4.U(4.W); val MXU_POP_BF16     = 5.U(4.W)
  val MXU_MATMUL       = 6.U(4.W); val MXU_MATMUL_ACC   = 7.U(4.W)
  val MXU_X            = 0.U(4.W)

  // ── MXU select (1-bit) ─────────────────────────────────────────────
  val MXUSEL_0 = 0.U(1.W); val MXUSEL_1 = 1.U(1.W); val MXUSEL_X = 0.U(1.W)

  // ── VPU command (5-bit) ────────────────────────────────────────────
  val VPU_NONE =  0.U(5.W); val VPU_ADD  =  1.U(5.W); val VPU_RSUM =  2.U(5.W)
  val VPU_SUB  =  3.U(5.W); val VPU_MIN  =  4.U(5.W); val VPU_MAX  =  5.U(5.W)
  val VPU_MUL  =  6.U(5.W); val VPU_MOV  =  7.U(5.W); val VPU_RCP  =  8.U(5.W)
  val VPU_EXP  =  9.U(5.W); val VPU_RELU = 10.U(5.W)
  val VPU_LI_ALL = 11.U(5.W); val VPU_LI_ROW = 12.U(5.W)
  val VPU_LI_COL = 13.U(5.W); val VPU_LI_ONE = 14.U(5.W); val VPU_X = 0.U(5.W)

  // ── XLU command (2-bit) ────────────────────────────────────────────
  val XLU_NONE = 0.U(2.W); val XLU_TRPOSE = 1.U(2.W)
  val XLU_RMAX = 2.U(2.W); val XLU_RSUM   = 3.U(2.W); val XLU_X = 0.U(2.W)

  // ── CSR address map ────────────────────────────────────────────────
  val CSR_CYCLE_COUNTER = 0xC00.U(12.W); val CSR_INST_COUNTER = 0xC01.U(12.W)
  val CSR_STATUS        = 0xC02.U(12.W); val CSR_ILLEGAL_INSTR= 0xC03.U(12.W)
  val CSR_DBG0          = 0xC10.U(12.W); val CSR_DBG1         = 0xC11.U(12.W)

  val NUM_SCALE_REGS = 32
  val CSR_SCALE_BASE = 0x860.U(12.W)
  def isScaleAddr(addr: UInt): Bool = addr >= 0x860.U && addr <= 0x87F.U
  def scaleIndex(addr: UInt): UInt  = (addr - CSR_SCALE_BASE)(4, 0)
}

/*
IDecode.scala — Instruction decode table (17 columns).
   0  legal(1)   1  alu_fn(4)    2  br_type(3)   3  op1_sel(2)
   4  op2_sel(2) 5  imm_sel(3)   6  rd_wen(1)    7  jal(1)
   8  jalr(1)    9  csr_cmd(3)  10  dma_cmd(3)  11  mxu_cmd(4)
  12  mxu_sel(1) 13  vpu_cmd(5) 14  xlu_cmd(2)  15  mem_cmd(4)
  16  lsu_cmd(2)
*/

package atlas.scalar

import chisel3._
import chisel3.util._

object IDecode {
  import ScalarISA._
  import Instructions._

  val default: List[UInt] = List(
    N, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X)

  val table: Array[(BitPat, List[UInt])] = Array(
    // ── R-type ALU ────────────────────────────────────────────────────
    ADD  -> List(Y, ALU_ADD,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SUB  -> List(Y, ALU_SUB,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SLL  -> List(Y, ALU_SLL,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SLT  -> List(Y, ALU_SLT,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SLTU -> List(Y, ALU_SLTU, BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    XOR  -> List(Y, ALU_XOR,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SRL  -> List(Y, ALU_SRL,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SRA  -> List(Y, ALU_SRA,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    OR   -> List(Y, ALU_OR,   BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    AND  -> List(Y, ALU_AND,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),

    // ── I-type ALU ────────────────────────────────────────────────────
    ADDI  -> List(Y, ALU_ADD,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SLTI  -> List(Y, ALU_SLT,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SLTIU -> List(Y, ALU_SLTU, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    XORI  -> List(Y, ALU_XOR,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    ORI   -> List(Y, ALU_OR,   BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    ANDI  -> List(Y, ALU_AND,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SLLI  -> List(Y, ALU_SLL,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SRLI  -> List(Y, ALU_SRL,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    SRAI  -> List(Y, ALU_SRA,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),

    // ── Branches ──────────────────────────────────────────────────────
    BEQ  -> List(Y, ALU_X, BR_EQ,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    BNE  -> List(Y, ALU_X, BR_NE,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    BLT  -> List(Y, ALU_X, BR_LT,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    BGE  -> List(Y, ALU_X, BR_GE,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    BLTU -> List(Y, ALU_X, BR_LTU, OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    BGEU -> List(Y, ALU_X, BR_GEU, OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),

    JAL   -> List(Y, ALU_ADD, BR_X, OP1_PC,  OP2_IMM, IMM_J, Y, Y, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    JALR  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, Y, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    DELAY -> List(Y, ALU_X,  BR_X, OP1_X,   OP2_X,   IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    LUI   -> List(Y, ALU_PASS_B, BR_X, OP1_X,  OP2_IMM, IMM_U, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    AUIPC -> List(Y, ALU_ADD,    BR_X, OP1_PC, OP2_IMM, IMM_U, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),

    // ── CSR ──────────────────────────────────────────────────────────
    CSRRW  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RW,  DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    CSRRS  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RS,  DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    CSRRC  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RC,  DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    CSRRWI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RWI, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    CSRRSI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RSI, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    CSRRCI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RCI, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    ECALL  -> List(N, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    EBREAK -> List(N, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    FENCE  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),

    // ── Scalar loads/stores ─────────────────────────────────────────
    LB   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_LB,   LSU_X),
    LH   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_LH,   LSU_X),
    LW   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_LW,   LSU_X),
    LBU  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_LBU,  LSU_X),
    LHU  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_LHU,  LSU_X),
    SELD -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_I, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_SELD, LSU_X),
    SELI -> List(Y, ALU_X,  BR_X, OP1_X,   OP2_X,   IMM_I, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X,    LSU_X),
    SB   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_S, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_SB,   LSU_X),
    SH   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_S, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_SH,   LSU_X),
    SW   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_S, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_SW,   LSU_X),

    // ── Tensor load/store ───────────────────────────────────────────
    VLOAD  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_VLOAD),
    VSTORE -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_VSTORE),

    // ── DMA ─────────────────────────────────────────────────────────
    DMA_LOAD_ANY   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_LD,     MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    DMA_STORE_ANY  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_ST,     MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    DMA_CONFIG_ANY -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_CONFIG, MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),
    DMA_WAIT_ANY   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_WAIT,   MXU_X, MXUSEL_X, VPU_X, XLU_X, MEM_X, LSU_X),

    // ── MXU0/1, VPU, VLI, XLU — unchanged ──────────────────────────
    VMATPUSH_W_MXU0     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_PUSH_WEIGHT,   MXUSEL_0, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPUSH_AFP8_MXU0  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_PUSH_ACC_FP8,  MXUSEL_0, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPUSH_ABF16_MXU0 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_PUSH_ACC_BF16, MXUSEL_0, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPOP_FP8_MXU0    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_POP_FP8,       MXUSEL_0, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPOP_BF16_MXU0   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_POP_BF16,      MXUSEL_0, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATMUL_MXU0        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_MATMUL,        MXUSEL_0, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATMUL_ACC_MXU0    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_MATMUL_ACC,    MXUSEL_0, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPUSH_W_MXU1     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_PUSH_WEIGHT,   MXUSEL_1, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPUSH_AFP8_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_PUSH_ACC_FP8,  MXUSEL_1, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPUSH_ABF16_MXU1 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_PUSH_ACC_BF16, MXUSEL_1, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPOP_FP8_MXU1    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_POP_FP8,       MXUSEL_1, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATPOP_BF16_MXU1   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_POP_BF16,      MXUSEL_1, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATMUL_MXU1        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_MATMUL,        MXUSEL_1, VPU_X, XLU_X, MEM_X, LSU_X),
    VMATMUL_ACC_MXU1    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_MATMUL_ACC,    MXUSEL_1, VPU_X, XLU_X, MEM_X, LSU_X),

    VADD_BF16    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_ADD,    XLU_X, MEM_X, LSU_X),
    VREDSUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_RSUM,   XLU_X, MEM_X, LSU_X),
    VSUB_BF16    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_SUB,    XLU_X, MEM_X, LSU_X),
    VMIN_BF16    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_MIN,    XLU_X, MEM_X, LSU_X),
    VMAX_BF16    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_MAX,    XLU_X, MEM_X, LSU_X),
    VMUL_BF16    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_MUL,    XLU_X, MEM_X, LSU_X),
    VMOV         -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_MOV,    XLU_X, MEM_X, LSU_X),
    VRECIP_BF16  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_RCP,    XLU_X, MEM_X, LSU_X),
    VEXP         -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_EXP,    XLU_X, MEM_X, LSU_X),
    VRELU        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_RELU,   XLU_X, MEM_X, LSU_X),
    VLI_ALL      -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_LI_ALL, XLU_X, MEM_X, LSU_X),
    VLI_ROW      -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_LI_ROW, XLU_X, MEM_X, LSU_X),
    VLI_COL      -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_LI_COL, XLU_X, MEM_X, LSU_X),
    VLI_ONE      -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_LI_ONE, XLU_X, MEM_X, LSU_X),
    VTRPOSE_XLU  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_TRPOSE, MEM_X, LSU_X),
    VREDMAX_XLU  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_RMAX,   MEM_X, LSU_X),
    VREDSUM_XLU  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, MXU_X, MXUSEL_X, VPU_X, XLU_RSUM,   MEM_X, LSU_X)
  )
}

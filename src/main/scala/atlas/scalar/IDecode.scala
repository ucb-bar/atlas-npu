/*
IDecode.scala — Instruction decode table (17 columns).
   0  legal(1)   1  alu_fn(4)    2  br_type(3)   3  op1_sel(2)
   4  op2_sel(2) 5  imm_sel(3)   6  rd_wen(1)    7  jal(1)
   8  jalr(1)    9  csr_cmd(3)  10  dma_cmd(3)  11  mxu(1)
  12  vpu_cmd(5) 13  xlu_cmd(1)  14  mem_cmd(4)  15  lsu_cmd(3)
  16  sc_cmd(2)
An MXU instruction (mxu = Y) takes its command from funct7[3:0] and its slot from funct7[4].
*/

package atlas.scalar

import chisel3._
import chisel3.util._

object IDecode {
  import ScalarISA._
  import Instructions._

  val default: List[UInt] = List(
    N, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X)

  val table: Array[(BitPat, List[UInt])] = Array(
    // ── R-type ALU ────────────────────────────────────────────────────
    ADD  -> List(Y, ALU_ADD,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SUB  -> List(Y, ALU_SUB,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SLL  -> List(Y, ALU_SLL,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SLT  -> List(Y, ALU_SLT,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SLTU -> List(Y, ALU_SLTU, BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    XOR  -> List(Y, ALU_XOR,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SRL  -> List(Y, ALU_SRL,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SRA  -> List(Y, ALU_SRA,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    OR   -> List(Y, ALU_OR,   BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    AND  -> List(Y, ALU_AND,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),

    // ── I-type ALU ────────────────────────────────────────────────────
    ADDI  -> List(Y, ALU_ADD,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SLTI  -> List(Y, ALU_SLT,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SLTIU -> List(Y, ALU_SLTU, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    XORI  -> List(Y, ALU_XOR,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    ORI   -> List(Y, ALU_OR,   BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    ANDI  -> List(Y, ALU_AND,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SLLI  -> List(Y, ALU_SLL,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SRLI  -> List(Y, ALU_SRL,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    SRAI  -> List(Y, ALU_SRA,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),

    // ── Branches ──────────────────────────────────────────────────────
    BEQ  -> List(Y, ALU_X, BR_EQ,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    BNE  -> List(Y, ALU_X, BR_NE,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    BLT  -> List(Y, ALU_X, BR_LT,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    BGE  -> List(Y, ALU_X, BR_GE,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    BLTU -> List(Y, ALU_X, BR_LTU, OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    BGEU -> List(Y, ALU_X, BR_GEU, OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),

    JAL   -> List(Y, ALU_ADD,    BR_X, OP1_PC,  OP2_IMM, IMM_J, Y, Y, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    JALR  -> List(Y, ALU_ADD,    BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, Y, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    DELAY -> List(Y, ALU_X,      BR_X, OP1_X,   OP2_X,   IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_DELAY),
    LUI   -> List(Y, ALU_PASS_B, BR_X, OP1_X,   OP2_IMM, IMM_U, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    AUIPC -> List(Y, ALU_ADD,    BR_X, OP1_PC,  OP2_IMM, IMM_U, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),

    // ── CSR ──────────────────────────────────────────────────────────
    CSRRW  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RW,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    CSRRS  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RS,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    CSRRC  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RC,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    CSRRWI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RWI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    CSRRSI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RSI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    CSRRCI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RCI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    ECALL  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    EBREAK -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    FENCE  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),

    // ── Scalar loads/stores ─────────────────────────────────────────
    LB   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LB,   LSU_X, SC_X),
    LH   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LH,   LSU_X, SC_X),
    LW   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LW,   LSU_X, SC_X),
    LBU  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LBU,  LSU_X, SC_X),
    LHU  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LHU,  LSU_X, SC_X),
    SELD -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_ZERO, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SELD, LSU_X, SC_X),
    SELI -> List(Y, ALU_X,   BR_X, OP1_X,   OP2_X,    IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X,    LSU_X, SC_SELI),
    SB   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SB,   LSU_X, SC_X),
    SH   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SH,   LSU_X, SC_X),
    SW   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SW,   LSU_X, SC_X),

    // ── Tensor load/store ───────────────────────────────────────────
    VLOAD        -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VLOAD,        SC_X),
    VSTORE       -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VSTORE,       SC_X),
    VLOAD_SCALE  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VLOAD_SCALE,  SC_X),
    VSTORE_SCALE -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VSTORE_SCALE, SC_X),

    // ── DMA ─────────────────────────────────────────────────────────
    // DMA_CONFIG and DMA_WAIT name both a BitPat and a dma_cmd.
    DMA_LOAD                -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_LD,               N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    DMA_STORE               -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_ST,               N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    Instructions.DMA_CONFIG -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, ScalarISA.DMA_CONFIG, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    Instructions.DMA_WAIT   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, ScalarISA.DMA_WAIT,   N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),

    // ── MXU slot 1 (mxu1) and slot 0 (the NVFP4 MXU) ────────────────
    VMATPUSH_WEIGHT_MXU1   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPUSH_ACC_FP8_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPUSH_ACC_BF16_MXU1 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPOP_FP8_ACC_MXU1   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPOP_BF16_ACC_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATMUL_MXU1           -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATMUL_ACC_MXU1       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPUSH_WEIGHT_NV     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPUSH_ACC_BF16_NV   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPUSH_ACC_FP32_NV   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPOP_BF16_ACC_NV    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATPOP_FP32_ACC_NV    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATMUL_NV             -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),
    VMATMUL_ACC_NV         -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X),

    // ── VPU: element-wise binary ops ────────────────────────────────
    VADD_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_ADD,     XLU_X, MEM_X, LSU_X, SC_X),
    VSUB_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SUB,     XLU_X, MEM_X, LSU_X, SC_X),
    VMUL_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_MUL,     XLU_X, MEM_X, LSU_X, SC_X),
    VMINIMUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMIN, XLU_X, MEM_X, LSU_X, SC_X),
    VMAXIMUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMAX, XLU_X, MEM_X, LSU_X, SC_X),

    // ── VPU: sublane (column) reductions — result[i,j] = op(src[:,j]) ─
    VREDSUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CSUM, XLU_X, MEM_X, LSU_X, SC_X),
    VREDMIN_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CMIN, XLU_X, MEM_X, LSU_X, SC_X),
    VREDMAX_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CMAX, XLU_X, MEM_X, LSU_X, SC_X),

    // ── VPU: lane (row) reductions — result[i,j] = op(src[i,:]) ────
    VREDSUM_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RSUM, XLU_X, MEM_X, LSU_X, SC_X),
    VREDMIN_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RMIN, XLU_X, MEM_X, LSU_X, SC_X),
    VREDMAX_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RMAX, XLU_X, MEM_X, LSU_X, SC_X),

    // ── VPU: unary ops ──────────────────────────────────────────────
    VMOV             -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_MOV,       XLU_X, MEM_X, LSU_X, SC_X),
    VRECIP_BF16      -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RCP,       XLU_X, MEM_X, LSU_X, SC_X),
    VEXP_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_EXP,       XLU_X, MEM_X, LSU_X, SC_X),
    VEXP2_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_EXP2,      XLU_X, MEM_X, LSU_X, SC_X),
    VSQUARE_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SQUARE,    XLU_X, MEM_X, LSU_X, SC_X),
    VCUBE_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CUBE,      XLU_X, MEM_X, LSU_X, SC_X),
    VPACK_BF16_FP8   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_FP8PACK,   XLU_X, MEM_X, LSU_X, SC_X),
    VUNPACK_FP8_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_FP8UNPACK, XLU_X, MEM_X, LSU_X, SC_X),
    VRELU_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RELU,      XLU_X, MEM_X, LSU_X, SC_X),
    VSIN_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SIN,       XLU_X, MEM_X, LSU_X, SC_X),
    VCOS_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_COS,       XLU_X, MEM_X, LSU_X, SC_X),
    VTANH_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_TANH,      XLU_X, MEM_X, LSU_X, SC_X),
    VLOG2_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LOG,       XLU_X, MEM_X, LSU_X, SC_X),
    VSQRT_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SQRT,      XLU_X, MEM_X, LSU_X, SC_X),

    // ── VLI ─────────────────────────────────────────────────────────
    VLI_ALL -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ALL, XLU_X, MEM_X, LSU_X, SC_X),
    VLI_ROW -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ROW, XLU_X, MEM_X, LSU_X, SC_X),
    VLI_COL -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_COL, XLU_X, MEM_X, LSU_X, SC_X),
    VLI_ONE -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ONE, XLU_X, MEM_X, LSU_X, SC_X),

    // ── XLU ─────────────────────────────────────────────────────────
    VTRPOSE_XLU -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_TRPOSE, MEM_X, LSU_X, SC_X)
  )
}

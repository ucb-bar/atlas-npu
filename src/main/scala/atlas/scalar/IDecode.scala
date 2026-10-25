/*
IDecode.scala — Instruction decode table (18 columns).
   0  legal(1)   1  alu_fn(4)    2  br_type(3)   3  op1_sel(2)
   4  op2_sel(2) 5  imm_sel(3)   6  rd_wen(1)    7  jal(1)
   8  jalr(1)    9  csr_cmd(3)  10  dma_cmd(3)  11  mxu(1)
  12  vpu_cmd(5) 13  xlu_cmd(1)  14  mem_cmd(4)  15  lsu_cmd(3)
  16  sc_cmd(2)  17  xr(3)
An MXU instruction (mxu = Y) takes its command from funct7[3:0] and its slot from funct7[4].
xr names the fields that hold an x-register: D = rd (written, or read by dma.load and dma.store),
1 = rs1 and 2 = rs2 (read). The scalar-load interlock checks only these.
*/

package atlas.scalar

import chisel3._
import chisel3.util._

object IDecode {
  import ScalarISA._
  import Instructions._

  val default: List[UInt] = List(
    N, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N)

  val table: Array[(BitPat, List[UInt])] = Array(
    // ── R-type ALU ────────────────────────────────────────────────────
    ADD  -> List(Y, ALU_ADD,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    SUB  -> List(Y, ALU_SUB,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    SLL  -> List(Y, ALU_SLL,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    SLT  -> List(Y, ALU_SLT,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    SLTU -> List(Y, ALU_SLTU, BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    XOR  -> List(Y, ALU_XOR,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    SRL  -> List(Y, ALU_SRL,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    SRA  -> List(Y, ALU_SRA,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    OR   -> List(Y, ALU_OR,   BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    AND  -> List(Y, ALU_AND,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),

    // ── I-type ALU ────────────────────────────────────────────────────
    ADDI  -> List(Y, ALU_ADD,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    SLTI  -> List(Y, ALU_SLT,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    SLTIU -> List(Y, ALU_SLTU, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    XORI  -> List(Y, ALU_XOR,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    ORI   -> List(Y, ALU_OR,   BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    ANDI  -> List(Y, ALU_AND,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    SLLI  -> List(Y, ALU_SLL,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    SRLI  -> List(Y, ALU_SRL,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    SRAI  -> List(Y, ALU_SRA,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),

    // ── Branches ──────────────────────────────────────────────────────
    BEQ  -> List(Y, ALU_X, BR_EQ,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12),
    BNE  -> List(Y, ALU_X, BR_NE,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12),
    BLT  -> List(Y, ALU_X, BR_LT,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12),
    BGE  -> List(Y, ALU_X, BR_GE,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12),
    BLTU -> List(Y, ALU_X, BR_LTU, OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12),
    BGEU -> List(Y, ALU_X, BR_GEU, OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12),

    JAL   -> List(Y, ALU_ADD,    BR_X, OP1_PC,  OP2_IMM, IMM_J, Y, Y, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D),
    JALR  -> List(Y, ALU_ADD,    BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, Y, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    DELAY -> List(Y, ALU_X,      BR_X, OP1_X,   OP2_X,   IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_DELAY, XR_N),
    LUI   -> List(Y, ALU_PASS_B, BR_X, OP1_X,   OP2_IMM, IMM_U, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D),
    AUIPC -> List(Y, ALU_ADD,    BR_X, OP1_PC,  OP2_IMM, IMM_U, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D),

    // ── CSR ──────────────────────────────────────────────────────────
    CSRRW  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RW,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    CSRRS  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RS,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    CSRRC  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RC,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1),
    CSRRWI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RWI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D),
    CSRRSI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RSI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D),
    CSRRCI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RCI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D),
    ECALL  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    EBREAK -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    FENCE  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),

    // ── Scalar loads/stores ─────────────────────────────────────────
    LB   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LB,   LSU_X, SC_X, XR_D1),
    LH   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LH,   LSU_X, SC_X, XR_D1),
    LW   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LW,   LSU_X, SC_X, XR_D1),
    LBU  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LBU,  LSU_X, SC_X, XR_D1),
    LHU  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LHU,  LSU_X, SC_X, XR_D1),
    SELD -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_ZERO, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SELD, LSU_X, SC_X, XR_1),
    SELI -> List(Y, ALU_X,   BR_X, OP1_X,   OP2_X,    IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X,    LSU_X, SC_SELI, XR_N),
    SB   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SB,   LSU_X, SC_X, XR_12),
    SH   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SH,   LSU_X, SC_X, XR_12),
    SW   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SW,   LSU_X, SC_X, XR_12),

    // ── Tensor load/store ───────────────────────────────────────────
    VLOAD        -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VLOAD,        SC_X, XR_1),
    VSTORE       -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VSTORE,       SC_X, XR_1),
    VLOAD_SCALE  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VLOAD_SCALE,  SC_X, XR_1),
    VSTORE_SCALE -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VSTORE_SCALE, SC_X, XR_1),

    // ── DMA ─────────────────────────────────────────────────────────
    // DMA_CONFIG and DMA_WAIT name both a BitPat and a dma_cmd.
    DMA_LOAD                -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_LD,               N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    DMA_STORE               -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_ST,               N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12),
    Instructions.DMA_CONFIG -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, ScalarISA.DMA_CONFIG, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_1),
    Instructions.DMA_WAIT   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, ScalarISA.DMA_WAIT,   N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),

    // ── MXU slot 1 (mxu1) and slot 0 (the NVFP4 MXU) ────────────────
    VMATPUSH_WEIGHT_MXU1   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPUSH_ACC_FP8_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPUSH_ACC_BF16_MXU1 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPUSH_ACC_FP32_MXU1 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPOP_FP8_ACC_MXU1   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPOP_BF16_ACC_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPOP_BF16_SCALED_ACC_MXU1 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPOP_FP32_ACC_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATMUL_MXU1           -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATMUL_ACC_MXU1       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPUSH_WEIGHT_NV     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPUSH_ACC_BF16_NV   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPUSH_ACC_FP32_NV   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPOP_BF16_ACC_NV    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPOP_BF16_SCALED_ACC_NV -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATPOP_FP32_ACC_NV    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATMUL_NV             -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMATMUL_ACC_NV         -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N),

    // ── VPU: element-wise binary ops ────────────────────────────────
    VADD_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_ADD,     XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VSUB_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SUB,     XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMUL_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_MUL,     XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMINIMUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMIN, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VMAXIMUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMAX, XLU_X, MEM_X, LSU_X, SC_X, XR_N),

    // ── VPU: sublane (column) reductions — result[i,j] = op(src[:,j]) ─
    VREDSUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CSUM, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VREDMIN_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CMIN, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VREDMAX_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CMAX, XLU_X, MEM_X, LSU_X, SC_X, XR_N),

    // ── VPU: lane (row) reductions — result[i,j] = op(src[i,:]) ────
    VREDSUM_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RSUM, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VREDMIN_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RMIN, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VREDMAX_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RMAX, XLU_X, MEM_X, LSU_X, SC_X, XR_N),

    // ── VPU: unary ops ──────────────────────────────────────────────
    VMOV             -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_MOV,       XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VRECIP_BF16      -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RCP,       XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VEXP_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_EXP,       XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VEXP2_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_EXP2,      XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VSQUARE_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SQUARE,    XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VCUBE_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CUBE,      XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VPACK_BF16_FP8   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_FP8PACK,   XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VUNPACK_FP8_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_FP8UNPACK, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VRELU_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RELU,      XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VSIN_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SIN,       XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VCOS_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_COS,       XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VTANH_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_TANH,      XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VLOG2_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LOG,       XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VSQRT_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SQRT,      XLU_X, MEM_X, LSU_X, SC_X, XR_N),

    // ── VLI ─────────────────────────────────────────────────────────
    VLI_ALL -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ALL, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VLI_ROW -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ROW, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VLI_COL -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_COL, XLU_X, MEM_X, LSU_X, SC_X, XR_N),
    VLI_ONE -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ONE, XLU_X, MEM_X, LSU_X, SC_X, XR_N),

    // ── XLU ─────────────────────────────────────────────────────────
    VTRPOSE_XLU -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_TRPOSE, MEM_X, LSU_X, SC_X, XR_N)
  )
}

/*
IDecode.scala — Instruction decode table (19 columns).
   0  legal(1)   1  alu_fn(4)    2  br_type(3)   3  op1_sel(2)
   4  op2_sel(2) 5  imm_sel(3)   6  rd_wen(1)    7  jal(1)
   8  jalr(1)    9  csr_cmd(3)  10  dma_cmd(3)  11  mxu(1)
  12  vpu_cmd(5) 13  xlu_cmd(1)  14  mem_cmd(4)  15  lsu_cmd(3)
  16  sc_cmd(2)  17  xr(3)       18  vpu_src(2)
An MXU instruction (mxu = Y) takes its command from funct7[3:0] and its slot from funct7[4].
xr names the fields that hold an x-register: D = rd (written, or read by dma.load and dma.store),
1 = rs1 and 2 = rs2 (read). The scalar-load interlock checks only these.
A .vx or .vr instruction has the vpu_cmd of its .bf16 form; vpu_src gives its second operand.
*/

package atlas.scalar

import chisel3._
import chisel3.util._

object IDecode {
  import ScalarISA._
  import Instructions._

  val default: List[UInt] = List(
    N, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X)

  val table: Array[(BitPat, List[UInt])] = Array(
    // ── R-type ALU ────────────────────────────────────────────────────
    ADD  -> List(Y, ALU_ADD,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    SUB  -> List(Y, ALU_SUB,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    SLL  -> List(Y, ALU_SLL,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    SLT  -> List(Y, ALU_SLT,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    SLTU -> List(Y, ALU_SLTU, BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    XOR  -> List(Y, ALU_XOR,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    SRL  -> List(Y, ALU_SRL,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    SRA  -> List(Y, ALU_SRA,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    OR   -> List(Y, ALU_OR,   BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    AND  -> List(Y, ALU_AND,  BR_X, OP1_RS1, OP2_RS2, IMM_X, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),

    // ── I-type ALU ────────────────────────────────────────────────────
    ADDI  -> List(Y, ALU_ADD,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    SLTI  -> List(Y, ALU_SLT,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    SLTIU -> List(Y, ALU_SLTU, BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    XORI  -> List(Y, ALU_XOR,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    ORI   -> List(Y, ALU_OR,   BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    ANDI  -> List(Y, ALU_AND,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    SLLI  -> List(Y, ALU_SLL,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    SRLI  -> List(Y, ALU_SRL,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    SRAI  -> List(Y, ALU_SRA,  BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),

    // ── Branches ──────────────────────────────────────────────────────
    BEQ  -> List(Y, ALU_X, BR_EQ,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12, VSRC_X),
    BNE  -> List(Y, ALU_X, BR_NE,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12, VSRC_X),
    BLT  -> List(Y, ALU_X, BR_LT,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12, VSRC_X),
    BGE  -> List(Y, ALU_X, BR_GE,  OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12, VSRC_X),
    BLTU -> List(Y, ALU_X, BR_LTU, OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12, VSRC_X),
    BGEU -> List(Y, ALU_X, BR_GEU, OP1_RS1, OP2_RS2, IMM_B, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_12, VSRC_X),

    JAL   -> List(Y, ALU_ADD,    BR_X, OP1_PC,  OP2_IMM, IMM_J, Y, Y, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D, VSRC_X),
    JALR  -> List(Y, ALU_ADD,    BR_X, OP1_RS1, OP2_IMM, IMM_I, Y, N, Y, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    DELAY -> List(Y, ALU_X,      BR_X, OP1_X,   OP2_X,   IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_DELAY, XR_N, VSRC_X),
    VSETM -> List(Y, ALU_X,      BR_X, OP1_X,   OP2_X,   IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_VSETM, XR_N, VSRC_X),
    LUI   -> List(Y, ALU_PASS_B, BR_X, OP1_X,   OP2_IMM, IMM_U, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D, VSRC_X),
    AUIPC -> List(Y, ALU_ADD,    BR_X, OP1_PC,  OP2_IMM, IMM_U, Y, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D, VSRC_X),

    // ── CSR ──────────────────────────────────────────────────────────
    CSRRW  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RW,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    CSRRS  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RS,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    CSRRC  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_I, Y, N, N, CSR_RC,  DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D1, VSRC_X),
    CSRRWI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RWI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D, VSRC_X),
    CSRRSI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RSI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D, VSRC_X),
    CSRRCI -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_Z, Y, N, N, CSR_RCI, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D, VSRC_X),
    ECALL  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    EBREAK -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    FENCE  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X,   DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),

    // ── Scalar loads/stores ─────────────────────────────────────────
    LB   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LB,   LSU_X, SC_X, XR_D1, VSRC_X),
    LH   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LH,   LSU_X, SC_X, XR_D1, VSRC_X),
    LW   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LW,   LSU_X, SC_X, XR_D1, VSRC_X),
    LBU  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LBU,  LSU_X, SC_X, XR_D1, VSRC_X),
    LHU  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_I, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_LHU,  LSU_X, SC_X, XR_D1, VSRC_X),
    SELD -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_ZERO, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SELD, LSU_X, SC_X, XR_1, VSRC_X),
    SELI -> List(Y, ALU_X,   BR_X, OP1_X,   OP2_X,    IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X,    LSU_X, SC_SELI, XR_N, VSRC_X),
    SB   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SB,   LSU_X, SC_X, XR_12, VSRC_X),
    SH   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SH,   LSU_X, SC_X, XR_12, VSRC_X),
    SW   -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM,  IMM_S, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_SW,   LSU_X, SC_X, XR_12, VSRC_X),

    // ── Tensor load/store ───────────────────────────────────────────
    VLOAD        -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VLOAD,        SC_X, XR_1, VSRC_X),
    VSTORE       -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VSTORE,       SC_X, XR_1, VSRC_X),
    VLOAD_SCALE  -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VLOAD_SCALE,  SC_X, XR_1, VSRC_X),
    VSTORE_SCALE -> List(Y, ALU_ADD, BR_X, OP1_RS1, OP2_IMM, IMM_VLS, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_X, MEM_X, LSU_VSTORE_SCALE, SC_X, XR_1, VSRC_X),

    // ── DMA ─────────────────────────────────────────────────────────
    // DMA_CONFIG and DMA_WAIT name both a BitPat and a dma_cmd.
    DMA_LOAD                -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_LD,               N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    DMA_STORE               -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_ST,               N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_D12, VSRC_X),
    Instructions.DMA_CONFIG -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, ScalarISA.DMA_CONFIG, N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_1, VSRC_X),
    Instructions.DMA_WAIT   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, ScalarISA.DMA_WAIT,   N, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),

    // ── MXU slot 1 (mxu1) and slot 0 (the NVFP4 MXU) ────────────────
    VMATPUSH_WEIGHT_MXU1   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPUSH_ACC_FP8_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPUSH_ACC_BF16_MXU1 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPUSH_ACC_FP32_MXU1 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPOP_FP8_ACC_MXU1   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPOP_BF16_ACC_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPOP_BF16_SCALED_ACC_MXU1 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPOP_FP32_ACC_MXU1  -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATMUL_MXU1           -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATMUL_ACC_MXU1       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPUSH_WEIGHT_NV     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPUSH_ACC_BF16_NV   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPUSH_ACC_FP32_NV   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPOP_BF16_ACC_NV    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPOP_BF16_SCALED_ACC_NV -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATPOP_FP32_ACC_NV    -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATMUL_NV             -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMATMUL_ACC_NV         -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, Y, VPU_X, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),

    // ── VPU: element-wise binary ops ────────────────────────────────
    VADD_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_ADD,     XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VSUB_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SUB,     XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMUL_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_MUL,     XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMINIMUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMIN, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VMAXIMUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMAX, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),

    // ── VPU: .vx (x[rs2] scalar) and .vr (row 0 of T(vs2)) forms ──────────
    VADD_BF16_VX     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_ADD,     XLU_X, MEM_X, LSU_X, SC_X, XR_2, VSRC_XREG),
    VSUB_BF16_VX     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SUB,     XLU_X, MEM_X, LSU_X, SC_X, XR_2, VSRC_XREG),
    VMUL_BF16_VX     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_MUL,     XLU_X, MEM_X, LSU_X, SC_X, XR_2, VSRC_XREG),
    VMINIMUM_BF16_VX -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMIN, XLU_X, MEM_X, LSU_X, SC_X, XR_2, VSRC_XREG),
    VMAXIMUM_BF16_VX -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMAX, XLU_X, MEM_X, LSU_X, SC_X, XR_2, VSRC_XREG),
    VADD_BF16_VR     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_ADD,     XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_ROW0),
    VSUB_BF16_VR     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SUB,     XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_ROW0),
    VMUL_BF16_VR     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_MUL,     XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_ROW0),
    VMINIMUM_BF16_VR -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMIN, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_ROW0),
    VMAXIMUM_BF16_VR -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_PAIRMAX, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_ROW0),

    // ── VPU: sublane (column) reductions — result[i,j] = op(src[:,j]) ─
    VREDSUM_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CSUM, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VREDMIN_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CMIN, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VREDMAX_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CMAX, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),

    // ── VPU: lane (row) reductions — result[i,j] = op(src[i,:]) ────
    VREDSUM_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RSUM, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VREDMIN_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RMIN, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VREDMAX_ROW_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RMAX, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),

    // ── VPU: unary ops ──────────────────────────────────────────────
    VMOV             -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_MOV,       XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VRECIP_BF16      -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RCP,       XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VEXP_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_EXP,       XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VEXP2_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_EXP2,      XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VSQUARE_BF16     -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SQUARE,    XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VCUBE_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_CUBE,      XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VPACK_BF16_FP8   -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_FP8PACK,   XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VUNPACK_FP8_BF16 -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_FP8UNPACK, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VRELU_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_RELU,      XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VSIN_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SIN,       XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VCOS_BF16        -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_COS,       XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VTANH_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_TANH,      XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VLOG2_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LOG,       XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VSQRT_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_SQRT,      XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VGELU_BF16       -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_GELU,      XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),

    // ── VLI ─────────────────────────────────────────────────────────
    VLI_ALL -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ALL, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VLI_ROW -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ROW, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VLI_COL -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_COL, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),
    VLI_ONE -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_LI_ONE, XLU_X, MEM_X, LSU_X, SC_X, XR_N, VSRC_X),

    // ── XLU ─────────────────────────────────────────────────────────
    VTRPOSE_XLU -> List(Y, ALU_X, BR_X, OP1_X, OP2_X, IMM_X, N, N, N, CSR_X, DMA_X, N, VPU_X, XLU_TRPOSE, MEM_X, LSU_X, SC_X, XR_N, VSRC_X)
  )
}

# branch_delay_slots_mxu1_highbank_transpose.S
# @TIMEOUT 250000
# @DRAM_BASE 0x90000000
#
# Coverage intent:
#   - Taken BEQ with VLOAD side effect in a delay slot.
#   - Not-taken BEQ with VLOAD side effect in a delay slot.
#   - JAL delay-slot side effects launching XLU transpose and MXU1 push/matmul/pop.
#   - End-to-end DMA -> VMEM -> VLOAD -> XLU -> MXU1 -> VSTORE -> DMA path.
#   - Cold upper-bank activity in m53, m54, m55, m56, and m57.
#
# DRAM byte layout (generator-backed):
#   [0x0000 .. 0x03FF]  A_fp8 input tile                    (1024 B)
#   [0x0400 .. 0x07FF]  B_fp8 input tile                    (1024 B)
#   [0x0800 .. 0x0BFF]  output BF16 cols  0..15             (1024 B)
#   [0x0C00 .. 0x0FFF]  output BF16 cols 16..31             (1024 B)
#
# VMEM word-address layout:
#   x20 =    0   activation staging region   (1024 B)
#   x21 =  256   weight staging region       (1024 B)
#   x22 = 4096   output staging low half     (1024 B)
#   x22+256      output staging high half    (1024 B)
#
# MREG allocation:
#   m53        = activation tile A (FP8)
#   m54        = raw weight tile B (FP8)
#   m55        = transposed weight tile B^T (FP8)
#   m56,m57    = MXU1 BF16 output tile C = A @ B

    ADDI  x28, x0, 1

    LI    x5,  0x90000000
    DMA.CONFIG x5, 0

    LI    x10, 0           # DRAM byte offset of A
    LI    x11, 1024        # DRAM byte offset of B
    LI    x12, 1024        # 1 KiB transfer size
    LI    x13, 2048        # DRAM byte offset of output
    LI    x14, 2048        # 2 KiB transfer size

    ADDI  x20, x0, 0       # VMEM word addr for A staging
    ADDI  x21, x0, 256     # VMEM word addr for B staging
    LI    x22, 4096        # VMEM word addr for output staging

    ADDI  x24, x0, 0       # aggregate JAL fallthrough marker (must stay 0)
    ADDI  x25, x0, 0       # unexpected taken marker for not-taken BEQ
    ADDI  x26, x0, 0       # expected fallthrough marker for not-taken BEQ
    ADDI  x27, x0, 0       # skipped marker for taken BEQ
    ADDI  x29, x0, 1       # compare register for not-taken BEQ

    # Preload both FP8 tiles from DRAM into VMEM.
    DMA.LOAD  x20, x10, x12, 0
    DMA.WAIT  0
    DMA.LOAD  x21, x11, x12, 1
    DMA.WAIT  1

    # Taken branch: VLOAD in the first delay slot must execute before redirect.
    BEQ   x0, x0, act_loaded
    VLOAD 53, x20, 0
    NOP
    ADDI  x27, x27, 1      # must be skipped
act_loaded:
    DELAY 40

    # Not-taken branch: delay-slot VLOAD must still execute, and fallthrough must occur.
    BEQ   x0, x29, weight_taken
    VLOAD 54, x21, 0
    NOP
    ADDI  x26, x26, 1      # must execute exactly once
    JAL   x0, weight_done
    NOP
    NOP
weight_taken:
    ADDI  x25, x25, 1      # must stay zero
weight_done:
    DELAY 40

    # JAL delay-slot XLU transpose into another cold upper bank.
    JAL   x1, xlu_done
    VTRPOSE.XLU 55, 54
    NOP
    ADDI  x24, x24, 1      # must be skipped
xlu_done:
    DELAY 80

    # JAL delay-slot MXU1 weight push from the transposed bank.
    JAL   x1, push_done
    VMATPUSH.W.MXU1 0, 55
    NOP
    ADDI  x24, x24, 1      # must be skipped
push_done:
    DELAY 40

    # JAL delay-slot MXU1 compute launch.
    JAL   x1, compute_done
    VMATMUL.MXU1 0, 53, 0
    NOP
    ADDI  x24, x24, 1      # must be skipped
compute_done:
    DELAY 120

    # JAL delay-slot MXU1 BF16 pop into m56/m57.
    JAL   x1, pop_done
    VMATPOP.BF16.MXU1 56, 0
    NOP
    ADDI  x24, x24, 1      # must be skipped
pop_done:
    DELAY 40

    # Spill both BF16 halves to contiguous VMEM, then DMA-store them back.
    VSTORE 56, x22, 0
    DELAY 40
    VSTORE 57, x22, 8
    DELAY 40

    DMA.STORE x13, x22, x14, 2
    DMA.WAIT  2

    # Scalar control-path checks for taken / not-taken / JAL redirect behavior.
    ADDI  x28, x0, 2
    BNE   x27, x0, fail
    NOP
    NOP

    ADDI  x28, x0, 3
    ADDI  x30, x0, 1
    BNE   x26, x30, fail
    NOP
    NOP

    ADDI  x28, x0, 4
    BNE   x25, x0, fail
    NOP
    NOP

    ADDI  x28, x0, 5
    BNE   x24, x0, fail
    NOP
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

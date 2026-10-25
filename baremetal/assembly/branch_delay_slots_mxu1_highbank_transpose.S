# branch_delay_slots_mxu1_highbank_transpose.S
# @TIMEOUT 250000
# @DRAM_BASE 0x90000000
#
# Coverage intent:
#   - Taken BEQ with a VLOAD side effect just before it (no delay slot).
#   - Not-taken BEQ with a VLOAD side effect just before it.
#   - JALs just after XLU transpose and MXU1 push/matmul/pop launches.
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

    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x10, 0x90000000  # DRAM byte offset of A
    LI    x11, 0x90000400  # DRAM byte offset of B
    LI    x12, 1024        # 1 KiB transfer size
    LI    x13, 0x90000800  # DRAM byte offset of output
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
    dma.load x20, x10, x12, ch0
    dma.wait ch0
    dma.load x21, x11, x12, ch1
    dma.wait ch1

    # Taken branch: the VLOAD before it must execute; the next instruction is dropped.
    vload m53, 0(x20)
    BEQ   x0, x0, act_loaded
    ADDI  x27, x27, 1      # must be skipped
act_loaded:
    DELAY 32

    # Not-taken branch: the VLOAD before it must execute, and fallthrough must occur.
    vload m54, 0(x21)
    BEQ   x0, x29, weight_taken
    ADDI  x26, x26, 1      # must execute exactly once
    JAL   x0, weight_done
weight_taken:
    ADDI  x25, x25, 1      # must stay zero
weight_done:
    DELAY 32

    # XLU transpose into another cold upper bank, then a JAL.
    vtrpose.xlu m55, m54
    JAL   x1, xlu_done
    ADDI  x24, x24, 1      # must be skipped
xlu_done:
    DELAY 64

    # MXU1 weight push from the transposed bank, then a JAL.
    vmatpush.weight.mxu1 w0, m55
    JAL   x1, push_done
    ADDI  x24, x24, 1      # must be skipped
push_done:
    DELAY 31

    # MXU1 compute launch, then a JAL.
    vmatmul.mxu1 acc0, m53, w0
    JAL   x1, compute_done
    ADDI  x24, x24, 1      # must be skipped
compute_done:
    DELAY 34

    # MXU1 BF16 pop into m56/m57, then a JAL.
    vmatpop.bf16.acc.mxu1 m56, acc0
    JAL   x1, pop_done
    ADDI  x24, x24, 1      # must be skipped
pop_done:
    DELAY 31

    # Spill both BF16 halves to contiguous VMEM, then DMA-store them back.
    vstore m56, 0(x22)
    DELAY 33
    vstore m57, 8(x22)
    DELAY 33

    dma.store x13, x22, x14, ch2
    dma.wait ch2

    # Scalar control-path checks for taken / not-taken / JAL redirect behavior.
    ADDI  x28, x0, 2
    BNE   x27, x0, fail

    ADDI  x28, x0, 3
    ADDI  x30, x0, 1
    BNE   x26, x30, fail

    ADDI  x28, x0, 4
    BNE   x25, x0, fail

    ADDI  x28, x0, 5
    BNE   x24, x0, fail

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

# mxu1_back_to_back.S — mxu1 commands back to back, issued as the unit takes them
# @TIMEOUT 40000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu1_back_to_back.py
#
# acc0 = C0 + A0 W0^T + A2 W1^T + A4 W2^T from an FP8-pushed C0, and
# acc1 = C1 + A1 W1^T + A3 W2^T + A5 W3^T from a BF16-pushed C1, interleaved, with weight pushes
# into the slot that the previous matmul has just read. No DELAY: each command issues when mxu1
# takes it (cmdReady) and its registers are free. BF16 and FP8 pops of both accumulators end it.
#
# VMEM (bytes): A0..A5 0-6K, W0..W3 6K-10K, C0 10K-11K, C1 11K-13K, results 16K-22K

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 4096

    ADDI  x6, x0, 0                     # inputs: 13 KiB
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    LI    x7, 1024
    LI    x1, 0x90001000
    dma.load x7, x1, x2, ch1
    LI    x8, 2048
    LI    x1, 0x90002000
    dma.load x8, x1, x2, ch2
    LI    x9, 3072
    LI    x1, 0x90003000
    LI    x3, 1024
    dma.load x9, x1, x3, ch3
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    vload m0, 0(x6)
    vload m1, 8(x6)
    vload m2, 16(x6)
    vload m3, 24(x6)
    vload m4, 32(x6)
    vload m5, 40(x6)
    vload m8, 48(x6)
    vload m9, 56(x6)
    vload m10, 64(x6)
    vload m11, 72(x6)
    vload m12, 80(x6)
    vload m14, 88(x6)
    vload m15, 96(x6)

    seli  e1, 123
    seli  e2, 121
    vmatpush.acc.fp8.mxu1  acc0, m12    # C0
    vmatpush.acc.bf16.mxu1 acc1, m14    # C1
    vmatpush.weight.mxu1   w0, m8       # W0
    vmatpush.weight.mxu1   w1, m9       # W1
    vmatmul.acc.mxu1       acc0, m0, w0
    vmatmul.acc.mxu1       acc1, m1, w1
    vmatpush.weight.mxu1   w0, m10      # W2
    vmatmul.acc.mxu1       acc0, m2, w1
    vmatmul.acc.mxu1       acc1, m3, w0
    vmatpush.weight.mxu1   w1, m11      # W3
    vmatmul.acc.mxu1       acc0, m4, w0
    vmatmul.acc.mxu1       acc1, m5, w1
    vmatpop.bf16.acc.mxu1  m16, acc0
    vmatpop.fp8.acc.mxu1   m18, acc0, e1
    vmatpop.bf16.acc.mxu1  m20, acc1
    vmatpop.fp8.acc.mxu1   m22, acc1, e2

    LI    x10, 4096                     # results: 6 KiB
    vstore m16, 0(x10)
    vstore m17, 8(x10)
    vstore m18, 16(x10)
    vstore m20, 24(x10)
    vstore m21, 32(x10)
    vstore m22, 40(x10)
    LI    x11, 0x90004000               # a DMA transfer moves at most 4 KiB
    LI    x3, 4096
    dma.store x11, x10, x3, ch4
    LI    x11, 0x90005000
    ADDI  x12, x10, 1024                # 4 KiB further in VMEM (word address)
    LI    x3, 2048
    dma.store x11, x12, x3, ch5
    dma.wait ch4
    dma.wait ch5

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL
fail:
    CSRRW x0, 0x800, x28
    ECALL

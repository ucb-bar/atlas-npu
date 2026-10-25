# mxu1_scaled_pop.S — mxu1 FP32 push, scaled BF16 pops and FP32 pop
# @TIMEOUT 40000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu1_scaled_pop.py
#
# acc1 = an FP32 tile of special and random values (vmatpush.acc.fp32.mxu1). Back to back, with
# no DELAY: scaled BF16 pops at e = 0, 1, 126, 127, 128, 254 and 255, the plain BF16 pop and the
# FP32 pop. Each scaled pop is BF16_RNE(acc * 2^(e - 127)), one rounding.
#
# VMEM (bytes): tile 0-4K, results 8K-28K (pairs m8..m23, then m24..m27)

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 4096

    ADDI  x6, x0, 0                     # tile: 4 KiB
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    vload m4, 0(x6)
    vload m5, 8(x6)
    vload m6, 16(x6)
    vload m7, 24(x6)

    seli  e0, 0
    seli  e1, 1
    seli  e2, 126
    seli  e3, 127
    seli  e4, 128
    seli  e5, 254
    seli  e6, 255
    vmatpush.acc.fp32.mxu1        acc1, m4
    vmatpop.bf16.scaled.acc.mxu1  m8, acc1, e0
    vmatpop.bf16.scaled.acc.mxu1  m10, acc1, e1
    vmatpop.bf16.scaled.acc.mxu1  m12, acc1, e2
    vmatpop.bf16.scaled.acc.mxu1  m14, acc1, e3
    vmatpop.bf16.scaled.acc.mxu1  m16, acc1, e4
    vmatpop.bf16.scaled.acc.mxu1  m18, acc1, e5
    vmatpop.bf16.scaled.acc.mxu1  m20, acc1, e6
    vmatpop.bf16.acc.mxu1         m22, acc1
    vmatpop.fp32.acc.mxu1         m24, acc1

    LI    x10, 2048                     # results: 20 KiB
    vstore m8, 0(x10)
    vstore m9, 8(x10)
    vstore m10, 16(x10)
    vstore m11, 24(x10)
    vstore m12, 32(x10)
    vstore m13, 40(x10)
    vstore m14, 48(x10)
    vstore m15, 56(x10)
    vstore m16, 64(x10)
    vstore m17, 72(x10)
    vstore m18, 80(x10)
    vstore m19, 88(x10)
    vstore m20, 96(x10)
    vstore m21, 104(x10)
    vstore m22, 112(x10)
    vstore m23, 120(x10)
    vstore m24, 128(x10)
    vstore m25, 136(x10)
    vstore m26, 144(x10)
    vstore m27, 152(x10)
    LI    x11, 0x90001000
    dma.store x11, x10, x2, ch1
    LI    x12, 3072
    LI    x11, 0x90002000
    dma.store x11, x12, x2, ch2
    LI    x12, 4096
    LI    x11, 0x90003000
    dma.store x11, x12, x2, ch3
    LI    x12, 5120
    LI    x11, 0x90004000
    dma.store x11, x12, x2, ch4
    LI    x12, 6144
    LI    x11, 0x90005000
    dma.store x11, x12, x2, ch5
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL
fail:
    CSRRW x0, 0x800, x28
    ECALL

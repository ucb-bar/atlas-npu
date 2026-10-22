# nvfp4_fp32_specials.S — FP32 accumulator push/pop with special values (ipt-nvfp4 config)
# @TIMEOUT 20000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_nvfp4_fp32_specials.py
#
# acc0 = C + A @ W^T, C from m12..m15 (FP32, specials), scales with zero /
# subnormal / NaN / largest values; result m24..m27 (FP32).
#
# VMEM (bytes): A data 0, A scales 1024, W data 2048, W scales 3072,
#               C 4096-8191, result 8192-12287

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x3, 128
    LI    x4, 4096

    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    LI    x7, 256
    LI    x1, 0x90000400
    dma.load x7, x1, x3, ch1
    LI    x8, 512
    LI    x1, 0x90000500
    dma.load x8, x1, x2, ch2
    LI    x9, 768
    LI    x1, 0x90000900
    dma.load x9, x1, x3, ch3
    LI    x13, 1024                     # C (4 KiB)
    LI    x1, 0x90000A00
    dma.load x13, x1, x4, ch5
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch5

    vload        m0, 0(x6)
    vload.scale  m0, 0(x7)
    vload        m1, 0(x8)
    vload.scale  m1, 0(x9)
    vload        m12, 0(x13)
    vload        m13, 8(x13)
    vload        m14, 16(x13)
    vload        m15, 24(x13)
    vmatpush.weight.nv   w0, m1
    vmatpush.acc.fp32.nv acc0, m12
    vmatmul.acc.nv       acc0, m0, w0
    vmatpop.fp32.acc.nv  m24, acc0

    LI    x10, 2048                     # result: 8 KiB
    vstore m24, 0(x10)
    vstore m25, 8(x10)
    vstore m26, 16(x10)
    vstore m27, 24(x10)
    LI    x11, 0x90001C00
    dma.store x11, x10, x4, ch4
    dma.wait ch4

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL
fail:
    CSRRW x0, 0xC10, x28
    ECALL

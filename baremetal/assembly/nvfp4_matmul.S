# nvfp4_matmul.S — one NVFP4 MXU matmul, FP32 result (ipt-nvfp4 config)
# @TIMEOUT 20000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_nvfp4_matmul.py
#
# acc0 = A @ W^T over 64 NVFP4 elements per row (block scales from the
# m-registers' scale sidecars), popped as FP32 into m4..m7. No DELAY: the
# interlock spaces every dependent instruction.
#
# VMEM: A data @0, A scales @1024 B, W data @2048 B, W scales @3072 B,
#       result @4096 B (4 KiB, m4..m7)

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x3, 128

    ADDI  x6, x0, 0                     # A data
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    LI    x7, 256                       # A scales (VMEM word 256 = 1024 B)
    LI    x1, 0x90000400
    dma.load x7, x1, x3, ch1
    LI    x8, 512                       # W data
    LI    x1, 0x90000500
    dma.load x8, x1, x2, ch2
    LI    x9, 768                       # W scales
    LI    x1, 0x90000900
    dma.load x9, x1, x3, ch3
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

    vload        m0, 0(x6)
    vload.scale  m0, 0(x7)
    vload        m1, 0(x8)
    vload.scale  m1, 0(x9)
    vmatpush.weight.nv w0, m1
    vmatmul.nv   acc0, m0, w0
    vmatpop.fp32.acc.nv m4, acc0

    LI    x10, 1024                     # result: VMEM word 1024 = 4096 B
    vstore m4, 0(x10)
    vstore m5, 8(x10)
    vstore m6, 16(x10)
    vstore m7, 24(x10)
    LI    x11, 0x90001000
    LI    x12, 4096
    dma.store x11, x10, x12, ch4
    dma.wait ch4

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL
fail:
    CSRRW x0, 0xC10, x28
    ECALL

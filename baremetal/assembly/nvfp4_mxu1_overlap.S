# nvfp4_mxu1_overlap.S — NV MXU and mxu1 pushes, matmuls and pops in flight together
# @TIMEOUT 50000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_nvfp4_mxu1_overlap.py
#
# The two MXUs alternate instruction by instruction, with no DELAY: weight pushes, accumulator
# pushes (NV FP32 quad, mxu1 E4M3), matmul-accumulates, matmuls and pops (NV FP32, mxu1 BF16 and
# E4M3). Every register is in its own physical bank (m0-m27), so the interlock lets the MXUs
# overlap and orders only the dependent instructions of each MXU.
#   nv:   acc0 = C_nv + A_nv @ W_nv^T -> m16..m19;  acc1 = A_nv @ W_nv^T -> m24..m27
#   mxu1: acc0 = C_1 + A_1 @ W_1^T -> {m20, m21};  acc1 = A_1 @ W_1^T -> m22 (x 2^-3)
#
# VMEM (bytes): NV A data 0, A scales 1024, W data 2048, W scales 3072, C_nv 4096-8191,
#               A_1 8192, W_1 9216, C_1 10240; results 16384-27647

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x3, 128
    LI    x4, 4096

    ADDI  x6, x0, 0                     # NV A data
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    LI    x7, 256                       # NV A scales (VMEM word 256 = 1024 B)
    LI    x1, 0x90000400
    dma.load x7, x1, x3, ch1
    LI    x8, 512                       # NV W data
    LI    x1, 0x90000500
    dma.load x8, x1, x2, ch2
    LI    x9, 768                       # NV W scales
    LI    x1, 0x90000900
    dma.load x9, x1, x3, ch3
    LI    x10, 1024                     # C_nv (4 KiB)
    LI    x1, 0x90001000
    dma.load x10, x1, x4, ch4
    LI    x11, 2048                     # A_1, W_1 and C_1 (3 KiB)
    LI    x1, 0x90002000
    LI    x12, 3072
    dma.load x11, x1, x12, ch5
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5

    vload        m0, 0(x6)
    vload.scale  m0, 0(x7)
    vload        m1, 0(x8)
    vload.scale  m1, 0(x9)
    vload        m12, 0(x10)
    vload        m13, 8(x10)
    vload        m14, 16(x10)
    vload        m15, 24(x10)
    vload        m2, 0(x11)
    vload        m3, 8(x11)
    vload        m6, 16(x11)
    seli         e5, 124

    vmatpush.weight.nv    w0, m1
    vmatpush.weight.mxu1  w0, m3
    vmatpush.acc.fp32.nv  acc0, m12
    vmatpush.acc.fp8.mxu1 acc0, m6
    vmatmul.acc.nv        acc0, m0, w0
    vmatmul.acc.mxu1      acc0, m2, w0
    vmatmul.nv            acc1, m0, w0
    vmatmul.mxu1          acc1, m2, w0
    vmatpop.fp32.acc.nv   m16, acc0
    vmatpop.bf16.acc.mxu1 m20, acc0
    vmatpop.fp32.acc.nv   m24, acc1
    vmatpop.fp8.acc.mxu1  m22, acc1, e5

    LI    x13, 4096                     # results: VMEM word 4096 = 16384 B
    vstore m16, 0(x13)
    vstore m17, 8(x13)
    vstore m18, 16(x13)
    vstore m19, 24(x13)
    vstore m20, 32(x13)
    vstore m21, 40(x13)
    vstore m24, 48(x13)
    vstore m25, 56(x13)
    vstore m26, 64(x13)
    vstore m27, 72(x13)
    vstore m22, 80(x13)
    LI    x14, 0x90004000               # 11 KiB in transfers of at most 4 KiB
    dma.store x14, x13, x4, ch6
    LI    x14, 0x90005000
    LI    x15, 5120
    dma.store x14, x15, x4, ch7
    LI    x14, 0x90006000
    LI    x15, 6144
    dma.store x14, x15, x12, ch0
    dma.wait ch6
    dma.wait ch7
    dma.wait ch0

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL
fail:
    CSRRW x0, 0x800, x28
    ECALL

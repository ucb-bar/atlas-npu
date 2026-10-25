# nvfp4_fp32_spill.S — spill and restore an NVFP4 MXU accumulator in a vmatmul.acc.nv chain
# (ipt-nvfp4 config)
# @TIMEOUT 40000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_nvfp4_fp32_spill.py
#
# acc0 and acc1 = C + A0 W0^T + A1 W1^T + A2 W2^T + A3 W3^T, four K = 64 steps from a BF16 C.
# After the second step acc0 goes to m24..m27 (FP32 pop), acc0 holds other work (X + A2 W2^T),
# and the FP32 push restores it. acc1 runs the same chain without the spill: both results must
# equal the chain of four NVFP4 MMAs bit for bit. No DELAY: the interlock orders all.
#
# VMEM (bytes): A data 0-4K, A scales 4K-4.5K, W data 8K-12K, W scales 12K-12.5K,
#               C 16K-18K, X 18K-20K, results 20K-34K

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 4096
    LI    x3, 512

    ADDI  x6, x0, 0                     # A data
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    LI    x7, 1024                      # A scales
    LI    x1, 0x90001000
    dma.load x7, x1, x3, ch1
    LI    x8, 2048                      # W data
    LI    x1, 0x90001200
    dma.load x8, x1, x2, ch2
    LI    x9, 3072                      # W scales
    LI    x1, 0x90002200
    dma.load x9, x1, x3, ch3
    LI    x13, 4096                     # C, X (BF16 tiles)
    LI    x1, 0x90002400
    dma.load x13, x1, x2, ch5
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch5

    vload        m0, 0(x6)
    vload        m1, 8(x6)
    vload        m2, 16(x6)
    vload        m3, 24(x6)
    vload.scale  m0, 0(x7)
    vload.scale  m1, 1(x7)
    vload.scale  m2, 2(x7)
    vload.scale  m3, 3(x7)
    vload        m8, 0(x8)
    vload        m9, 8(x8)
    vload        m10, 16(x8)
    vload        m11, 24(x8)
    vload.scale  m8, 0(x9)
    vload.scale  m9, 1(x9)
    vload.scale  m10, 2(x9)
    vload.scale  m11, 3(x9)
    vload        m16, 0(x13)
    vload        m17, 8(x13)
    vload        m18, 16(x13)
    vload        m19, 24(x13)

    vmatpush.acc.bf16.nv acc0, m16      # C
    vmatpush.acc.bf16.nv acc1, m16
    vmatpush.weight.nv   w0, m8         # W0
    vmatpush.weight.nv   w1, m9         # W1
    vmatmul.acc.nv       acc0, m0, w0
    vmatmul.acc.nv       acc1, m0, w0
    vmatmul.acc.nv       acc0, m1, w1
    vmatmul.acc.nv       acc1, m1, w1
    vmatpop.fp32.acc.nv  m24, acc0      # spill
    vmatpush.acc.bf16.nv acc0, m18      # other work on acc0: X + A2 W2^T
    vmatpush.weight.nv   w0, m10        # W2
    vmatmul.acc.nv       acc0, m2, w0
    vmatpop.bf16.acc.nv  m20, acc0
    vmatpush.acc.fp32.nv acc0, m24      # restore
    vmatpush.weight.nv   w1, m11        # W3
    vmatmul.acc.nv       acc0, m2, w0
    vmatmul.acc.nv       acc1, m2, w0
    vmatmul.acc.nv       acc0, m3, w1
    vmatmul.acc.nv       acc1, m3, w1
    vmatpop.fp32.acc.nv  m28, acc0
    vmatpop.fp32.acc.nv  m32, acc1

    LI    x10, 5120                     # results: 14 KiB
    vstore m20, 0(x10)
    vstore m21, 8(x10)
    vstore m24, 16(x10)
    vstore m25, 24(x10)
    vstore m26, 32(x10)
    vstore m27, 40(x10)
    vstore m28, 48(x10)
    vstore m29, 56(x10)
    vstore m30, 64(x10)
    vstore m31, 72(x10)
    vstore m32, 80(x10)
    vstore m33, 88(x10)
    vstore m34, 96(x10)
    vstore m35, 104(x10)
    LI    x11, 0x90003400
    dma.store x11, x10, x2, ch4
    LI    x12, 6144
    LI    x11, 0x90004400
    dma.store x11, x12, x2, ch6
    LI    x12, 7168
    LI    x11, 0x90005400
    dma.store x11, x12, x2, ch7
    LI    x12, 8192
    LI    x11, 0x90006400
    LI    x3, 2048
    dma.store x11, x12, x3, ch1
    dma.wait ch4
    dma.wait ch6
    dma.wait ch7
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL
fail:
    CSRRW x0, 0x800, x28
    ECALL

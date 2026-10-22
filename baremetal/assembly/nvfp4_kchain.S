# nvfp4_kchain.S — K = 256 NVFP4 chain with double-buffered weights (ipt-nvfp4 config)
# @TIMEOUT 40000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_nvfp4_kchain.py
#
# acc0 = A0 W0^T + A1 W1^T + A2 W2^T + A3 W3^T. Weights alternate between w0
# and w1; each push overlaps the matmul on the other slot. DBG1 = cycles from
# the first push to the pop's issue (four matmuls take at least 128).
#
# VMEM (bytes): A data 0-4K, A scales 4K-4.5K, W data 8K-12K, W scales 12K-12.5K,
#               result 16K-20K

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 4096
    LI    x3, 512

    ADDI  x6, x0, 0                     # A data (4 tiles)
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
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

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

    csrrs x20, 0xC00, x0                # start
    vmatpush.weight.nv w0, m8
    vmatpush.weight.nv w1, m9
    vmatmul.nv     acc0, m0, w0
    vmatpush.weight.nv w0, m10          # waits for the matmul to finish reading w0
    vmatmul.acc.nv acc0, m1, w1
    vmatpush.weight.nv w1, m11
    vmatmul.acc.nv acc0, m2, w0
    vmatmul.acc.nv acc0, m3, w1
    vmatpop.fp32.acc.nv m16, acc0
    csrrs x21, 0xC00, x0                # the pop has issued
    sub   x22, x21, x20
    csrrw x0, 0xC11, x22

    LI    x10, 4096                     # result: 16 KiB
    vstore m16, 0(x10)
    vstore m17, 8(x10)
    vstore m18, 16(x10)
    vstore m19, 24(x10)
    LI    x11, 0x90002400
    dma.store x11, x10, x2, ch4
    dma.wait ch4

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL
fail:
    CSRRW x0, 0xC10, x28
    ECALL

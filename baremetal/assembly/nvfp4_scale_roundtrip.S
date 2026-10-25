# nvfp4_scale_roundtrip.S — scale sidecar load/store (ipt-nvfp4 config)
# @TIMEOUT 20000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_nvfp4_scale_roundtrip.py
#
# VMEM (bytes): scales 0, data 1024, scales out 4096, data out 5120,
#               scales via m9 8192

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x3, 128

    ADDI  x6, x0, 0                     # scales
    LI    x1, 0x90000000
    dma.load x6, x1, x3, ch0
    LI    x7, 256                       # data
    LI    x1, 0x90000100
    dma.load x7, x1, x2, ch1
    dma.wait ch0
    dma.wait ch1

    vload.scale  m6, 0(x6)
    vload        m6, 0(x7)              # data does not touch the scales
    LI    x8, 1024
    vstore.scale m6, 0(x8)
    LI    x9, 1280
    vstore       m6, 0(x9)
    vload.scale  m9, 0(x8)
    LI    x10, 2048
    vstore.scale m9, 0(x10)

    LI    x11, 0x90000600               # scales out
    dma.store x11, x8, x3, ch2
    LI    x11, 0x90000700               # data out
    dma.store x11, x9, x2, ch3
    LI    x11, 0x90000C00               # scales via m9
    dma.store x11, x10, x3, ch4
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL
fail:
    CSRRW x0, 0x800, x28
    ECALL

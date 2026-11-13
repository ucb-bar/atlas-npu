# Repeated 1024-byte DMA reads, each completed before issuing the next.
# @TIMEOUT 2000000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_perf_read_fenced_1024b.py
# @PERF_REPORT

    LI x5, 0
    dma.config x5, ch0
    LI x12, 1024

    csrrs x20, 0xC00, x0
    LI x6, 0x1000
    LI x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI x6, 0x1100
    LI x10, 0x90002000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI x6, 0x1200
    LI x10, 0x90004000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI x6, 0x1300
    LI x10, 0x90006000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI x6, 0x1400
    LI x10, 0x90008000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI x6, 0x1500
    LI x10, 0x9000a000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI x6, 0x1600
    LI x10, 0x9000c000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI x6, 0x1700
    LI x10, 0x9000e000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    csrrs x21, 0xC00, x0
    SUB x22, x21, x20
    csrrw x0, 0xC11, x22

    LI x6, 0x1000
    LI x10, 0x90020000
    dma.store x10, x6, x12, ch0
    dma.wait ch0
    LI x6, 0x1100
    LI x10, 0x90022000
    dma.store x10, x6, x12, ch0
    dma.wait ch0
    LI x6, 0x1200
    LI x10, 0x90024000
    dma.store x10, x6, x12, ch0
    dma.wait ch0
    LI x6, 0x1300
    LI x10, 0x90026000
    dma.store x10, x6, x12, ch0
    dma.wait ch0
    LI x6, 0x1400
    LI x10, 0x90028000
    dma.store x10, x6, x12, ch0
    dma.wait ch0
    LI x6, 0x1500
    LI x10, 0x9002a000
    dma.store x10, x6, x12, ch0
    dma.wait ch0
    LI x6, 0x1600
    LI x10, 0x9002c000
    dma.store x10, x6, x12, ch0
    dma.wait ch0
    LI x6, 0x1700
    LI x10, 0x9002e000
    dma.store x10, x6, x12, ch0
    dma.wait ch0

    LI x30, 1
    csrrw x0, 0xC10, x30
    ECALL

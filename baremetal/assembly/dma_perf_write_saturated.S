# dma_perf_write_saturated.S — write, 1024 bytes/command, 8 command(s), saturated
# @TIMEOUT 2000000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_perf_write_saturated.py
# @PERF_REPORT
# DBG1 measures the selected direction only, including issue/setup instructions
# and DMA.WAIT drain. Opposite-direction staging/verification is outside timing.
# The PMU collects the entire run; use the selected direction's counters.

    LI x5, 0
    dma.config x5, ch0
    LI x12, 1024

# Stage source data before timing.
    LI x6, 0x1000
    LI x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI x6, 0x1400
    LI x10, 0x90002000
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    LI x6, 0x1800
    LI x10, 0x90004000
    dma.load x6, x10, x12, ch2
    dma.wait ch2
    LI x6, 0x1c00
    LI x10, 0x90006000
    dma.load x6, x10, x12, ch3
    dma.wait ch3
    LI x6, 0x2000
    LI x10, 0x90008000
    dma.load x6, x10, x12, ch4
    dma.wait ch4
    LI x6, 0x2400
    LI x10, 0x9000a000
    dma.load x6, x10, x12, ch5
    dma.wait ch5
    LI x6, 0x2800
    LI x10, 0x9000c000
    dma.load x6, x10, x12, ch6
    dma.wait ch6
    LI x6, 0x2c00
    LI x10, 0x9000e000
    dma.load x6, x10, x12, ch7
    dma.wait ch7

    csrrs x20, 0xC00, x0
    LI x6, 0x1000
    LI x10, 0x90020000
    dma.store x10, x6, x12, ch0
    LI x6, 0x1400
    LI x10, 0x90022000
    dma.store x10, x6, x12, ch1
    LI x6, 0x1800
    LI x10, 0x90024000
    dma.store x10, x6, x12, ch2
    LI x6, 0x1c00
    LI x10, 0x90026000
    dma.store x10, x6, x12, ch3
    LI x6, 0x2000
    LI x10, 0x90028000
    dma.store x10, x6, x12, ch4
    LI x6, 0x2400
    LI x10, 0x9002a000
    dma.store x10, x6, x12, ch5
    LI x6, 0x2800
    LI x10, 0x9002c000
    dma.store x10, x6, x12, ch6
    LI x6, 0x2c00
    LI x10, 0x9002e000
    dma.store x10, x6, x12, ch7
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7
    csrrs x21, 0xC00, x0
    SUB x22, x21, x20
    csrrw x0, 0xC11, x22

    LI x30, 1
    csrrw x0, 0xC10, x30
    ECALL

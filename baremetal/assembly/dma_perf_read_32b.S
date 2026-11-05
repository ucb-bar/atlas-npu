# dma_perf_read_32b.S — read, 32 bytes/command, 1 command(s), single
# @TIMEOUT 2000000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_perf_read_32b.py
# @PERF_REPORT
# DBG1 measures the selected direction only, including issue/setup instructions
# and DMA.WAIT drain. Opposite-direction staging/verification is outside timing.
# The PMU collects the entire run; use the selected direction's counters.

    LI x5, 0
    dma.config x5, ch0
    LI x12, 32

    csrrs x20, 0xC00, x0
    LI x6, 0x1000
    LI x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    csrrs x21, 0xC00, x0
    SUB x22, x21, x20
    csrrw x0, 0xC11, x22

# Echo read data for host golden checks after timing.
    LI x6, 0x1000
    LI x10, 0x90020000
    dma.store x10, x6, x12, ch0
    dma.wait ch0

    LI x30, 1
    csrrw x0, 0xC10, x30
    ECALL

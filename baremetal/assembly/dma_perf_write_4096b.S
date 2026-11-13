# dma_perf_write_4096b.S — write, 4096 bytes/command, 1 command(s), single
# @TIMEOUT 2000000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_perf_write_4096b.py
# @PERF_REPORT
# DBG1 measures only the selected DMA instruction and its completion wait.
# The PMU collects the entire run; use the selected direction's counters.

    LI x5, 0
    dma.config x5, ch0
    LI x12, 4096

# Stage source data before timing.
    LI x6, 0x1000
    LI x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    LI x6, 0x1000
    LI x10, 0x90020000
    csrrs x20, 0xC00, x0
    dma.store x10, x6, x12, ch0
    dma.wait ch0
    csrrs x21, 0xC00, x0
    SUB x22, x21, x20
    csrrw x0, 0xC11, x22

    LI x30, 1
    csrrw x0, 0xC10, x30
    ECALL

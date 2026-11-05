# dma_perf_write_vmem_contention.S — write, 4096 bytes/command, 1 command(s), vmem_contention
# @TIMEOUT 2000000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_perf_write_vmem_contention.py
# @PERF_REPORT
# DBG1 measures the selected direction only, including issue/setup instructions
# and DMA.WAIT drain. Opposite-direction staging/verification is outside timing.
# The PMU collects the entire run; use the selected direction's counters.

    LI x5, 0
    dma.config x5, ch0
    LI x12, 4096

# Stage source data before timing.
    LI x6, 0x1000
    LI x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    csrrs x20, 0xC00, x0
    LI x6, 0x1000
    LI x10, 0x90020000
    dma.store x10, x6, x12, ch0
# LSU writes another line in VMEM bank 0, competing with DMA without changing its payload.
    LI x7, 256
    LI x8, 0x12345678
contend:
    SW x8, 0(x0)
    ADDI x7, x7, -1
    BNE x7, x0, contend
    NOP
    dma.wait ch0
    csrrs x21, 0xC00, x0
    SUB x22, x21, x20
    csrrw x0, 0xC11, x22

    LI x30, 1
    csrrw x0, 0xC10, x30
    ECALL

# dma_perf_baseline.S — empty timing window for DMA benchmark overhead
# @TIMEOUT 1000
# @PERF_REPORT
    csrrs x20, 0xC00, x0
    csrrs x21, 0xC00, x0
    SUB x22, x21, x20
    csrrw x0, 0xC11, x22
    LI x30, 1
    csrrw x0, 0xC10, x30
    ECALL

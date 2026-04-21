# dma_reconfig_base.S - DMA.CONFIG base register changes between transfers
# @TIMEOUT 100000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_reconfig_base.py
#
# Existing tests configure one DMA base and leave it alone. This one switches
# between two DRAM base windows and proves both loads and stores use the latest
# DMA.CONFIG value.

    LI    x12, 128

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0

    LI    x5, 0x90001000
    DMA.CONFIG x5, 0
    LI    x6, 256
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    ADDI  x6, x0, 0
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 2
    DMA.WAIT  2

    LI    x5, 0x90001000
    DMA.CONFIG x5, 0
    LI    x6, 256
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 3
    DMA.WAIT  3

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

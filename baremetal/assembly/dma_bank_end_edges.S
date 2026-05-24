# dma_bank_end_edges.S - legacy DMA windows at the former end lines of each VMEM bank
# @TIMEOUT 300000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_bank_end_edges.py
#
# VMEM banks are now 65 536 words each (256 KiB). This test places 128-byte
# DMA windows at the current bank ends. Six channels cover the exact end window
# of each bank, and two extra channels cover the immediately preceding 32-word
# window in bank 0 and bank 5 so the full 8-channel DMA interface stays active.

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x12, 128

# Load one bank-end region per channel.
    LI    x6, 65472
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x12, 0

    LI    x6, 65504
    LI    x10, 0x90000400
    DMA.LOAD  x6, x10, x12, 1

    LI    x6, 131040
    LI    x10, 0x90000800
    DMA.LOAD  x6, x10, x12, 2

    LI    x6, 196576
    LI    x10, 0x90000C00
    DMA.LOAD  x6, x10, x12, 3

    LI    x6, 262112
    LI    x10, 0x90001000
    DMA.LOAD  x6, x10, x12, 4

    LI    x6, 327648
    LI    x10, 0x90001400
    DMA.LOAD  x6, x10, x12, 5

    LI    x6, 393152
    LI    x10, 0x90001800
    DMA.LOAD  x6, x10, x12, 6

    LI    x6, 393184
    LI    x10, 0x90001C00
    DMA.LOAD  x6, x10, x12, 7

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4
    DMA.WAIT  5
    DMA.WAIT  6
    DMA.WAIT  7

# Store all bank-end regions back out.
    LI    x6, 65472
    LI    x10, 0x90004000
    DMA.STORE x10, x6, x12, 0

    LI    x6, 65504
    LI    x10, 0x90004400
    DMA.STORE x10, x6, x12, 1

    LI    x6, 131040
    LI    x10, 0x90004800
    DMA.STORE x10, x6, x12, 2

    LI    x6, 196576
    LI    x10, 0x90004C00
    DMA.STORE x10, x6, x12, 3

    LI    x6, 262112
    LI    x10, 0x90005000
    DMA.STORE x10, x6, x12, 4

    LI    x6, 327648
    LI    x10, 0x90005400
    DMA.STORE x10, x6, x12, 5

    LI    x6, 393152
    LI    x10, 0x90005800
    DMA.STORE x10, x6, x12, 6

    LI    x6, 393184
    LI    x10, 0x90005C00
    DMA.STORE x10, x6, x12, 7

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4
    DMA.WAIT  5
    DMA.WAIT  6
    DMA.WAIT  7

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

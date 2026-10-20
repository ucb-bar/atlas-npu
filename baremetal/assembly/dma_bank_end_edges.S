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
    dma.config x5, ch0
    LI    x12, 128

# Load one bank-end region per channel.
    LI    x6, 65472
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0

    LI    x6, 65504
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1

    LI    x6, 131040
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2

    LI    x6, 196576
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch3

    LI    x6, 262112
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch4

    LI    x6, 327648
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch5

    LI    x6, 393152
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch6

    LI    x6, 393184
    LI    x10, 0x90001C00
    dma.load x6, x10, x12, ch7

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

# Store all bank-end regions back out.
    LI    x6, 65472
    LI    x10, 0x90004000
    dma.store x10, x6, x12, ch0

    LI    x6, 65504
    LI    x10, 0x90004400
    dma.store x10, x6, x12, ch1

    LI    x6, 131040
    LI    x10, 0x90004800
    dma.store x10, x6, x12, ch2

    LI    x6, 196576
    LI    x10, 0x90004C00
    dma.store x10, x6, x12, ch3

    LI    x6, 262112
    LI    x10, 0x90005000
    dma.store x10, x6, x12, ch4

    LI    x6, 327648
    LI    x10, 0x90005400
    dma.store x10, x6, x12, ch5

    LI    x6, 393152
    LI    x10, 0x90005800
    dma.store x10, x6, x12, ch6

    LI    x6, 393184
    LI    x10, 0x90005C00
    dma.store x10, x6, x12, ch7

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL

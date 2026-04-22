# dma_bank_end_edges.S - DMA at the last lines of every VMEM bank
# @TIMEOUT 300000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_bank_end_edges.py
#
# The block-banked VMEM has 32768 words per bank. Starting at word 32736 within
# a bank and transferring 128 bytes touches local lines 4092..4095, ending at
# the last word in that bank without crossing into the next bank.

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 128

# Load one bank-end region per channel.
    LI    x6, 32736
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0

    LI    x6, 65504
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 1

    LI    x6, 98272
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 2

    LI    x6, 131040
    LI    x10, 0x0c00
    DMA.LOAD  x6, x10, x12, 3

    LI    x6, 163808
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 4

    LI    x6, 196576
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 5

    LI    x6, 229344
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 6

    LI    x6, 262112
    LI    x10, 0x1c00
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
    LI    x6, 32736
    LI    x10, 0x4000
    DMA.STORE x10, x6, x12, 0

    LI    x6, 65504
    LI    x10, 0x4400
    DMA.STORE x10, x6, x12, 1

    LI    x6, 98272
    LI    x10, 0x4800
    DMA.STORE x10, x6, x12, 2

    LI    x6, 131040
    LI    x10, 0x4c00
    DMA.STORE x10, x6, x12, 3

    LI    x6, 163808
    LI    x10, 0x5000
    DMA.STORE x10, x6, x12, 4

    LI    x6, 196576
    LI    x10, 0x5400
    DMA.STORE x10, x6, x12, 5

    LI    x6, 229344
    LI    x10, 0x5800
    DMA.STORE x10, x6, x12, 6

    LI    x6, 262112
    LI    x10, 0x5c00
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

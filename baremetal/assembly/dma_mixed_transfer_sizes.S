# dma_mixed_transfer_sizes.S - mixed DMA command sizes in one burst
# @TIMEOUT 200000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_mixed_transfer_sizes.py
#
# Existing DMA tests mostly use 1024-byte tiles, with one 128-byte loopback.
# This test queues several legal sub-tile sizes together so the DMA's per-slot
# outstanding counters and isLast metadata see 1, 2, 3, 5, and 7 beat commands.

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0

    LI    x12, 32
    LI    x13, 64
    LI    x14, 96
    LI    x15, 160
    LI    x16, 224

# Load five differently sized regions before waiting.
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0

    LI    x6, 256
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x13, 1

    LI    x6, 512
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x14, 2

    LI    x6, 768
    LI    x10, 0x0c00
    DMA.LOAD  x6, x10, x15, 3

    LI    x6, 1024
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x16, 4

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4

# Store the same differently sized regions back out before waiting.
    ADDI  x6, x0, 0
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 0

    LI    x6, 256
    LI    x10, 0x2400
    DMA.STORE x10, x6, x13, 1

    LI    x6, 512
    LI    x10, 0x2800
    DMA.STORE x10, x6, x14, 2

    LI    x6, 768
    LI    x10, 0x2c00
    DMA.STORE x10, x6, x15, 3

    LI    x6, 1024
    LI    x10, 0x3000
    DMA.STORE x10, x6, x16, 4

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

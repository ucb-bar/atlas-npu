# dma_mixed_transfer_sizes.S - mixed DMA command sizes in one burst
# @TIMEOUT 200000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_mixed_transfer_sizes.py
#
# Existing DMA tests mostly use 1024-byte tiles, with one 128-byte loopback.
# This test queues several legal sub-tile sizes together so the DMA's per-slot
# outstanding counters and isLast metadata see 1, 2, 3, 5, and 7 beat commands.

    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x12, 32
    LI    x13, 64
    LI    x14, 96
    LI    x15, 160
    LI    x16, 224

# Load five differently sized regions before waiting.
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0

    LI    x6, 256
    LI    x10, 0x90000400
    dma.load x6, x10, x13, ch1

    LI    x6, 512
    LI    x10, 0x90000800
    dma.load x6, x10, x14, ch2

    LI    x6, 768
    LI    x10, 0x90000C00
    dma.load x6, x10, x15, ch3

    LI    x6, 1024
    LI    x10, 0x90001000
    dma.load x6, x10, x16, ch4

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4

# Store the same differently sized regions back out before waiting.
    ADDI  x6, x0, 0
    LI    x10, 0x90002000
    dma.store x10, x6, x12, ch0

    LI    x6, 256
    LI    x10, 0x90002400
    dma.store x10, x6, x13, ch1

    LI    x6, 512
    LI    x10, 0x90002800
    dma.store x10, x6, x14, ch2

    LI    x6, 768
    LI    x10, 0x90002C00
    dma.store x10, x6, x15, ch3

    LI    x6, 1024
    LI    x10, 0x90003000
    dma.store x10, x6, x16, ch4

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4

    LI    x30, 1
    csrrw x0, 0x800, x30
    ECALL

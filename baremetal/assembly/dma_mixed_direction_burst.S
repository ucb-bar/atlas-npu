# dma_mixed_direction_burst.S - queue DMA loads and stores in the same burst
# @TIMEOUT 400000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_dma_mixed_direction_burst.py
#
# This exercises source-ID metadata with both response directions live:
# AccessAckData beats from LOADs and AccessAck beats from STOREs can overlap
# once the memory system has outstanding work from both command types.

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 256

# Prefill four VMEM store-source tiles.
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0

    LI    x6, 256
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1

    LI    x6, 512
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 2
    DMA.WAIT  2

    LI    x6, 768
    LI    x10, 0x0c00
    DMA.LOAD  x6, x10, x12, 3
    DMA.WAIT  3

# Queue alternating LOAD and STORE commands before any wait.
    LI    x6, 1024
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 0

    ADDI  x6, x0, 0
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 1

    LI    x6, 1280
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 2

    LI    x6, 256
    LI    x10, 0x2400
    DMA.STORE x10, x6, x12, 3

    LI    x6, 1536
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 4

    LI    x6, 512
    LI    x10, 0x2800
    DMA.STORE x10, x6, x12, 5

    LI    x6, 1792
    LI    x10, 0x1c00
    DMA.LOAD  x6, x10, x12, 6

    LI    x6, 768
    LI    x10, 0x2c00
    DMA.STORE x10, x6, x12, 7

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4
    DMA.WAIT  5
    DMA.WAIT  6
    DMA.WAIT  7

# Store the four loaded tiles so the LOAD side of the mixed burst is checked.
    LI    x6, 1024
    LI    x10, 0x3000
    DMA.STORE x10, x6, x12, 0

    LI    x6, 1280
    LI    x10, 0x3400
    DMA.STORE x10, x6, x12, 1

    LI    x6, 1536
    LI    x10, 0x3800
    DMA.STORE x10, x6, x12, 2

    LI    x6, 1792
    LI    x10, 0x3c00
    DMA.STORE x10, x6, x12, 3

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

# dma_mixed_direction_burst.S - queue DMA loads and stores in the same burst
# @TIMEOUT 400000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_mixed_direction_burst.py
#
# This exercises source-ID metadata with both response directions live:
# AccessAckData beats from LOADs and AccessAck beats from STOREs can overlap
# once the memory system has outstanding work from both command types.

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 256

# Prefill four VMEM store-source tiles.
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    LI    x6, 256
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1
    dma.wait ch1

    LI    x6, 512
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2
    dma.wait ch2

    LI    x6, 768
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch3
    dma.wait ch3

# Queue alternating LOAD and STORE commands before any wait.
    LI    x6, 1024
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch0

    ADDI  x6, x0, 0
    LI    x10, 0x90002000
    dma.store x10, x6, x12, ch1

    LI    x6, 1280
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch2

    LI    x6, 256
    LI    x10, 0x90002400
    dma.store x10, x6, x12, ch3

    LI    x6, 1536
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch4

    LI    x6, 512
    LI    x10, 0x90002800
    dma.store x10, x6, x12, ch5

    LI    x6, 1792
    LI    x10, 0x90001C00
    dma.load x6, x10, x12, ch6

    LI    x6, 768
    LI    x10, 0x90002C00
    dma.store x10, x6, x12, ch7

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

# Store the four loaded tiles so the LOAD side of the mixed burst is checked.
    LI    x6, 1024
    LI    x10, 0x90003000
    dma.store x10, x6, x12, ch0

    LI    x6, 1280
    LI    x10, 0x90003400
    dma.store x10, x6, x12, ch1

    LI    x6, 1536
    LI    x10, 0x90003800
    dma.store x10, x6, x12, ch2

    LI    x6, 1792
    LI    x10, 0x90003C00
    dma.store x10, x6, x12, ch3

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

    LI    x30, 1
    csrrw x0, 0x800, x30
    ECALL

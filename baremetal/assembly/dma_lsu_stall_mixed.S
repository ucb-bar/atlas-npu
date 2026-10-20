# dma_lsu_stall_mixed.S - DMA mixed directions under repeated LSU grant stalls
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_lsu_stall_mixed.py
#
# Baremetal cannot force TileLink D-channel reordering; that needs a harness or
# memory model that deliberately returns grants out of order. This test creates
# the pressure pattern that should be paired with such a responder:
#   - mixed DMA.LOAD and DMA.STORE commands queued together
#   - scalar stores repeatedly hit the same banks as DMA LOAD writebacks
#   - scalar loads repeatedly hit the same banks as DMA STORE VMEM reads
#   - final checks prove both load-data and store-data slots retire correctly

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024

# ============================================================================
# Prefill VMEM banks 4..5. Two tiles per bank become the source tiles for the
# mixed-burst DMA.STORE half.
# ============================================================================

    LI    x6, 262144
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0

    LI    x6, 262400
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1

    LI    x6, 327680
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2

    LI    x6, 327936
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch3

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

# Scalar scratch byte addresses. Each one is in the same bank as a DMA region
# but outside the 1024-byte DMA tile being checked.
    LI    x20, 0x00800       # bank 0 scratch
    LI    x21, 0x40800       # bank 1 scratch
    LI    x22, 0x80800       # bank 2 scratch
    LI    x23, 0xC0800       # bank 3 scratch
    LI    x24, 0x100800      # bank 4 scratch, slot 0
    LI    x25, 0x101800      # bank 4 scratch, slot 1
    LI    x26, 0x140800      # bank 5 scratch, slot 0
    LI    x27, 0x141800      # bank 5 scratch, slot 1
    LI    x13, 0xCAFED00D
    LI    x18, 24

# ============================================================================
# Mixed DMA burst: LOADs target banks 0..3, STOREs source banks 4..5
# (two tiles per bank).
# ============================================================================

    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch0

    LI    x6, 262144
    LI    x10, 0x90003000
    dma.store x10, x6, x12, ch1

    LI    x6, 65536
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch2

    LI    x6, 262400
    LI    x10, 0x90003400
    dma.store x10, x6, x12, ch3

    LI    x6, 131072
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch4

    LI    x6, 327680
    LI    x10, 0x90003800
    dma.store x10, x6, x12, ch5

    LI    x6, 196608
    LI    x10, 0x90001C00
    dma.load x6, x10, x12, ch6

    LI    x6, 327936
    LI    x10, 0x90003C00
    dma.store x10, x6, x12, ch7

# ============================================================================
# LSU pressure loop. Stores deny DMA write grants on banks 0..3; loads deny DMA
# read grants on banks 4..5. Each scalar load uses DELAY 2 so the loop respects
# the longer fixed scalar-load latency.
# ============================================================================

lsu_stall_loop:
    sw x13, 0(x20)
    sw x13, 0(x21)
    sw x13, 0(x22)
    sw x13, 0(x23)

    lw x14, 0(x24)
    DELAY 2
    lw x15, 0(x25)
    DELAY 2
    lw x16, 0(x26)
    DELAY 2
    lw x17, 0(x27)
    DELAY 2

    ADDI  x18, x18, -1
    BNE   x18, x0, lsu_stall_loop
    NOP

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

# Store the four DMA-loaded tiles back out so the LOAD half is checked.
    ADDI  x6, x0, 0
    LI    x10, 0x90004000
    dma.store x10, x6, x12, ch0

    LI    x6, 65536
    LI    x10, 0x90004400
    dma.store x10, x6, x12, ch1

    LI    x6, 131072
    LI    x10, 0x90004800
    dma.store x10, x6, x12, ch2

    LI    x6, 196608
    LI    x10, 0x90004C00
    dma.store x10, x6, x12, ch3

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL

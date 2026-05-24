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
    DMA.CONFIG x5, 0
    LI    x12, 1024

# ============================================================================
# Prefill VMEM banks 4..5. Two tiles per bank become the source tiles for the
# mixed-burst DMA.STORE half.
# ============================================================================

    LI    x6, 262144
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x12, 0

    LI    x6, 262400
    LI    x10, 0x90000400
    DMA.LOAD  x6, x10, x12, 1

    LI    x6, 327680
    LI    x10, 0x90000800
    DMA.LOAD  x6, x10, x12, 2

    LI    x6, 327936
    LI    x10, 0x90000C00
    DMA.LOAD  x6, x10, x12, 3

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

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
    DMA.LOAD  x6, x10, x12, 0

    LI    x6, 262144
    LI    x10, 0x90003000
    DMA.STORE x10, x6, x12, 1

    LI    x6, 65536
    LI    x10, 0x90001400
    DMA.LOAD  x6, x10, x12, 2

    LI    x6, 262400
    LI    x10, 0x90003400
    DMA.STORE x10, x6, x12, 3

    LI    x6, 131072
    LI    x10, 0x90001800
    DMA.LOAD  x6, x10, x12, 4

    LI    x6, 327680
    LI    x10, 0x90003800
    DMA.STORE x10, x6, x12, 5

    LI    x6, 196608
    LI    x10, 0x90001C00
    DMA.LOAD  x6, x10, x12, 6

    LI    x6, 327936
    LI    x10, 0x90003C00
    DMA.STORE x10, x6, x12, 7

# ============================================================================
# LSU pressure loop. Stores deny DMA write grants on banks 0..3; loads deny DMA
# read grants on banks 4..5. Each scalar load uses DELAY 2 so the loop respects
# the longer fixed scalar-load latency.
# ============================================================================

lsu_stall_loop:
    SW    x13, x20, 0
    SW    x13, x21, 0
    SW    x13, x22, 0
    SW    x13, x23, 0

    LW    x14, x24, 0
    DELAY 2
    LW    x15, x25, 0
    DELAY 2
    LW    x16, x26, 0
    DELAY 2
    LW    x17, x27, 0
    DELAY 2

    ADDI  x18, x18, -1
    BNE   x18, x0, lsu_stall_loop
    NOP

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4
    DMA.WAIT  5
    DMA.WAIT  6
    DMA.WAIT  7

# Store the four DMA-loaded tiles back out so the LOAD half is checked.
    ADDI  x6, x0, 0
    LI    x10, 0x90004000
    DMA.STORE x10, x6, x12, 0

    LI    x6, 65536
    LI    x10, 0x90004400
    DMA.STORE x10, x6, x12, 1

    LI    x6, 131072
    LI    x10, 0x90004800
    DMA.STORE x10, x6, x12, 2

    LI    x6, 196608
    LI    x10, 0x90004C00
    DMA.STORE x10, x6, x12, 3

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

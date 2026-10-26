# dma_illegal_size_unaligned.S — a DMA transfer size that is not a multiple of 32 is illegal
# @TIMEOUT 2000
# @EXPECT_ILLEGAL
#
# x[rs2] must be a multiple of 32 from 32 to 4096. The 32-byte store is legal. The core stops at
# the 48-byte store (pc 28) with an illegal instruction; DBG0 stays 0.

    ADDI  x5, x0, 0
    dma.config x5, ch0
    ADDI  x1, x0, 0
    LI    x2, 0x80000000
    ADDI  x3, x0, 32
    dma.store x2, x1, x3, ch0
    ADDI  x3, x0, 48
    dma.store x2, x1, x3, ch1
    dma.wait ch0
    ADDI  x4, x0, 1
    CSRRW x0, 0x800, x4
    ECALL

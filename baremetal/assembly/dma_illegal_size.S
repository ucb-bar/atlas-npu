# dma_illegal_size.S — a DMA transfer of more than 4096 bytes is illegal
# @TIMEOUT 2000
# @EXPECT_ILLEGAL
#
# x[rs2] must be a multiple of 32 from 32 to 4096. The 4096-byte load is legal. The core stops at
# the 4128-byte load (pc 28) with an illegal instruction; DBG0 stays 0.

    ADDI  x5, x0, 0
    dma.config x5, ch0
    ADDI  x1, x0, 0
    LI    x2, 0x80000000
    LI    x3, 4096
    dma.load x1, x2, x3, ch0
    ADDI  x3, x3, 32
    dma.load x1, x2, x3, ch1
    dma.wait ch0
    ADDI  x4, x0, 1
    CSRRW x0, 0x800, x4
    ECALL

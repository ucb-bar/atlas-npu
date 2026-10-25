# csr_illegal_unmapped.S — an access to an unmapped CSR is illegal, also a read
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# The core stops at the CSRRS (pc 4) with an illegal instruction; DBG0 stays 0.

    ADDI  x1, x0, 1
    CSRRS x2, 0xC20, x0
    CSRRW x0, 0x800, x1
    ECALL

# csr_illegal_v1_c03.S — a read of CSR 0xC03 is illegal
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# 0xC03 was the illegal-instruction PC in ISA 1.0; ISA 2.0 does not map it.
# The core stops at the CSRRS (pc 4) with an illegal instruction; DBG0 stays 0.

    ADDI  x1, x0, 1
    CSRRS x2, 0xC03, x0
    CSRRW x0, 0x800, x1
    ECALL

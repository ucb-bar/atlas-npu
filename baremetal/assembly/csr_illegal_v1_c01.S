# csr_illegal_v1_c01.S — a read of CSR 0xC01 is illegal
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# 0xC01 was the instruction count in ISA 1.0; ISA 2.0 does not map it.
# The core stops at the CSRRS (pc 4) with an illegal instruction; DBG0 stays 0.

    ADDI  x1, x0, 1
    CSRRS x2, 0xC01, x0
    CSRRW x0, 0x800, x1
    ECALL

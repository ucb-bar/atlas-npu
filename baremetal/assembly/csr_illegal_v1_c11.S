# csr_illegal_v1_c11.S — a write to CSR 0xC11 is illegal
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# 0xC11 was DBG1 in ISA 1.0; ISA 2.0 does not map it.
# The core stops at the CSRRW (pc 4) with an illegal instruction; DBG0 stays 0.

    ADDI  x1, x0, 1
    CSRRW x0, 0xC11, x1
    CSRRW x0, 0x800, x1
    ECALL

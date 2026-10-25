# csr_illegal_ro_write.S — a write to a read-only CSR is illegal
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# A CSRRS with rs1 other than x0 writes, although x5 holds 0. On the ISA CSR (0xCC0) the core
# stops at it (word 1) with an illegal instruction; DBG0 stays 0.

    ADDI  x5, x0, 0
    CSRRS x2, 0xCC0, x5
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

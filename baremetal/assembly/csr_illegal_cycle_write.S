# csr_illegal_cycle_write.S — a write to cycle (0xC00) is illegal
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# cycle is read-only (ISA 2.0). The core stops at the CSRRW (pc 4) with an illegal instruction;
# DBG0 stays 0.

    ADDI  x1, x0, 1
    CSRRW x0, 0xC00, x1
    CSRRW x0, 0x800, x1
    ECALL

# jump_illegal_target.S — a taken jump to a target with bit 1 set is illegal
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# The not-taken BNE to byte offset 6 is legal. The core stops at the JALR (pc 8) to byte 6 with
# an illegal instruction; DBG0 stays 0.

    ADDI  x1, x0, 6
    BNE   x0, x0, 6
    JALR  x0, x1, 0
    ADDI  x2, x0, 1
    CSRRW x0, 0x800, x2
    ECALL

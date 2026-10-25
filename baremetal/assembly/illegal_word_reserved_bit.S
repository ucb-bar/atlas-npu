# illegal_word_reserved_bit.S — a word with a reserved bit set is illegal
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# vsquare.bf16 m4, m8 is 0x8C01020B; its vs2 field (bits 24:19) is reserved and must be 0.
# 0x8C09020B sets bit 19: the core stops at it (pc 4) with an illegal instruction; DBG0 stays 0.

    ADDI  x1, x0, 1
    .word 0x8C09020B
    CSRRW x0, 0x800, x1
    ECALL

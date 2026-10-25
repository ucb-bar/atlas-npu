# illegal_word_v1.S — an ISA 1.0 word is illegal
# @TIMEOUT 200
# @EXPECT_ILLEGAL
#
# 0x00610257 is the ISA 1.0 word of vadd.bf16 m4, m8, m12 (major opcode 0x57). No ISA 2.0
# instruction has opcode 0x57: the core stops at it (pc 4) with an illegal instruction; DBG0
# stays 0.

    ADDI  x1, x0, 1
    .word 0x00610257
    CSRRW x0, 0x800, x1
    ECALL

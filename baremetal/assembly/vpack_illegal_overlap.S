# vpack_illegal_overlap.S — vpack.bf16.fp8 with vd = vs2 + 1 is illegal
# @TIMEOUT 400
# @EXPECT_ILLEGAL
#
# vpack.bf16.fp8 m[vd], m[vs2] writes rows of m[vd] before it has read them as m[vs2+1], so
# vd = vs2 + 1 is illegal. vd = vs2 (in place) and vd = vs2 + 32 (another register of the
# physical bank of m[vs2+1]) are legal. The core stops at the third vpack (pc 12) with an illegal
# instruction; DBG0 stays 0.

    seli  e0, 127
    vpack.bf16.fp8 m2, m2, e0
    vpack.bf16.fp8 m35, m2, e0
    vpack.bf16.fp8 m3, m2, e0
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

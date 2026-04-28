# sab_a_to_zero.S — Kernel A, variant "JAL-to-zero"
#
# Used when A lives at IMEM[0x10000..) and B will be staged at IMEM[0..).
# Same busy loop + marker write as sab_a_to_high.S, but terminal JALR
# targets IMEM byte 0.
#
# @TIMEOUT 200000

_start:
    LI    x10, 5000
    ADDI  x11, x0, 0

a_busy:
    ADDI  x11, x11, 1
    BLT   x11, x10, a_busy
    NOP
    NOP

    LI    x2, 0xAAAAAAAA
    ADDI  x3, x0, 0
    SW    x2, x3, 0
    DELAY 1

    ADDI  x1, x0, 0                 # target = IMEM byte 0
    JALR  x0, x1, 0
    NOP
    NOP

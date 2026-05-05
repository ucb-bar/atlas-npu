# sab_a_ecall.S — Kernel A, variant "ECALL" (used by the optB-restart test).
#
# Same busy loop + VMEM marker as the JAL variants, but terminates with
# ECALL so the host can regain control, overwrite IMEM[0..) with B, then
# restart via execControl=1 (PC resets to 0 and runs B from scratch).
#
# @TIMEOUT 200000

_start:
    LI    x10, 5000
    ADDI  x11, x0, 0

a_busy:
    ADDI  x11, x11, 1
    BLT   x11, x10, a_busy
    NOP

    LI    x2, 0xAAAAAAAA
    ADDI  x3, x0, 0
    SW    x2, x3, 0
    DELAY 1

    ADDI  x1, x0, 1
    CSRW  x1, 0xC10                 # dbg0 = 1 (A done)
    ECALL

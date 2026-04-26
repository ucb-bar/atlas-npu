# sab_b.S — Kernel B for buffered dual-kernel tests.
#
# Runs after kernel A in all three dual-kernel variants. Reads the VMEM marker
# A left behind and signals PASS/FAIL via dbg0 so the host can tell which
# kernel actually executed.
#
#   VMEM[0] == 0xAAAAAAAA  → dbg0 = 2 (PASS: A's marker present, B ran)
#   VMEM[0] != 0xAAAAAAAA  → dbg0 = 0xBAD (FAIL: marker missing)
#
# @TIMEOUT 50000

_start:
    ADDI  x3, x0, 0
    LW    x4, x3, 0                 # got = VMEM[0]
    DELAY 1

    LI    x5, 0xAAAAAAAA             # expected marker
    BNE   x4, x5, b_fail
    NOP
    NOP

    ADDI  x1, x0, 2
    CSRW  x1, 0xC10                 # dbg0 = 2 PASS
    ECALL

b_fail:
    LI    x1, 0xBAD
    CSRW  x1, 0xC10                 # dbg0 = 0xBAD
    ECALL

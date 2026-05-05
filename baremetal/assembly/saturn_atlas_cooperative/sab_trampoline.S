# sab_trampoline.S — Boot trampoline for buffered dual-kernel tests.
#
# Two-instruction jumper placed at IMEM[0] when kernel A is staged at
# IMEM[0x10000..). Atlas starts at PC=0, runs the trampoline, and jumps
# into A.
#
# Used by the lower-first and optB-restart variants prior to staging B.

_start:
    LI    x1, 0x4000                # word index of IMEM byte 0x10000
    JALR  x0, x1, 0
    NOP

# sas_carousel_d.S — kernel D of the 4-kernel IMEM carousel (T3).
#
# Lives at IMEM word 0x4000..0x4000+LEN_D-1, painted by Saturn after
# kernel B JALR'd to low. Final kernel in the carousel. Sequence:
#   1. Brief busy-spin (no further host repaint depends on this delay,
#      but a small hold helps post-mortem stability).
#   2. Write VMEM[12] = 0xDDDDDDDD (marker_D).
#   3. CSRW dbg0, 4 — host's halt-with-dbg0 check expects this value.
#   4. ECALL — terminates Atlas execution.

_start:
    LI    x10, 1000
    ADDI  x11, x0, 0

d_busy:
    ADDI  x11, x11, 1
    BLT   x11, x10, d_busy
    NOP
    NOP

    LI    x2, 0xDDDDDDDD            # marker_D
    ADDI  x3, x0, 12                # VMEM byte 12
    SW    x2, x3, 0
    DELAY 1

    ADDI  x1, x0, 4                 # dbg0 = 4
    CSRW  x1, 0xC10
    ECALL

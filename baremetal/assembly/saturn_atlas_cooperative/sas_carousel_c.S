# sas_carousel_c.S — kernel C of the 4-kernel IMEM carousel (T3).
#
# Lives at IMEM word 0..LEN_C-1, painted by Saturn after kernel A
# JALR'd to high. Sequence:
#   1. Busy-spin so the host has time to observe marker_C AND so D's
#      stage into the high half completes before C JALRs there.
#   2. Write VMEM[8] = 0xCCCCCCCC (marker_C).
#   3. JALR to IMEM word 0x4000 (high-half entry) — runs kernel D,
#      which the host has staged over B while we were running here.

_start:
    LI    x10, 5000
    ADDI  x11, x0, 0

c_busy:
    ADDI  x11, x11, 1
    BLT   x11, x10, c_busy
    NOP
    NOP

    LI    x2, 0xCCCCCCCC            # marker_C
    ADDI  x3, x0, 8                 # VMEM byte 8
    SW    x2, x3, 0
    DELAY 1

    LI    x1, 0x4000                # IMEM byte 0x10000 (word 0x4000)
    JALR  x0, x1, 0
    NOP
    NOP

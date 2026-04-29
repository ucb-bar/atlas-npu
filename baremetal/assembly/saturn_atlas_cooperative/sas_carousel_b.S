# sas_carousel_b.S — kernel B of the 4-kernel IMEM carousel (T3).
#
# Lives at IMEM word 0x4000..0x4000+LEN_B-1 (high half). Runs after A
# JALRs into it. Sequence:
#   1. Busy-spin so the host has time to observe marker_B and stage D
#      into the high half before B finishes.
#   2. Write VMEM[4] = 0xBBBBBBBB (marker_B).
#   3. JALR to IMEM word 0 (low half) — entry of kernel C, which the
#      host has staged over A while we were running here.

_start:
    LI    x10, 5000
    ADDI  x11, x0, 0

b_busy:
    ADDI  x11, x11, 1
    BLT   x11, x10, b_busy
    NOP
    NOP

    LI    x2, 0xBBBBBBBB            # marker_B
    ADDI  x3, x0, 4                 # VMEM byte 4
    SW    x2, x3, 0
    DELAY 1

    ADDI  x1, x0, 0                 # IMEM word 0 (low-half entry)
    JALR  x0, x1, 0
    NOP
    NOP

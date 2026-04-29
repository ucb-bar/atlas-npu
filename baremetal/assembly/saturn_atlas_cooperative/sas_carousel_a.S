# sas_carousel_a.S — kernel A of the 4-kernel IMEM carousel (T3).
#
# Lives at IMEM word 0..LEN_A-1 (low half). Initial body of the
# carousel. Sequence:
#   1. Busy-spin so the host has time to observe the marker AND so any
#      in-flight A fetches have drained before C's stage starts.
#   2. Write VMEM[0] = 0xAAAAAAAA (marker_A).
#   3. JALR to IMEM word 0x4000 (= byte 0x10000), i.e. high-half entry
#      where kernel B is staged.
#
# After A's JALR, host overwrites low IMEM with kernel C while Atlas is
# busy-spinning in B's high half.

_start:
    LI    x10, 5000                # busy loop iterations (~20 K Atlas cycles)
    ADDI  x11, x0, 0

a_busy:
    ADDI  x11, x11, 1
    BLT   x11, x10, a_busy
    NOP
    NOP

    LI    x2, 0xAAAAAAAA            # marker_A
    ADDI  x3, x0, 0                 # VMEM byte 0
    SW    x2, x3, 0
    DELAY 1

    LI    x1, 0x4000                # IMEM byte 0x10000 (word 0x4000)
    JALR  x0, x1, 0
    NOP
    NOP

# sab_a_to_high.S — Kernel A, variant "JAL-to-high"
#
# Used when A lives at IMEM[0..0x10000) and control should flow into B
# at IMEM[0x10000..) WITHOUT halting. A does a ~20k-cycle busy loop so
# the host has time to stage B concurrently, writes a marker to VMEM[0],
# then absolute-jumps to byte 0x10000.
#
# @TIMEOUT 200000

_start:
    LI    x10, 5000                # busy loop iterations
    ADDI  x11, x0, 0               # i

a_busy:
    ADDI  x11, x11, 1
    BLT   x11, x10, a_busy
    NOP

    LI    x2, 0xAAAAAAAA            # marker
    ADDI  x3, x0, 0                 # VMEM byte 0
    SW    x2, x3, 0
    DELAY 1

    LI    x1, 0x4000                # word index of IMEM byte 0x10000
    JALR  x0, x1, 0
    NOP

# Position-independent dispatcher test kernel. Uses only x1-x4; preserves ABI.
# The long loop gives the host time to stage and queue the other slot.
    LI x1, 20000
    ADDI x2, x0, 0
busy:
    ADDI x2, x2, 1
    BLT x2, x1, busy
    NOP
    LW x4, x0, 0
    DELAY 2
    ADDI x4, x4, 1
    SW x4, x0, 0
    DELAY 2
    JALR x0, x31, 1
    NOP

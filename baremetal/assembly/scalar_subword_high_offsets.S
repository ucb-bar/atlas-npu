# scalar_subword_high_offsets.S - high-offset scalar byte masks/sign extension
# @TIMEOUT 10000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_scalar_subword_high_offsets.py
#
# mem_scalar.S covers sub-word loads/stores near byte 0. This test targets the
# upper half of a 32-byte VMEM line, including byte 31 and halfwords ending at
# byte 29/27, to toggle different mask bits and sign-extension paths.

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 32

    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x28, x28, 1

    ADDI  x8, x0, 0

# Byte 31: signed and unsigned byte loads from the top byte of the line.
    LI    x10, 0x81
    SB    x10, x8, 31    
    LBU   x9, x8, 31
    DELAY 2
    LI    x30, 0x81
    BNE   x9, x30, fail
    NOP
    LB    x9, x8, 31
    DELAY 2
    LI    x30, 0xFFFFFF81
    BNE   x9, x30, fail
    NOP
    ADDI  x28, x28, 1

# Halfword at byte 28: positive sign bit.
    LI    x11, 0x7F80
    SH    x11, x8, 28
    LH    x9, x8, 28
    DELAY 2
    LI    x30, 0x7F80
    BNE   x9, x30, fail
    NOP
    ADDI  x28, x28, 1

# Halfword at byte 26: negative sign bit.
    LI    x12, 0x8001
    SH    x12, x8, 26    
    LH    x9, x8, 26
    DELAY 2
    LI    x30, 0xFFFF8001
    BNE   x9, x30, fail
    NOP
    LHU   x9, x8, 26
    DELAY 2
    LI    x30, 0x8001
    BNE   x9, x30, fail
    NOP
    ADDI  x28, x28, 1

# Word at byte 20: upper-half word lane.
    LI    x13, 0xA1B2C3D4
    SW    x13, x8, 20
    LW    x9, x8, 20
    DELAY 2
    LI    x30, 0xA1B2C3D4
    BNE   x9, x30, fail
    NOP
    ADDI  x28, x28, 1

    ADDI  x6, x0, 0
    LI    x3, 0x90000400
    DMA.STORE x3, x6, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

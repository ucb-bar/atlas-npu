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
    dma.config x5, ch0
    LI    x2, 32

    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x28, x28, 1

    ADDI  x8, x0, 0

# Byte 31: signed and unsigned byte loads from the top byte of the line.
    LI    x10, 0x81
    sb x10, 31(x8)
    lbu x9, 31(x8)
    DELAY 2
    LI    x30, 0x81
    BNE   x9, x30, fail
    lb x9, 31(x8)
    DELAY 2
    LI    x30, 0xFFFFFF81
    BNE   x9, x30, fail
    ADDI  x28, x28, 1

# Halfword at byte 28: positive sign bit.
    LI    x11, 0x7F80
    sh x11, 28(x8)
    lh x9, 28(x8)
    DELAY 2
    LI    x30, 0x7F80
    BNE   x9, x30, fail
    ADDI  x28, x28, 1

# Halfword at byte 26: negative sign bit.
    LI    x12, 0x8001
    sh x12, 26(x8)
    lh x9, 26(x8)
    DELAY 2
    LI    x30, 0xFFFF8001
    BNE   x9, x30, fail
    lhu x9, 26(x8)
    DELAY 2
    LI    x30, 0x8001
    BNE   x9, x30, fail
    ADDI  x28, x28, 1

# Word at byte 20: upper-half word lane.
    LI    x13, 0xA1B2C3D4
    sw x13, 20(x8)
    lw x9, 20(x8)
    DELAY 2
    LI    x30, 0xA1B2C3D4
    BNE   x9, x30, fail
    ADDI  x28, x28, 1

    ADDI  x6, x0, 0
    LI    x3, 0x90000400
    dma.store x3, x6, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

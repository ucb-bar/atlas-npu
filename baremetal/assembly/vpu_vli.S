# vpu_vli.S — VPU vector load immediate operations
# Tests: vli.all, vli.row, vli.col, vli.one
# @TIMEOUT 20000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_vli.py
#
# VLI instructions write an immediate BF16 constant into a tensor
# register with no source read. Each tensor register is one 32x16 BF16
# tile (1024 B), so each VLI must only modify m[vd].
#
#   vli.all vd, imm : m[vd][:, :] = imm   (all elements)
#   vli.row vd, imm : m[vd][0, :] = imm   (first row)
#   vli.col vd, imm : m[vd][:, 0] = imm   (first column)
#   vli.one vd, imm : m[vd][0, 0] = imm   (single element)
#
# BF16 immediates used:
#   0x3F80 = 1.0     0x4000 = 2.0     0x4040 = 3.0
#   0x4080 = 4.0     0x40A0 = 5.0     (neighbor-register sentinel)
#
# DRAM layout:
#   [0x0000..0x03FF]  vli.all  m4 output        (1024 B)
#   [0x0400..0x07FF]  vli.all  m5 sentinel      (1024 B)
#   [0x0800..0x0BFF]  vli.row  m4 output        (1024 B)
#   [0x0C00..0x0FFF]  vli.row  m5 sentinel      (1024 B)
#   [0x1000..0x13FF]  vli.col  m4 output        (1024 B)
#   [0x1400..0x17FF]  vli.col  m5 sentinel      (1024 B)
#   [0x1800..0x1BFF]  vli.one  m4 output        (1024 B)
#   [0x1C00..0x1FFF]  vli.one  m5 sentinel      (1024 B)

    ADDI  x28, x0, 1
    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ================================================================
    # Test 1: vli.all  m4, 0x3F80    (1.0 to all elements)
    # ================================================================
    VLI.ALL 5, 0x40A0
    DELAY 150

    VLI.ALL 4, 0x3F80
    DELAY 150

    LI    x8, 0
    VSTORE 4, x8, 0
    DELAY 40
    LI    x8, 256
    VSTORE 5, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x0000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 256
    LI    x3, 0x0400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 2: vli.row  m4, 0x4000    (2.0 to first row)
    # ================================================================
    # First zero out m4, then reload the untouched-neighbor sentinel in m5.
    VLI.ALL 4, 0x0000
    DELAY 150

    VLI.ALL 5, 0x40A0
    DELAY 150

    VLI.ROW 4, 0x4000
    DELAY 150

    LI    x8, 0
    VSTORE 4, x8, 0
    DELAY 40
    LI    x8, 256
    VSTORE 5, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 256
    LI    x3, 0x0C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 3: vli.col  m4, 0x4040    (3.0 to first column)
    # ================================================================
    VLI.ALL 4, 0x0000
    DELAY 150

    VLI.ALL 5, 0x40A0
    DELAY 150

    VLI.COL 4, 0x4040
    DELAY 150

    LI    x8, 0
    VSTORE 4, x8, 0
    DELAY 40
    LI    x8, 256
    VSTORE 5, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x1000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 256
    LI    x3, 0x1400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 4: vli.one  m4, 0x4080    (4.0 to element [0,0])
    # ================================================================
    VLI.ALL 4, 0x0000
    DELAY 150

    VLI.ALL 5, 0x40A0
    DELAY 150

    VLI.ONE 4, 0x4080
    DELAY 150

    LI    x8, 0
    VSTORE 4, x8, 0
    DELAY 40
    LI    x8, 256
    VSTORE 5, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x1800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 256
    LI    x3, 0x1C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

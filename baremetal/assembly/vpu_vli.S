# vpu_vli.S — VPU vector load immediate operations
# Tests: vli.all, vli.row, vli.col, vli.one
# @TIMEOUT 20000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_vli.py
#
# VLI instructions write an immediate BF16 constant into a tensor
# register with no source read. Each physical tensor register is one
# 32x16 BF16 tile (1024 B); a full 32x32 BF16 tensor therefore spans an
# even/odd register pair.
#
#   vli.all vd, imm : m[vd][:, :]   = imm and m[vd+1][:, :]   = imm
#   vli.row vd, imm : m[vd][0, :]   = imm and m[vd+1][0, :]   = imm
#   vli.col vd, imm : m[vd][:, 0]   = imm   (single physical register)
#   vli.one vd, imm : m[vd][0, 0]   = imm   (single physical register)
#
# BF16 immediates used:
#   0x3F80 = 1.0     0x4000 = 2.0     0x4040 = 3.0     0x4080 = 4.0
#
# DRAM layout:
#   [0x0000..0x03FF]  vli.all  m4 output        (1024 B)
#   [0x0400..0x07FF]  vli.all  m5 output        (1024 B)
#   [0x0800..0x0BFF]  vli.row  m4 output        (1024 B)
#   [0x0C00..0x0FFF]  vli.row  m5 output        (1024 B)
#   [0x1000..0x13FF]  vli.col  m4 untouched     (1024 B)
#   [0x1400..0x17FF]  vli.col  m5 output        (1024 B)
#   [0x1800..0x1BFF]  vli.one  m4 untouched     (1024 B)
#   [0x1C00..0x1FFF]  vli.one  m5 output        (1024 B)

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ================================================================
    # Test 1: vli.all  m4, 0x3F80    (1.0 to the full BF16 register pair)
    # ================================================================
    VLI.ALL 4, 0x3F80
    DELAY 63

    LI    x8, 0
    VSTORE 4, x8, 0
    DELAY 33
    LI    x8, 256
    VSTORE 5, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x90000000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 256
    LI    x3, 0x90000400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 2: vli.row  m4, 0x4000    (2.0 to row 0 of the full BF16 pair)
    # ================================================================
    VLI.ALL 4, 0x0000
    DELAY 63

    VLI.ROW 4, 0x4000
    DELAY 63

    LI    x8, 0
    VSTORE 4, x8, 0
    DELAY 33
    LI    x8, 256
    VSTORE 5, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x90000800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 256
    LI    x3, 0x90000C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 3: vli.col  m5, 0x4040    (3.0 to first column of one register)
    # ================================================================
    VLI.ALL 4, 0x0000
    DELAY 63

    VLI.COL 5, 0x4040
    DELAY 31

    LI    x8, 0
    VSTORE 4, x8, 0
    DELAY 33
    LI    x8, 256
    VSTORE 5, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x90001000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 256
    LI    x3, 0x90001400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 4: vli.one  m5, 0x4080    (4.0 to one element in one register)
    # ================================================================
    VLI.ALL 4, 0x0000
    DELAY 63

    VLI.ONE 5, 0x4080
    DELAY 31

    LI    x8, 0
    VSTORE 4, x8, 0
    DELAY 33
    LI    x8, 256
    VSTORE 5, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x90001800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 256
    LI    x3, 0x90001C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

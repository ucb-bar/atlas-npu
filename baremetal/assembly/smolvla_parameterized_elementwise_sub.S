# smolvla_parameterized_elementwise_sub.S — tiled C = A - B for 64x64 BF16
# Port of jrmills20 parameterized_elementwise_sub.py, with Atlas bank-split
# tile layout (matching stock smolvla_elementwise_sub) and a scalar tile loop.
# @TIMEOUT 1000000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_elementwise_sub.py
#
# Shape: M=128, N=128 → M_tiles=4, N_tiles=4, total_tiles=16. Tile-major order.
# DRAM layout matches elementwise_add / _mul siblings.

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x21, 0x8000
    LI    x22, 0x10000
    ADDI  x11, x0, 16

    ADDI  x10, x0, 0

tile_loop:
    SLLI  x20, x10, 11

    ADDI  x1, x20, 0
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x1, x20, 1024
    LI    x6, 256
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    ADD   x1, x21, x20
    LI    x7, 512
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    ADDI  x1, x1, 1024
    LI    x7, 768
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0

    LI    x7, 512
    VLOAD 2, x7, 0
    DELAY 40
    VLOAD 3, x7, 8
    DELAY 40

    # (m4, m5) = (m0, m1) - (m2, m3)
    VSUB.BF16 4, 0, 2
    DELAY 150

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40

    ADD   x3, x22, x20
    LI    x8, 1024
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    ADDI  x3, x3, 1024
    LI    x8, 1280
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    ADDI  x10, x10, 1
    BNE   x10, x11, tile_loop
    NOP
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x21, 0x8000
    LI    x22, 0x10000
    ADDI  x11, x0, 16

    ADDI  x10, x0, 0

    LUI   x15, 0x90000         # FPGA DRAM offset 0x9000_0000
tile_loop:
    SLLI  x20, x10, 11
    ADD   x20, x20, x15        # x20 += FPGA DRAM base

    ADDI  x1, x20, 0
    ADDI  x6, x0, 0
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x1, x20, 1024
    LI    x6, 256
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    ADD   x1, x21, x20
    LI    x7, 512
    dma.load x7, x1, x2, ch0
    dma.wait ch0
    ADDI  x1, x1, 1024
    LI    x7, 768
    dma.load x7, x1, x2, ch0
    dma.wait ch0

    LI    x7, 512
    vload m2, 0(x7)
    DELAY 33
    vload m3, 8(x7)
    DELAY 33

    # (m4, m5) = (m0, m1) - (m2, m3)
    vsub.bf16 m4, m0, m2
    DELAY 64

    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33

    ADD   x3, x22, x20
    LI    x8, 1024
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    ADDI  x3, x3, 1024
    LI    x8, 1280
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    ADDI  x10, x10, 1
    BNE   x10, x11, tile_loop
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

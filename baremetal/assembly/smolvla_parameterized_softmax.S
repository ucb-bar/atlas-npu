# smolvla_parameterized_softmax.S — tiled row-wise stable softmax, 128x32 BF16
# Port of jrmills20 parameterized_softmax.py. N fixed at 32 (Jeremy's
# constraint); M scales via tile loop. Each 32-row group is processed
# independently — all reductions local to one bank-split group.
#
#   Y = exp(X - rowmax(X)) / rowsum(exp(...))
#
# @TIMEOUT 2500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_softmax.py
#
# Shape: M=512, N=32 → 16 groups.
# DRAM layout:
#   [0x00000..0x07FFF]  Input X   (16 groups x 2048B BF16)
#   [0x08000..0x0FFFF]  Output Y  (16 groups x 2048B BF16)
#
# Mreg allocation (per group, all local):
#   m0,m1   = X                    m2,m3   = rowmax(X)
#   m4,m5   = X - rowmax           m6,m7   = exp(...)
#   m8,m9   = rowsum(exp)          m10,m11 = 1/rowsum
#   m12,m13 = Y = exp * inv_rowsum

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x22, 0x8000

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

    LUI   x15, 0x90000         # FPGA DRAM offset 0x9000_0000
tile_loop:
    SLLI  x20, x10, 11
    ADD   x20, x20, x15        # x20 += FPGA DRAM base

    # ── Load X ──
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

    # ── Softmax chain ──
    vredmax.row.bf16 m2, m0
    DELAY 32
    vsub.bf16 m4, m0, m2
    DELAY 64
    vexp.bf16 m6, m4
    DELAY 64
    vredsum.row.bf16 m8, m6
    DELAY 37
    vrecip.bf16 m10, m8
    DELAY 64
    vmul.bf16 m12, m6, m10
    DELAY 64

    LI    x8, 512
    vstore m12, 0(x8)
    DELAY 33
    vstore m13, 8(x8)
    DELAY 33

    # ── DMA out ──
    ADD   x3, x22, x20
    LI    x8, 512
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    ADDI  x3, x3, 1024
    LI    x8, 768
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    ADDI  x10, x10, 1
    BNE   x10, x11, tile_loop
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

# smolvla_parameterized_softmax.S — tiled row-wise stable softmax, 128x32 BF16
# Port of jrmills20 parameterized_softmax.py. N fixed at 32 (Jeremy's
# constraint); M scales via tile loop. Each 32-row group is processed
# independently — all reductions local to one bank-split group.
#
#   Y = exp(X - rowmax(X)) / rowsum(exp(...))
#
# @TIMEOUT 2500000
# @DRAM_BASE 0x00000001
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

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x22, 0x8000

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

tile_loop:
    SLLI  x20, x10, 11

    # ── Load X ──
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
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ── Softmax chain ──
    VREDMAX.ROW.BF16 2, 0
    DELAY 32
    VSUB.BF16 4, 0, 2
    DELAY 64
    VEXP 6, 4
    DELAY 64
    VREDSUM.ROW.BF16 8, 6
    DELAY 37
    VRECIP.BF16 10, 8
    DELAY 64
    VMUL.BF16 12, 6, 10
    DELAY 64

    LI    x8, 512
    VSTORE 12, x8, 0
    DELAY 33
    VSTORE 13, x8, 8
    DELAY 33

    # ── DMA out ──
    ADD   x3, x22, x20
    LI    x8, 512
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    ADDI  x3, x3, 1024
    LI    x8, 768
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

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

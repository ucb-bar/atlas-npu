# smolvla_parameterized_gelu_tanh.S — tiled GELU (tanh approx), 64x64 BF16
# Port of jrmills20 parameterized_gelu_tanh.py:
#   y = 0.5 * X * (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X^3)))
# VLI.ALL constants hoisted out of the tile loop.
# @TIMEOUT 2500000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_parameterized_gelu_tanh.py
#
# Shape: M=128, N=128 → 16 tiles.
# DRAM layout:
#   [0x00000..0x07FFF]  Input X   (16 tiles x 2048B BF16)
#   [0x08000..0x0FFFF]  Output Y  (16 tiles x 2048B BF16)
#
# Mreg allocation (m2..m9 invariant across tiles; m10..m15 rotate):
#   m0,m1   = X                     m2,m3   = 0.044715
#   m4,m5   = sqrt(2/pi)            m6,m7   = 1.0
#   m8,m9   = 0.5                   m10..m15 rotate across intermediates

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x22, 0x8000

    # ── Invariant constants ──
    VLI.ALL 2, 0x3D37          # 0.044715
    DELAY 63
    VLI.ALL 4, 0x3F4C          # sqrt(2/pi)
    DELAY 63
    VLI.ALL 6, 0x3F80          # 1.0
    DELAY 63
    VLI.ALL 8, 0x3F00          # 0.5
    DELAY 63

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

    # ── GELU tanh chain ──
    VCUBE.BF16 10, 0           # X^3
    DELAY 64
    VMUL.BF16 12, 2, 10        # 0.044715 * X^3
    DELAY 64
    VADD.BF16 10, 0, 12        # X + 0.044715 * X^3
    DELAY 64
    VMUL.BF16 12, 4, 10        # sqrt(2/pi) * (...)
    DELAY 64
    VTANH 10, 12               # tanh(...)
    DELAY 64
    VADD.BF16 12, 6, 10        # 1 + tanh(...)
    DELAY 64
    VMUL.BF16 14, 8, 12        # 0.5 * (1 + tanh(...))
    DELAY 64
    VMUL.BF16 10, 0, 14        # GELU(X)
    DELAY 64

    LI    x8, 512
    VSTORE 10, x8, 0
    DELAY 33
    VSTORE 11, x8, 8
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

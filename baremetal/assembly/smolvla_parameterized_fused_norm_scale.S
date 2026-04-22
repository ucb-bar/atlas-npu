# smolvla_parameterized_fused_norm_scale.S — tiled out = matrix * rsqrt(variance), 64x64
# Port of jrmills20 parameterized_fused_norm_scale.py. rsqrt synthesized as
# VRECIP.BF16(VSQRT(...)) (atlas has no VRSQRT opcode).
# @TIMEOUT 1500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_fused_norm_scale.py
#
# Shape: M=128, N=128 → 16 tiles.
# DRAM layout:
#   [0x00000..0x07FFF]  variance  (16 tiles x 2048B BF16)
#   [0x08000..0x0FFFF]  matrix    (16 tiles x 2048B BF16)
#   [0x10000..0x17FFF]  output    (16 tiles x 2048B BF16)
#
# Mreg allocation (per tile):
#   m0,m1 = variance    m2,m3 = matrix
#   m4,m5 = sqrt(var)   m6,m7 = rsqrt(var) = 1/sqrt(var)
#   m8,m9 = output = matrix * rsqrt(var)

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x21, 0x8000          # matrix DRAM base
    LI    x22, 0x10000         # output DRAM base

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

tile_loop:
    SLLI  x20, x10, 11

    # ── Load variance tile → (m0, m1) ──
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

    # ── Load matrix tile → (m2, m3) ──
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

    # ── rsqrt(variance) = 1 / sqrt(variance) ──
    VSQRT 4, 0
    DELAY 150
    VRECIP.BF16 6, 4
    DELAY 150

    # ── matrix * rsqrt(variance) ──
    VMUL.BF16 8, 2, 6
    DELAY 150

    LI    x8, 1024
    VSTORE 8, x8, 0
    DELAY 40
    VSTORE 9, x8, 8
    DELAY 40

    # ── DMA out ──
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

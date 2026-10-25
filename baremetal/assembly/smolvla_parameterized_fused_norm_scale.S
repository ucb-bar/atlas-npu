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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x21, 0x8000          # matrix DRAM base
    LI    x22, 0x10000         # output DRAM base

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

    LUI   x15, 0x90000         # FPGA DRAM offset 0x9000_0000
tile_loop:
    SLLI  x20, x10, 11
    ADD   x20, x20, x15        # x20 += FPGA DRAM base

    # ── Load variance tile → (m0, m1) ──
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

    # ── Load matrix tile → (m2, m3) ──
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

    # ── rsqrt(variance) = 1 / sqrt(variance) ──
    vsqrt.bf16 m4, m0
    DELAY 64
    vrecip.bf16 m6, m4
    DELAY 64

    # ── matrix * rsqrt(variance) ──
    vmul.bf16 m8, m2, m6
    DELAY 64

    LI    x8, 1024
    vstore m8, 0(x8)
    DELAY 33
    vstore m9, 8(x8)
    DELAY 33

    # ── DMA out ──
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

# smolvla_fused_norm_scale.S - port of npu-model smolvla_fused_norm_scale
# Tests: out[i,j] = matrix[i,j] * rsqrt(variance[i,j]) on seeded BF16 32x32 tiles
#   rsqrt = VRECIP.BF16(VSQRT(...))   (atlas has no VRSQRT)
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_fused_norm_scale.py
#
# DRAM layout (set by gen_smolvla_fused_norm_scale.py):
#   [0x0000..0x07FF]  variance (BF16 [32,32], 2 banks = 2048 B)
#   [0x0800..0x0FFF]  matrix   (BF16 [32,32], 2 banks)
#   [0x1000..0x17FF]  output   (BF16 [32,32], 2 banks)
#
# Mreg allocation:
#   m0,m1   = variance
#   m2,m3   = matrix
#   m4,m5   = sqrt(variance)
#   m6,m7   = rsqrt(variance) = 1 / sqrt(variance)
#   m8,m9   = output = matrix * rsqrt(variance)

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024              # bank size in bytes

    # ── Load variance → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 256
    LI    x1, 0x90000400
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ── Load matrix → VMEM[512..1023] → (m2, m3) ──
    LI    x7, 512
    LI    x1, 0x90000800
    dma.load x7, x1, x2, ch0
    dma.wait ch0
    LI    x7, 768
    LI    x1, 0x90000C00
    dma.load x7, x1, x2, ch0
    dma.wait ch0

    LI    x7, 512
    vload m2, 0(x7)
    DELAY 33
    vload m3, 8(x7)
    DELAY 33

    # ── (m4, m5) = sqrt(variance) ──
    vsqrt.bf16 m4, m0
    DELAY 64

    # ── (m6, m7) = 1 / sqrt(variance) = rsqrt(variance) ──
    vrecip.bf16 m6, m4
    DELAY 64

    # ── (m8, m9) = matrix * rsqrt(variance) ──
    vmul.bf16 m8, m2, m6
    DELAY 64

    # ── Store output → VMEM[1024..1535] → DRAM[0x1000..0x17FF] ──
    LI    x8, 1024
    vstore m8, 0(x8)
    DELAY 33
    vstore m9, 8(x8)
    DELAY 33

    LI    x8, 1024
    LI    x3, 0x90001000
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90001400
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

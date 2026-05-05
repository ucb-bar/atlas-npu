# smolvla_fused_norm_scale.S - port of npu-model smolvla_fused_norm_scale
# Tests: out[i,j] = matrix[i,j] * rsqrt(variance[i,j]) on seeded BF16 32x32 tiles
#   rsqrt = VRECIP.BF16(VSQRT(...))   (atlas has no VRSQRT)
#
# @TIMEOUT 30000
# @DRAM_BASE 0x00000001
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

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024              # bank size in bytes

    # ── Load variance → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ── Load matrix → VMEM[512..1023] → (m2, m3) ──
    LI    x7, 512
    LI    x1, 2048
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 768
    LI    x1, 3072
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0

    LI    x7, 512
    VLOAD 2, x7, 0
    DELAY 40
    VLOAD 3, x7, 8
    DELAY 40

    # ── (m4, m5) = sqrt(variance) ──
    VSQRT 4, 0
    DELAY 150

    # ── (m6, m7) = 1 / sqrt(variance) = rsqrt(variance) ──
    VRECIP.BF16 6, 4
    DELAY 150

    # ── (m8, m9) = matrix * rsqrt(variance) ──
    VMUL.BF16 8, 2, 6
    DELAY 150

    # ── Store output → VMEM[1024..1535] → DRAM[0x1000..0x17FF] ──
    LI    x8, 1024
    VSTORE 8, x8, 0
    DELAY 40
    VSTORE 9, x8, 8
    DELAY 40

    LI    x8, 1024
    LI    x3, 0x1000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x1400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

# smolvla_gelu_tanh.S — port of npu-model smolvla_gelu_tanh
# Tests: GELU (tanh approx) on seeded random BF16 32x32 tile
#   y = 0.5 * X * (1 + tanh(sqrt(2/pi) * (X + 0.044715 * X^3)))
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_gelu_tanh.py
#
# Differs from npu-model's program: uses VLI.ALL constants instead of
# DMA-preloaded constant tiles. All vector ops are pair-ops on (m,m+1).
#
# DRAM layout:
#   [0x0000..0x03FF]  Input X bank 0    (1024 B)
#   [0x0400..0x07FF]  Input X bank 1    (1024 B)
#   [0x0800..0x0FFF]  Output            (2048 B, 2 banks)
#
# Mreg allocation (16 total; m10..m15 reused across stages):
#   m0,m1   = X                     (kept across pipeline)
#   m2,m3   = 0.044715              m4,m5   = sqrt(2/pi)
#   m6,m7   = 1.0                   m8,m9   = 0.5
#   m10..m15 rotate across intermediates
# Final GELU result lands in m10,m11 → VMEM[512..1023] → DRAM[0x0800..0x0FFF]

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input X → VMEM[0..511] ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 0x90000400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ── Constants ──
    VLI.ALL 2, 0x3D37          # 0.044715
    DELAY 63
    VLI.ALL 4, 0x3F4C          # sqrt(2/pi)
    DELAY 63
    VLI.ALL 6, 0x3F80          # 1.0
    DELAY 63
    VLI.ALL 8, 0x3F00          # 0.5
    DELAY 63

    # ── m10,m11 = X^3 ──
    VCUBE.BF16 10, 0
    DELAY 64

    # ── m12,m13 = 0.044715 * X^3 ──
    VMUL.BF16 12, 2, 10
    DELAY 64

    # ── m10,m11 = X + 0.044715 * X^3 ──
    VADD.BF16 10, 0, 12
    DELAY 64

    # ── m12,m13 = sqrt(2/pi) * (X + 0.044715 * X^3) ──
    VMUL.BF16 12, 4, 10
    DELAY 64

    # ── m10,m11 = tanh(...) ──
    VTANH 10, 12
    DELAY 64

    # ── m12,m13 = 1 + tanh(...) ──
    VADD.BF16 12, 6, 10
    DELAY 64

    # ── m14,m15 = 0.5 * (1 + tanh(...)) ──
    VMUL.BF16 14, 8, 12
    DELAY 64

    # ── m10,m11 = X * 0.5*(1+tanh(...)) = GELU(X) ──
    VMUL.BF16 10, 0, 14
    DELAY 64

    LI    x8, 512
    VSTORE 10, x8, 0
    DELAY 33
    VSTORE 11, x8, 8
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90000800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x90000C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

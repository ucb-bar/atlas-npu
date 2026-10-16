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
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load Input X → VMEM[0..511] ──
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

    # ── Constants ──
    vli.all m2, 0x3D37          # 0.044715
    DELAY 63
    vli.all m4, 0x3F4C          # sqrt(2/pi)
    DELAY 63
    vli.all m6, 0x3F80          # 1.0
    DELAY 63
    vli.all m8, 0x3F00          # 0.5
    DELAY 63

    # ── m10,m11 = X^3 ──
    vcube.bf16 m10, m0
    DELAY 64

    # ── m12,m13 = 0.044715 * X^3 ──
    vmul.bf16 m12, m2, m10
    DELAY 64

    # ── m10,m11 = X + 0.044715 * X^3 ──
    vadd.bf16 m10, m0, m12
    DELAY 64

    # ── m12,m13 = sqrt(2/pi) * (X + 0.044715 * X^3) ──
    vmul.bf16 m12, m4, m10
    DELAY 64

    # ── m10,m11 = tanh(...) ──
    vtanh.bf16 m10, m12
    DELAY 64

    # ── m12,m13 = 1 + tanh(...) ──
    vadd.bf16 m12, m6, m10
    DELAY 64

    # ── m14,m15 = 0.5 * (1 + tanh(...)) ──
    vmul.bf16 m14, m8, m12
    DELAY 64

    # ── m10,m11 = X * 0.5*(1+tanh(...)) = GELU(X) ──
    vmul.bf16 m10, m0, m14
    DELAY 64

    LI    x8, 512
    vstore m10, 0(x8)
    DELAY 33
    vstore m11, 8(x8)
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90000800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 768
    LI    x3, 0x90000C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

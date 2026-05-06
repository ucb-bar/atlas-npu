# smolvla_silu.S — port of npu-model smolvla_silu
# Tests: silu(x) = x / (1 + exp(-x)) on seeded random BF16 32x32 tile
# @TIMEOUT 30000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_silu.py
#
# Atlas has no VSILU opcode; this synthesizes silu from primitives.
# All vector ops are pair-ops on (m, m+1) even-bank pairs.
#
# DRAM layout (set by gen_smolvla_silu.py):
#   [0x0000..0x03FF]  Input X bank 0    (1024 B)
#   [0x0400..0x07FF]  Input X bank 1    (1024 B)
#   [0x0800..0x0FFF]  Output            (2048 B, 2 banks)
#
# Mreg allocation:
#   m0,m1   = X
#   m2,m3   = -1.0 constant        m4,m5   = +1.0 constant
#   m6,m7   = -X                   m8,m9   = exp(-X)
#   m10,m11 = 1+exp(-X)            m12,m13 = 1/(1+exp(-X)) = sigmoid
#   m14,m15 = X * sigmoid = silu
# VMEM word addrs:  [0..255] X0  [256..511] X1
#                   [512..767] out0  [768..1023] out1

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024              # bank size in bytes

    # ── Load Input X → VMEM[0..511] ──
    ADDI  x6, x0, 0             # VMEM[0]
    ADDI  x1, x0, 0             # DRAM[0]
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256               # VMEM[256]
    LI    x1, 1024              # DRAM[1024]
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ── Constants: m2,m3 = -1.0   m4,m5 = +1.0 ──
    VLI.ALL 2, 0xBF80
    DELAY 63
    VLI.ALL 4, 0x3F80
    DELAY 63

    # ── m6,m7 = X * -1 ──
    VMUL.BF16  6, 0, 2
    DELAY 64

    # ── m8,m9 = exp(-X) ──
    VEXP 8, 6
    DELAY 64

    # ── m10,m11 = exp(-X) + 1 ──
    VADD.BF16 10, 8, 4
    DELAY 64

    # ── m12,m13 = 1 / (1 + exp(-X)) = sigmoid(X) ──
    VRECIP.BF16 12, 10
    DELAY 64

    # ── m14,m15 = X * sigmoid(X) = silu(X) ──
    VMUL.BF16 14, 0, 12
    DELAY 64

    LI    x8, 512
    VSTORE 14, x8, 0
    DELAY 33
    VSTORE 15, x8, 8
    DELAY 33

    LI    x8, 512
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x0C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

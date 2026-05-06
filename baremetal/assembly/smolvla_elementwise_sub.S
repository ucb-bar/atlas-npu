# smolvla_elementwise_sub.S — port of npu-model smolvla_elementwise_sub
# Tests: vsub.bf16 on seeded random BF16 32x32 tiles
# @TIMEOUT 30000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_elementwise_sub.py
#
# DRAM layout (set by gen_smolvla_elementwise_sub.py):
#   [0x0000..0x03FF]  Input A bank 0    (1024 B)
#   [0x0400..0x07FF]  Input A bank 1    (1024 B)
#   [0x0800..0x0BFF]  Input B bank 0    (1024 B)
#   [0x0C00..0x0FFF]  Input B bank 1    (1024 B)
#   [0x1000..0x17FF]  Output            (2048 B, 2 banks)
#
# Mreg allocation:  m0,m1 = A    m2,m3 = B    m4,m5 = result
# VMEM word addrs:  [0..255] A0  [256..511] A1  [512..767] B0  [768..1023] B1
#                   [1024..1279] out0  [1280..1535] out1

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024              # bank size in bytes

    # ── Load Input A → VMEM[0..511] ──
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

    # ── Load Input B → VMEM[512..1023] ──
    LI    x7, 512               # VMEM[512]
    LI    x1, 2048              # DRAM[2048]
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 768               # VMEM[768]
    LI    x1, 3072              # DRAM[3072]
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0

    LI    x7, 512
    VLOAD 2, x7, 0
    DELAY 33
    VLOAD 3, x7, 8
    DELAY 33

    # ── Single pair op: (m4,m5) = (m0,m1) - (m2,m3) ──
    VSUB.BF16  4, 0, 2
    DELAY 64

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33

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

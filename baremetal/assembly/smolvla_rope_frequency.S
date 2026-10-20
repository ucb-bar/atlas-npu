# smolvla_rope_frequency.S — port of npu-model smolvla_rope_frequency
# Tests: y = cos(X) on seeded random BF16 32x32 tile
#   (trailing cos leg of RoPE frequency precompute)
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_rope_frequency.py
#
# DRAM layout:
#   [0x0000..0x03FF]  Input X bank 0   (1024 B)
#   [0x0400..0x07FF]  Input X bank 1   (1024 B)
#   [0x0800..0x0FFF]  Output Y         (2048 B)
#
# Mreg allocation:  m0,m1 = X    m2,m3 = cos(X)

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

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

    vcos.bf16 m2, m0
    DELAY 64

    LI    x8, 512
    vstore m2, 0(x8)
    DELAY 33
    vstore m3, 8(x8)
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

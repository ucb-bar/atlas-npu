# smolvla_parameterized_rope_frequency.S — tiled y = cos(X) for 64x64 BF16
# Port of jrmills20 parameterized_rope_frequency.py with Atlas bank-split
# tile layout and a scalar tile loop.
# @TIMEOUT 1000000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_parameterized_rope_frequency.py
#
# Shape: M=128, N=128 → 16 tiles. Tile-major order.
#
# DRAM layout:
#   [0x00000..0x07FFF]  Input X  (16 tiles x 2048B BF16)
#   [0x08000..0x0FFFF]  Output Y (16 tiles x 2048B BF16)
#
# VMEM staging (reused every iteration):
#   [0..255]     X bank 0    [256..511]   X bank 1
#   [512..767]   Y bank 0    [768..1023]  Y bank 1

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x22, 0x8000          # output DRAM base

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

tile_loop:
    SLLI  x20, x10, 11         # x20 = tile_idx * 2048

    # ── Load X tile ──
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

    # ── (m2, m3) = cos(X) ──
    VCOS 2, 0
    DELAY 64

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 33
    VSTORE 3, x8, 8
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

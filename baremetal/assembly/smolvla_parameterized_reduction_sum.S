# smolvla_parameterized_reduction_sum.S — tiled row-sum broadcast, 128x32 BF16
# Port of jrmills20 parameterized_reduction_sum.py. N is fixed at 32
# (Jeremy's constraint); M scales via a tile loop. Reduction is local to
# each 32-row group, so no cross-tile helper is needed.
# @TIMEOUT 1500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_reduction_sum.py
#
# Shape: M=512, N=32 → 16 groups of 32x32. Tile-major DRAM order.
# Each 32x32 group is bank-split (bank 0 = cols 0..15, bank 1 = cols 16..31).
# Output (both banks of broadcast rowsum) matches stock smolvla_reduction_sum.
#
# DRAM layout:
#   [0x00000..0x07FFF]  Input X   (16 groups x 2048B BF16)
#   [0x08000..0x0FFFF]  Output Y  (16 groups x 2048B BF16, H0 == H1 under broadcast)
#
# VMEM staging (reused every iteration):
#   [0..255]     X bank 0    [256..511]   X bank 1
#   [512..767]   Y bank 0    [768..1023]  Y bank 1
#
# Mreg allocation:  m0,m1 = X    m4,m5 = rowsum broadcast

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x22, 0x8000

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

    LUI   x15, 0x90000         # FPGA DRAM offset 0x9000_0000
tile_loop:
    SLLI  x20, x10, 11         # tile_idx * 2048
    ADD   x20, x20, x15        # x20 += FPGA DRAM base

    # ── Load X group ──
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

    # ── Row-sum broadcast: (m4, m5) = rsum((m0, m1)) ──
    VREDSUM.ROW.BF16 4, 0
    DELAY 37

    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33

    # ── DMA out (both banks, matches stock smolvla_reduction_sum) ──
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

# smolvla_parameterized_requant.S — tiled BF16 → FP8 unit-scale cast, 64x64
# Port of jrmills20 parameterized_requant.py. Same VFP8PACK semantics as
# stock smolvla_requant (E8M0 scale 0x7F = 2^0, SELI hoisted out of loop).
# @TIMEOUT 1000000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_parameterized_requant.py
#
# Input X is BF16 2048 B per tile (2 banks); packed output is FP8 1024 B
# per tile (single mreg). So DRAM output stride is 1024 B / tile, not 2048.
#
# Shape: M=128, N=128 → 16 tiles.
# DRAM layout:
#   [0x00000..0x07FFF]  Input X   (16 tiles x 2048B BF16 = 32KB)
#   [0x08000..0x0BFFF]  Output Y  (16 tiles x 1024B FP8 = 16KB)
#
# Mreg allocation:  m0,m1 = X BF16    m4 = packed FP8

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x22, 0x8000          # output DRAM base (FP8)

    # ── Unit-scale E8M0 0x7F → scale reg 0 (invariant across tiles) ──
    SELI 0, 0x7F

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

tile_loop:
    SLLI  x20, x10, 11         # x20 = tile_idx * 2048 (input stride)
    SLLI  x23, x10, 10         # x23 = tile_idx * 1024 (output stride)

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
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ── Pack BF16 (m0, m1) → FP8 m4 via scale reg 0 ──
    VFP8PACK 4, 0, 0
    DELAY 150

    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 40

    # ── DMA out: single 1024B write per tile ──
    ADD   x3, x22, x23
    LI    x8, 512
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

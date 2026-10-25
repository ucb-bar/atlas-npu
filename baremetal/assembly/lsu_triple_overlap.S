# lsu_triple_overlap.S — scalar + VLOAD + VSTORE overlap on split LSU paths
# @TIMEOUT 200000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_lsu_triple_overlap.py
#
# This test exercises two overlap windows without any inter-vector waits:
#   Phase 1: scalar stores + VLOAD + VSTORE
#   Phase 2: scalar loads  + VLOAD + VSTORE
#
# The VMEM addresses below keep the low-bank source tiles compact while moving
# the overlap destinations into higher banks. Under the current 256 KiB /
# 65 536-word banking, the residency is:
#
# VMEM regions (word addresses / byte addresses / current bank):
#   tile_A            0      /      0      bank 0
#   tile_B            32768  / 131072      bank 0
#   scalar_ld_src     65536  / 262144      bank 1
#   tile_C            131072 / 524288      bank 2
#   out_store_1       163840 / 655360      bank 2
#   out_load_1        196608 / 786432      bank 3
#   scalar_store_out  262144 / 1048576     bank 4
#   out_store_2       327680 / 1310720     bank 5
#   out_load_2        196864 / 787456      bank 3
#   scalar_load_out   262400 / 1049600     bank 4

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024

# ──────────────────────────────────────────────────────────────
# PHASE 0: DMA load source tiles and zero-fill scalar result tiles
# ──────────────────────────────────────────────────────────────

    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0      # tile_A
    LI    x6, 32768
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1      # tile_B
    LI    x6, 65536
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2      # scalar load source
    LI    x6, 131072
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch3      # tile_C
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

    LI    x10, 0x90001000
    LI    x6, 262144
    dma.load x6, x10, x12, ch0      # scalar_store_out <- zeros
    dma.wait ch0
    LI    x6, 262400
    dma.load x6, x10, x12, ch0      # scalar_load_out <- zeros
    dma.wait ch0

# ──────────────────────────────────────────────────────────────
# Scalar helper addresses and constants
# ──────────────────────────────────────────────────────────────

    LI    x20, 1048576             # scalar_store_out byte base
    LI    x21, 0x11111111
    LI    x22, 0x22222222
    LI    x23, 0x33333333
    LI    x24, 0x44444444
    LI    x25, 0x55555555
    LI    x26, 0x66666666
    LI    x27, 0x77777777
    LI    x28, 0x88888888
    LI    x29, 262144              # scalar_ld_src byte base
    LI    x30, 1049600             # scalar_load_out byte base

# ──────────────────────────────────────────────────────────────
# Preload m1 from tile_B so both overlap phases can VSTORE from it.
# ──────────────────────────────────────────────────────────────

    LI    x6, 32768
    vload m1, 0(x6)
    DELAY 32

# ──────────────────────────────────────────────────────────────
# PHASE 1: scalar stores + VLOAD(tile_A) + VSTORE(m1 -> out_store_1)
# ──────────────────────────────────────────────────────────────

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    LI    x6, 163840
    vstore m1, 0(x6)
    sw x21, 0(x20)
    sw x22, 4(x20)
    sw x23, 8(x20)
    sw x24, 12(x20)
    sw x25, 16(x20)
    sw x26, 20(x20)
    sw x27, 24(x20)
    sw x28, 28(x20)
    DELAY 24

    LI    x6, 196608
    vstore m0, 0(x6)
    DELAY 32

# ──────────────────────────────────────────────────────────────
# PHASE 2: scalar loads + VLOAD(tile_C) + VSTORE(m1 -> out_store_2)
# ──────────────────────────────────────────────────────────────

    LI    x6, 131072
    vload m2, 0(x6)
    LI    x6, 327680
    vstore m1, 0(x6)
    lw x14, 0(x29)
    DELAY 2
    lw x15, 4(x29)
    DELAY 2
    lw x16, 8(x29)
    DELAY 2
    lw x17, 12(x29)
    DELAY 2
    DELAY 24

    LI    x6, 196864
    vstore m2, 0(x6)
    DELAY 32

    sw x14, 0(x30)
    sw x15, 4(x30)
    sw x16, 8(x30)
    sw x17, 12(x30)

# ──────────────────────────────────────────────────────────────
# PHASE 3: DMA.STORE outputs back to DRAM for checking
# ──────────────────────────────────────────────────────────────

    LI    x6, 163840
    LI    x10, 0x90001400
    dma.store x10, x6, x12, ch0      # out_store_1 = tile_B
    dma.wait ch0

    LI    x6, 196608
    LI    x10, 0x90001800
    dma.store x10, x6, x12, ch0      # out_load_1 = tile_A
    dma.wait ch0

    LI    x6, 262144
    LI    x10, 0x90001C00
    dma.store x10, x6, x12, ch0      # scalar_store_out
    dma.wait ch0

    LI    x6, 327680
    LI    x10, 0x90002000
    dma.store x10, x6, x12, ch0      # out_store_2 = tile_B
    dma.wait ch0

    LI    x6, 196864
    LI    x10, 0x90002400
    dma.store x10, x6, x12, ch0      # out_load_2 = tile_C
    dma.wait ch0

    LI    x6, 262400
    LI    x10, 0x90002800
    dma.store x10, x6, x12, ch0      # scalar_load_out
    dma.wait ch0

    LI    x31, 1
    csrrw x0, 0x800, x31
    ECALL

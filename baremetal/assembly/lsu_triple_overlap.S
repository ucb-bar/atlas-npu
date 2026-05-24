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
    DMA.CONFIG x5, 0
    LI    x12, 1024

# ──────────────────────────────────────────────────────────────
# PHASE 0: DMA load source tiles and zero-fill scalar result tiles
# ──────────────────────────────────────────────────────────────

    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x12, 0      # tile_A
    LI    x6, 32768
    LI    x10, 0x90000400
    DMA.LOAD  x6, x10, x12, 1      # tile_B
    LI    x6, 65536
    LI    x10, 0x90000800
    DMA.LOAD  x6, x10, x12, 2      # scalar load source
    LI    x6, 131072
    LI    x10, 0x90000C00
    DMA.LOAD  x6, x10, x12, 3      # tile_C
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    LI    x10, 0x90001000
    LI    x6, 262144
    DMA.LOAD  x6, x10, x12, 0      # scalar_store_out <- zeros
    DMA.WAIT  0
    LI    x6, 262400
    DMA.LOAD  x6, x10, x12, 0      # scalar_load_out <- zeros
    DMA.WAIT  0

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
    VLOAD 1, x6, 0
    DELAY 32

# ──────────────────────────────────────────────────────────────
# PHASE 1: scalar stores + VLOAD(tile_A) + VSTORE(m1 -> out_store_1)
# ──────────────────────────────────────────────────────────────

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    LI    x6, 163840
    VSTORE 1, x6, 0
    SW    x21, x20, 0
    SW    x22, x20, 4
    SW    x23, x20, 8
    SW    x24, x20, 12
    SW    x25, x20, 16
    SW    x26, x20, 20
    SW    x27, x20, 24
    SW    x28, x20, 28
    DELAY 24

    LI    x6, 196608
    VSTORE 0, x6, 0
    DELAY 32

# ──────────────────────────────────────────────────────────────
# PHASE 2: scalar loads + VLOAD(tile_C) + VSTORE(m1 -> out_store_2)
# ──────────────────────────────────────────────────────────────

    LI    x6, 131072
    VLOAD 2, x6, 0
    LI    x6, 327680
    VSTORE 1, x6, 0
    LW    x14, x29, 0
    DELAY 2
    LW    x15, x29, 4
    DELAY 2
    LW    x16, x29, 8
    DELAY 2
    LW    x17, x29, 12
    DELAY 2
    DELAY 24

    LI    x6, 196864
    VSTORE 2, x6, 0
    DELAY 32

    SW    x14, x30, 0
    SW    x15, x30, 4
    SW    x16, x30, 8
    SW    x17, x30, 12

# ──────────────────────────────────────────────────────────────
# PHASE 3: DMA.STORE outputs back to DRAM for checking
# ──────────────────────────────────────────────────────────────

    LI    x6, 163840
    LI    x10, 0x90001400
    DMA.STORE x10, x6, x12, 0      # out_store_1 = tile_B
    DMA.WAIT  0

    LI    x6, 196608
    LI    x10, 0x90001800
    DMA.STORE x10, x6, x12, 0      # out_load_1 = tile_A
    DMA.WAIT  0

    LI    x6, 262144
    LI    x10, 0x90001C00
    DMA.STORE x10, x6, x12, 0      # scalar_store_out
    DMA.WAIT  0

    LI    x6, 327680
    LI    x10, 0x90002000
    DMA.STORE x10, x6, x12, 0      # out_store_2 = tile_B
    DMA.WAIT  0

    LI    x6, 196864
    LI    x10, 0x90002400
    DMA.STORE x10, x6, x12, 0      # out_load_2 = tile_C
    DMA.WAIT  0

    LI    x6, 262400
    LI    x10, 0x90002800
    DMA.STORE x10, x6, x12, 0      # scalar_load_out
    DMA.WAIT  0

    LI    x31, 1
    CSRW  x31, 0xC10
    ECALL

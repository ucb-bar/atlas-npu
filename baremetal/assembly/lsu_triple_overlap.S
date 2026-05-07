# lsu_triple_overlap.S — scalar + VLOAD + VSTORE overlap on split LSU paths
# @TIMEOUT 200000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_lsu_triple_overlap.py
#
# This test exercises two overlap windows without any inter-vector waits:
#   Phase 1: scalar stores + VLOAD + VSTORE
#   Phase 2: scalar loads  + VLOAD + VSTORE
#
# The schedule is bank-safe by construction under 128 KiB block banking:
#   - Phase 1 scalar stores target bank 6 while VLOAD/VSTORE use banks 0/4.
#   - Phase 2 scalar loads target bank 2 while VLOAD/VSTORE use banks 3/7.
#
# VMEM regions (word addresses / byte addresses):
#   tile_A            0      /      0      bank 0 slot 0
#   tile_B            32768  / 131072      bank 1 slot 0
#   scalar_ld_src     65536  / 262144      bank 2 slot 0
#   tile_C            98304  / 393216      bank 3 slot 0
#   out_store_1       131072 / 524288      bank 4 slot 0
#   out_load_1        163840 / 655360      bank 5 slot 0
#   scalar_store_out  196608 / 786432      bank 6 slot 0
#   out_store_2       229376 / 917504      bank 7 slot 0
#   out_load_2        131328 / 525312      bank 4 slot 1
#   scalar_load_out   229632 / 918528      bank 7 slot 1

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024

# ──────────────────────────────────────────────────────────────
# PHASE 0: DMA load source tiles and zero-fill scalar result tiles
# ──────────────────────────────────────────────────────────────

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0      # tile_A
    LI    x6, 32768
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 1      # tile_B
    LI    x6, 65536
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 2      # scalar load source
    LI    x6, 98304
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 3      # tile_C
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    LI    x10, 0x1000
    LI    x6, 196608
    DMA.LOAD  x6, x10, x12, 0      # scalar_store_out <- zeros
    DMA.WAIT  0
    LI    x6, 229632
    DMA.LOAD  x6, x10, x12, 0      # scalar_load_out <- zeros
    DMA.WAIT  0

# ──────────────────────────────────────────────────────────────
# Scalar helper addresses and constants
# ──────────────────────────────────────────────────────────────

    LI    x20, 786432              # scalar_store_out byte base
    LI    x21, 0x11111111
    LI    x22, 0x22222222
    LI    x23, 0x33333333
    LI    x24, 0x44444444
    LI    x25, 0x55555555
    LI    x26, 0x66666666
    LI    x27, 0x77777777
    LI    x28, 0x88888888
    LI    x29, 262144              # scalar_ld_src byte base
    LI    x30, 918528              # scalar_load_out byte base

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
    LI    x6, 131072
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

    LI    x6, 163840
    VSTORE 0, x6, 0
    DELAY 32

# ──────────────────────────────────────────────────────────────
# PHASE 2: scalar loads + VLOAD(tile_C) + VSTORE(m1 -> out_store_2)
# ──────────────────────────────────────────────────────────────

    LI    x6, 98304
    VLOAD 2, x6, 0
    LI    x6, 229376
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

    LI    x6, 131328
    VSTORE 2, x6, 0
    DELAY 32

    SW    x14, x30, 0
    SW    x15, x30, 4
    SW    x16, x30, 8
    SW    x17, x30, 12

# ──────────────────────────────────────────────────────────────
# PHASE 3: DMA.STORE outputs back to DRAM for checking
# ──────────────────────────────────────────────────────────────

    LI    x6, 131072
    LI    x10, 0x1400
    DMA.STORE x10, x6, x12, 0      # out_store_1 = tile_B
    DMA.WAIT  0

    LI    x6, 163840
    LI    x10, 0x1800
    DMA.STORE x10, x6, x12, 0      # out_load_1 = tile_A
    DMA.WAIT  0

    LI    x6, 196608
    LI    x10, 0x1C00
    DMA.STORE x10, x6, x12, 0      # scalar_store_out
    DMA.WAIT  0

    LI    x6, 229376
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 0      # out_store_2 = tile_B
    DMA.WAIT  0

    LI    x6, 131328
    LI    x10, 0x2400
    DMA.STORE x10, x6, x12, 0      # out_load_2 = tile_C
    DMA.WAIT  0

    LI    x6, 229632
    LI    x10, 0x2800
    DMA.STORE x10, x6, x12, 0      # scalar_load_out
    DMA.WAIT  0

    LI    x31, 1
    CSRW  x31, 0xC10
    ECALL

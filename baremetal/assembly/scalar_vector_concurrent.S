# scalar_vector_concurrent.S — legacy scalar/vector overlap layout on block-banked VMEM
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_scalar_vector_concurrent.py 42
#
# VMEM is now 6-bank block-banked in 256 KiB chunks (65 536 words / bank), so
# each aligned 1 KiB tensor still stays in a single bank for the full
# VLOAD/VSTORE. The address plan below keeps the low-bank tiles compact while
# moving the scalar-only staging regions into higher banks so every overlap
# window stays legal under the current residency:
#   • During VLOAD  (vector reads)  → scalar SW only
#   • During VSTORE (vector writes) → scalar LW only
#   • During XLU / MXU compute      → scalar LW/SW chains with a 2-cycle gap after LW
#
# Address units:
#   • DMA / VLOAD / VSTORE use VMEM word addresses
#   • scalar LW / SW use VMEM byte addresses
#
# VMEM regions:
#   tile_A   word 0       byte 0        bank 0
#   tile_B   word 32768   byte 131072   bank 0
#   tile_W   word 65536   byte 262144   bank 1
#   bias     word 131072  byte 524288   bank 2
#   vstore   word 163840  byte 655360   bank 2
#   stage1   word 196608  byte 786432   bank 3
#   stage2   word 262144  byte 1048576  bank 4
#   stage3   word 327680  byte 1310720  bank 5

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024
    seli e0, 127
    seli e1, 127

# ═══════════════════════════════════════════════════════════════
# PHASE 0: DMA load inputs + pre-zero staging
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0      # tile_A
    LI    x6, 32768
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1      # tile_B
    LI    x6, 65536
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2      # tile_W
    LI    x6, 131072
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch3      # bias
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

    LI    x10, 0x90001000
    LI    x6, 196608
    dma.load x6, x10, x12, ch0      # stage1 <- zeros
    dma.wait ch0
    LI    x6, 262144
    dma.load x6, x10, x12, ch0      # stage2 <- zeros
    dma.wait ch0
    LI    x6, 327680
    dma.load x6, x10, x12, ch0      # stage3 <- zeros
    dma.wait ch0

# ═══════════════════════════════════════════════════════════════
# Scalar base addresses and constants
# ═══════════════════════════════════════════════════════════════

    LI    x14, 131072             # tile_B byte base
    LI    x15, 262144             # tile_W byte base
    LI    x16, 786432             # stage1 byte base
    LI    x17, 1048576            # stage2 byte base
    LI    x18, 1310720            # stage3 byte base
    LI    x27, 0xCAFEBABE
    LI    x28, 0xDEADBEEF

# ═══════════════════════════════════════════════════════════════
# PHASE 1: VLOAD m0 + scalar stores
#   Vector path reads VMEM, so scalar uses write port only.
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    sw x27, 0(x16)              # stage1 word 0 = CAFE
    sw x28, 4(x16)              # stage1 word 1 = DEAD
    sw x28, 0(x17)              # stage2 word 0 = DEAD
    sw x27, 4(x17)              # stage2 word 1 = CAFE
    sw x27, 0(x18)              # stage3 word 0 = CAFE
    sw x28, 4(x18)              # stage3 word 1 = DEAD
    DELAY 26

# ═══════════════════════════════════════════════════════════════
# PHASE 2: VSTORE m0 + scalar loads
#   Vector path writes VMEM, so scalar uses read port only.
#   Each scalar load is followed by DELAY 2 to respect the longer scalar LSU latency.
# ═══════════════════════════════════════════════════════════════

    LI    x6, 163840
    vstore m0, 0(x6)
    lw x21, 0(x14)              # tile_B word 0
    DELAY 2
    lw x22, 4(x14)              # tile_B word 1
    DELAY 2
    lw x23, 0(x15)              # tile_W word 0
    DELAY 2
    lw x24, 0(x16)              # stage1 word 0 = CAFE
    DELAY 2
    lw x25, 0(x17)              # stage2 word 0 = DEAD
    DELAY 2
    DELAY 22

# ═══════════════════════════════════════════════════════════════
# PHASE 3: VLOAD m1 + scalar stores from Phase 2 readback
# ═══════════════════════════════════════════════════════════════

    LI    x6, 65536
    vload m1, 0(x6)
    sw x21, 8(x16)              # stage1 word 2 = tile_B word 0
    sw x22, 12(x16)             # stage1 word 3 = tile_B word 1
    sw x23, 8(x17)              # stage2 word 2 = tile_W word 0
    sw x24, 12(x17)             # stage2 word 3 = CAFE
    sw x25, 8(x18)              # stage3 word 2 = DEAD
    sw x21, 12(x18)             # stage3 word 3 = tile_B word 0
    DELAY 27

    vtrpose.xlu m2, m1
    lw x26, 8(x16)              # read stage1 word 2 = tile_B word 0
    DELAY 2
    sw x26, 16(x17)             # stage2 word 4 = tile_B word 0
    lw x20, 12(x17)             # read stage2 word 3 = CAFE
    DELAY 2
    sw x20, 16(x18)             # stage3 word 4 = CAFE
    lw x19, 8(x18)              # read stage3 word 2 = DEAD
    DELAY 2
    sw x19, 16(x16)             # stage1 word 4 = DEAD
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# PHASE 4: VLOAD bias + MXU compute with scalar chains in VMEM-free windows
# ═══════════════════════════════════════════════════════════════

    LI    x6, 131072
    vload m3, 0(x6)
    sw x23, 20(x16)             # stage1 word 5 = tile_W word 0
    sw x22, 20(x17)             # stage2 word 5 = tile_B word 1
    sw x24, 20(x18)             # stage3 word 5 = CAFE
    DELAY 29

    vmatpush.weight.mxu1 w0, m2
    vmatpush.acc.fp8.mxu1 acc0, m3
    DELAY 31

    vmatmul.acc.mxu1 acc0, m0, w0
    lw x21, 16(x16)             # read stage1 word 4 = DEAD
    DELAY 2
    sw x21, 24(x17)             # stage2 word 6 = DEAD
    lw x22, 12(x18)             # read stage3 word 3 = tile_B word 0
    DELAY 2
    sw x22, 24(x16)             # stage1 word 6 = tile_B word 0
    lw x23, 8(x17)              # read stage2 word 2 = tile_W word 0
    DELAY 2
    sw x23, 24(x18)             # stage3 word 6 = tile_W word 0
    DELAY 95

    vmatpop.bf16.acc.mxu1 m4, acc0
    DELAY 31

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Final VSTOREs + scalar readback overlap
# ═══════════════════════════════════════════════════════════════

    LI    x6, 65536
    vstore m4, 0(x6)                # C0 lo -> tile_W region
    lw x24, 20(x16)             # read stage1 word 5 = tile_W word 0
    DELAY 2
    lw x25, 24(x17)             # read stage2 word 6 = DEAD
    DELAY 2
    DELAY 28

    LI    x6, 131072
    vstore m5, 0(x6)                # C0 hi -> bias region
    lw x26, 24(x18)             # read stage3 word 6 = tile_W word 0
    DELAY 2
    lw x19, 24(x16)             # read stage1 word 6 = tile_B word 0
    DELAY 2
    DELAY 28

# ═══════════════════════════════════════════════════════════════
# PHASE 6: DMA.STORE outputs
# ═══════════════════════════════════════════════════════════════

    LI    x6, 163840
    LI    x10, 0x90001400
    dma.store x10, x6, x12, ch0      # vstore echo of tile_A
    dma.wait ch0

    LI    x6, 196608
    LI    x10, 0x90001800
    dma.store x10, x6, x12, ch0      # stage1
    dma.wait ch0

    LI    x6, 262144
    LI    x10, 0x90001C00
    dma.store x10, x6, x12, ch0      # stage2
    dma.wait ch0

    LI    x6, 327680
    LI    x10, 0x90002000
    dma.store x10, x6, x12, ch0      # stage3
    dma.wait ch0

    LI    x6, 65536
    LI    x10, 0x90002400
    dma.store x10, x6, x12, ch0      # C0 lo
    dma.wait ch0

    LI    x6, 131072
    LI    x10, 0x90002800
    dma.store x10, x6, x12, ch0      # C0 hi
    dma.wait ch0

    LI    x30, 1
    csrrw x0, 0x800, x30
    ECALL

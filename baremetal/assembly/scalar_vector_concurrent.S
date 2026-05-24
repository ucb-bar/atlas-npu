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
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127
    SELI  1, 127

# ═══════════════════════════════════════════════════════════════
# PHASE 0: DMA load inputs + pre-zero staging
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x12, 0      # tile_A
    LI    x6, 32768
    LI    x10, 0x90000400
    DMA.LOAD  x6, x10, x12, 1      # tile_B
    LI    x6, 65536
    LI    x10, 0x90000800
    DMA.LOAD  x6, x10, x12, 2      # tile_W
    LI    x6, 131072
    LI    x10, 0x90000C00
    DMA.LOAD  x6, x10, x12, 3      # bias
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    LI    x10, 0x90001000
    LI    x6, 196608
    DMA.LOAD  x6, x10, x12, 0      # stage1 <- zeros
    DMA.WAIT  0
    LI    x6, 262144
    DMA.LOAD  x6, x10, x12, 0      # stage2 <- zeros
    DMA.WAIT  0
    LI    x6, 327680
    DMA.LOAD  x6, x10, x12, 0      # stage3 <- zeros
    DMA.WAIT  0

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
    VLOAD 0, x6, 0
    SW    x27, x16, 0              # stage1 word 0 = CAFE
    SW    x28, x16, 4              # stage1 word 1 = DEAD
    SW    x28, x17, 0              # stage2 word 0 = DEAD
    SW    x27, x17, 4              # stage2 word 1 = CAFE
    SW    x27, x18, 0              # stage3 word 0 = CAFE
    SW    x28, x18, 4              # stage3 word 1 = DEAD
    DELAY 26

# ═══════════════════════════════════════════════════════════════
# PHASE 2: VSTORE m0 + scalar loads
#   Vector path writes VMEM, so scalar uses read port only.
#   Each scalar load is followed by DELAY 2 to respect the longer scalar LSU latency.
# ═══════════════════════════════════════════════════════════════

    LI    x6, 163840
    VSTORE 0, x6, 0
    LW    x21, x14, 0              # tile_B word 0
    DELAY 2
    LW    x22, x14, 4              # tile_B word 1
    DELAY 2
    LW    x23, x15, 0              # tile_W word 0
    DELAY 2
    LW    x24, x16, 0              # stage1 word 0 = CAFE
    DELAY 2
    LW    x25, x17, 0              # stage2 word 0 = DEAD
    DELAY 2
    DELAY 22

# ═══════════════════════════════════════════════════════════════
# PHASE 3: VLOAD m1 + scalar stores from Phase 2 readback
# ═══════════════════════════════════════════════════════════════

    LI    x6, 65536
    VLOAD 1, x6, 0
    SW    x21, x16, 8              # stage1 word 2 = tile_B word 0
    SW    x22, x16, 12             # stage1 word 3 = tile_B word 1
    SW    x23, x17, 8              # stage2 word 2 = tile_W word 0
    SW    x24, x17, 12             # stage2 word 3 = CAFE
    SW    x25, x18, 8              # stage3 word 2 = DEAD
    SW    x21, x18, 12             # stage3 word 3 = tile_B word 0
    DELAY 27

    VTRPOSE.XLU 2, 1
    LW    x26, x16, 8              # read stage1 word 2 = tile_B word 0
    DELAY 2
    SW    x26, x17, 16             # stage2 word 4 = tile_B word 0
    LW    x20, x17, 12             # read stage2 word 3 = CAFE
    DELAY 2
    SW    x20, x18, 16             # stage3 word 4 = CAFE
    LW    x19, x18, 8              # read stage3 word 2 = DEAD
    DELAY 2
    SW    x19, x16, 16             # stage1 word 4 = DEAD
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# PHASE 4: VLOAD bias + MXU compute with scalar chains in VMEM-free windows
# ═══════════════════════════════════════════════════════════════

    LI    x6, 131072
    VLOAD 3, x6, 0
    SW    x23, x16, 20             # stage1 word 5 = tile_W word 0
    SW    x22, x17, 20             # stage2 word 5 = tile_B word 1
    SW    x24, x18, 20             # stage3 word 5 = CAFE
    DELAY 29

    VMATPUSH.W.MXU0       0, 2
    VMATPUSH.ACC.FP8.MXU0 0, 3
    DELAY 31

    VMATMUL.ACC.MXU0      0, 0, 0
    LW    x21, x16, 16             # read stage1 word 4 = DEAD
    DELAY 2
    SW    x21, x17, 24             # stage2 word 6 = DEAD
    LW    x22, x18, 12             # read stage3 word 3 = tile_B word 0
    DELAY 2
    SW    x22, x16, 24             # stage1 word 6 = tile_B word 0
    LW    x23, x17, 8              # read stage2 word 2 = tile_W word 0
    DELAY 2
    SW    x23, x18, 24             # stage3 word 6 = tile_W word 0
    DELAY 95

    VMATPOP.BF16.MXU0  4, 0
    DELAY 31

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Final VSTOREs + scalar readback overlap
# ═══════════════════════════════════════════════════════════════

    LI    x6, 65536
    VSTORE 4, x6, 0                # C0 lo -> tile_W region
    LW    x24, x16, 20             # read stage1 word 5 = tile_W word 0
    DELAY 2
    LW    x25, x17, 24             # read stage2 word 6 = DEAD
    DELAY 2
    DELAY 28

    LI    x6, 131072
    VSTORE 5, x6, 0                # C0 hi -> bias region
    LW    x26, x18, 24             # read stage3 word 6 = tile_W word 0
    DELAY 2
    LW    x19, x16, 24             # read stage1 word 6 = tile_B word 0
    DELAY 2
    DELAY 28

# ═══════════════════════════════════════════════════════════════
# PHASE 6: DMA.STORE outputs
# ═══════════════════════════════════════════════════════════════

    LI    x6, 163840
    LI    x10, 0x90001400
    DMA.STORE x10, x6, x12, 0      # vstore echo of tile_A
    DMA.WAIT  0

    LI    x6, 196608
    LI    x10, 0x90001800
    DMA.STORE x10, x6, x12, 0      # stage1
    DMA.WAIT  0

    LI    x6, 262144
    LI    x10, 0x90001C00
    DMA.STORE x10, x6, x12, 0      # stage2
    DMA.WAIT  0

    LI    x6, 327680
    LI    x10, 0x90002000
    DMA.STORE x10, x6, x12, 0      # stage3
    DMA.WAIT  0

    LI    x6, 65536
    LI    x10, 0x90002400
    DMA.STORE x10, x6, x12, 0      # C0 lo
    DMA.WAIT  0

    LI    x6, 131072
    LI    x10, 0x90002800
    DMA.STORE x10, x6, x12, 0      # C0 hi
    DMA.WAIT  0

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

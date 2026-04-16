# scalar_vector_concurrent.S — safe scalar/vector overlap on interleaved VMEM
# @TIMEOUT 800000
# @PYTHON_GEN gen_scalar_vector_concurrent.py 42
#
# VMEM is 8-bank interleaved by lineAddr[2:0], so every 32-line VLOAD/VSTORE
# sweeps all banks. This test avoids LSU same-port conflicts by construction:
#   • During VLOAD  (vector reads)  → scalar SW only
#   • During VSTORE (vector writes) → scalar LW only
#   • During XLU / MXU compute      → scalar LW/SW chains with a NOP after LW
#
# Address units:
#   • DMA / VLOAD / VSTORE use VMEM word addresses
#   • scalar LW / SW use VMEM byte addresses
#
# VMEM regions:
#   tile_A   word 0      byte 0
#   tile_B   word 256    byte 1024
#   tile_W   word 512    byte 2048
#   bias     word 768    byte 3072
#   vstore   word 1024   byte 4096
#   stage1   word 1280   byte 5120
#   stage2   word 1536   byte 6144
#   stage3   word 1792   byte 7168
#
# stage1/stage2/stage3 are 1024-byte VMEM regions, not physical banks.
# Their first lines all start on bank 0 because the base line addresses are
# multiples of 8.

    LI    x5,  0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127
    SELI  1, 127

# ═══════════════════════════════════════════════════════════════
# PHASE 0: DMA load inputs + pre-zero staging
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0      # tile_A
    LI    x6, 256
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 1      # tile_B
    LI    x6, 512
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 2      # tile_W
    LI    x6, 768
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 3      # bias
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    LI    x10, 0x1000
    LI    x6, 1280
    DMA.LOAD  x6, x10, x12, 0      # stage1 <- zeros
    DMA.WAIT  0
    LI    x6, 1536
    DMA.LOAD  x6, x10, x12, 0      # stage2 <- zeros
    DMA.WAIT  0
    LI    x6, 1792
    DMA.LOAD  x6, x10, x12, 0      # stage3 <- zeros
    DMA.WAIT  0

# ═══════════════════════════════════════════════════════════════
# Scalar base addresses and constants
# ═══════════════════════════════════════════════════════════════

    LI    x14, 1024               # tile_B byte base
    LI    x15, 2048               # tile_W byte base
    LI    x16, 5120               # stage1 byte base
    LI    x17, 6144               # stage2 byte base
    LI    x18, 7168               # stage3 byte base
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
#   Each scalar load is followed by NOP to satisfy scalar-core assertions.
# ═══════════════════════════════════════════════════════════════

    LI    x6, 1024
    VSTORE 0, x6, 0
    LW    x21, x14, 0              # tile_B word 0
    NOP
    LW    x22, x14, 4              # tile_B word 1
    NOP
    LW    x23, x15, 0              # tile_W word 0
    NOP
    LW    x24, x16, 0              # stage1 word 0 = CAFE
    NOP
    LW    x25, x17, 0              # stage2 word 0 = DEAD
    NOP
    DELAY 22

# ═══════════════════════════════════════════════════════════════
# PHASE 3: VLOAD m1 + scalar stores from Phase 2 readback
# ═══════════════════════════════════════════════════════════════

    LI    x6, 512
    VLOAD 1, x6, 0
    SW    x21, x16, 8              # stage1 word 2 = tile_B word 0
    SW    x22, x16, 12             # stage1 word 3 = tile_B word 1
    SW    x23, x17, 8              # stage2 word 2 = tile_W word 0
    SW    x24, x17, 12             # stage2 word 3 = CAFE
    SW    x25, x18, 8              # stage3 word 2 = DEAD
    SW    x21, x18, 12             # stage3 word 3 = tile_B word 0
    DELAY 26

    VTRPOSE.XLU 2, 1
    LW    x26, x16, 8              # read stage1 word 2 = tile_B word 0
    NOP
    SW    x26, x17, 16             # stage2 word 4 = tile_B word 0
    LW    x20, x17, 12             # read stage2 word 3 = CAFE
    NOP
    SW    x20, x18, 16             # stage3 word 4 = CAFE
    LW    x19, x18, 8              # read stage3 word 2 = DEAD
    NOP
    SW    x19, x16, 16             # stage1 word 4 = DEAD
    DELAY 71

# ═══════════════════════════════════════════════════════════════
# PHASE 4: VLOAD bias + MXU compute with scalar chains in VMEM-free windows
# ═══════════════════════════════════════════════════════════════

    LI    x6, 768
    VLOAD 3, x6, 0
    SW    x23, x16, 20             # stage1 word 5 = tile_W word 0
    SW    x22, x17, 20             # stage2 word 5 = tile_B word 1
    SW    x24, x18, 20             # stage3 word 5 = CAFE
    DELAY 29

    VMATPUSH.W.MXU0       0, 2
    VMATPUSH.ACC.FP8.MXU0 0, 3
    DELAY 35

    VMATMUL.ACC.MXU0      0, 0, 0
    LW    x21, x16, 16             # read stage1 word 4 = DEAD
    NOP
    SW    x21, x17, 24             # stage2 word 6 = DEAD
    LW    x22, x18, 12             # read stage3 word 3 = tile_B word 0
    NOP
    SW    x22, x16, 24             # stage1 word 6 = tile_B word 0
    LW    x23, x17, 8              # read stage2 word 2 = tile_W word 0
    NOP
    SW    x23, x18, 24             # stage3 word 6 = tile_W word 0
    DELAY 112

    VMATPOP.BF16.MXU0  4, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE 5: Final VSTOREs + scalar readback overlap
# ═══════════════════════════════════════════════════════════════

    LI    x6, 512
    VSTORE 4, x6, 0                # C0 lo -> tile_W region
    LW    x24, x16, 20             # read stage1 word 5 = tile_W word 0
    NOP
    LW    x25, x17, 24             # read stage2 word 6 = DEAD
    NOP
    DELAY 28

    LI    x6, 768
    VSTORE 5, x6, 0                # C0 hi -> bias region
    LW    x26, x18, 24             # read stage3 word 6 = tile_W word 0
    NOP
    LW    x19, x16, 24             # read stage1 word 6 = tile_B word 0
    NOP
    DELAY 28

# ═══════════════════════════════════════════════════════════════
# PHASE 6: DMA.STORE outputs
# ═══════════════════════════════════════════════════════════════

    LI    x6, 1024
    LI    x10, 0x1400
    DMA.STORE x10, x6, x12, 0      # vstore echo of tile_A
    DMA.WAIT  0

    LI    x6, 1280
    LI    x10, 0x1800
    DMA.STORE x10, x6, x12, 0      # stage1
    DMA.WAIT  0

    LI    x6, 1536
    LI    x10, 0x1C00
    DMA.STORE x10, x6, x12, 0      # stage2
    DMA.WAIT  0

    LI    x6, 1792
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 0      # stage3
    DMA.WAIT  0

    LI    x6, 512
    LI    x10, 0x2400
    DMA.STORE x10, x6, x12, 0      # C0 lo
    DMA.WAIT  0

    LI    x6, 768
    LI    x10, 0x2800
    DMA.STORE x10, x6, x12, 0      # C0 hi
    DMA.WAIT  0

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

# perf_vec_layernorm_32x32.S — row-wise layer norm on BF16 32×32 tile
# @TIMEOUT 20000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_perf_vec_layernorm_32x32.py
# @PERF_REPORT
#
# Computes: Y = (X - rowmean(X)) / sqrt(rowvar(X) + ε)
# over a seeded random BF16 32×32 tile, entirely in vector ops.
#
# Pipeline (critical path, no MXU):
#   1. VLOAD        m0,m1            34 cy  — load X
#   2. VREDSUM.ROW  m2,m3 ← m0,m1   39 cy  — row sums
#   3. VMUL.BF16    m2,m3 (×1/32)   65 cy  — scale to row mean (uses preloaded 1/32)
#   4. VSUB.BF16    m4,m5 ← m0-mean 65 cy  — x - mean  (feeds both paths)
#   5. VSQUARE.BF16 m6,m7 ← m4,m5   65 cy  — (x-mean)²
#   6. VREDSUM.ROW  m8,m9 ← m6,m7   39 cy  — row sum of squares
#   7. VMUL.BF16    m8,m9 (×1/32)   65 cy  — variance
#   8. VSQRT.BF16   m8,m9           65 cy  — sqrt(var+ε)  [ε folded into seed]
#   9. VRECIP.BF16  m8,m9           65 cy  — 1/std
#  10. VMUL.BF16    m10,m11←m4×m8   65 cy  — (x-mean)/std = Y
#  11. VSTORE       m10,m11          34 cy  — store Y
#
# theoretical cycle count = vli + rowsum + mul + sub  + square + rowsum + mul + sqrt + recip + mul
#                           9 operations on 32x32 entires = 9*1024 = 9216
#       throughput per op = 16, 32 , 16 , 16 , 32 , 16 , 16 , 16 , 16
# optimal cycle count     = 64 + 32 + 64 + 64 + 32 + 64 + 64 + 64 + 64 = 512
# actual = 743 (with vload/vstore)
# utilization = 68.9%
#
# actual = 605 (without vload/vstore)
# utilization = 84.6%

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 2048

# ═══════════════════════════════════════════════════════════════
# SETUP: load X into MREG (NOT timed)
# ═══════════════════════════════════════════════════════════════

    LI    x6, 0x20000000         # VMEM base address
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x12, 0
    DMA.WAIT  0

    # Load X (two halves → m0, m1)
    LI    x6, 0x20000000         # VMEM base address
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8               # Fixed typo (was 6)
    DELAY 33

    # ═══════════════════════════════════════════════════════════════
    # MEASURED REGION BEGINS
    # ═══════════════════════════════════════════════════════════════
    CSRRS x20, 0xC00, x0       # Read CSR_CYCLE_COUNTER (0xC00) into x20

    # Preload 1/32 scalar into m12,m13 (used for mean and variance scaling)
    # VLI.ALL fills every lane — 0x3D00 is BF16 for 0.03125 = 1/32
    VLI.ALL  12, 0x3D00          # Fixed BF16 constant
    DELAY 63
    
# ── Step 1: row sum → scale to mean ──
    VREDSUM.ROW.BF16  2, 0       # m2,m3 = rowsum(X)
    DELAY 37
    VMUL.BF16         2, 2, 12   # m2,m3 = rowsum × (1/32) = rowmean
    DELAY 64

# ── Step 2: x - mean ──
    VSUB.BF16         4, 0, 2    # m4,m5 = X - rowmean  (key intermediate)
    DELAY 64

# ── Step 3: variance via (x-mean)² summed per row ──
    VSQUARE.BF16      6, 4       # m6,m7 = (x-mean)²
    DELAY 64
    VREDSUM.ROW.BF16  8, 6       # m8,m9 = rowsum((x-mean)²)
    DELAY 37
    VMUL.BF16         8, 8, 12   # m8,m9 = variance (= sum/32)
    DELAY 64

# ── Step 4: 1/std = 1/sqrt(var)  [ε small, absorbed into golden gen] ──
    VSQRT        8, 8       # m8,m9 = sqrt(var)
    DELAY 64
    VRECIP.BF16       8, 8       # m8,m9 = 1/std
    DELAY 64

# ── Step 5: normalize ──
    VMUL.BF16         10, 4, 8   # m10,m11 = (x-mean) * (1/std) = Y
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS
# ═══════════════════════════════════════════════════════════════
    CSRRS  x21, 0xC00, x0             # x21 = cycle count at end
    SUB   x22, x21, x20          # x22 = elapsed cycles
    CSRRW  x0, 0xC11, x22              # stash in CSR_DBG1

# ═══════════════════════════════════════════════════════════════
# WRITEBACK (not timed) — MREG → VMEM → DRAM
# ═══════════════════════════════════════════════════════════════
    LI    x8, 0
    VSTORE 10, x8, 0
    DELAY 33
    VSTORE 11, x8, 8
    DELAY 33



    LI    x8, 0
    LI    x3, 0x0800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE — host checks DRAM 0x0800 against golden; reads CSR_DBG1
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL
# perf_vec_rmsnorm_softmax_32x32.S — fused RMSNorm(X) + Softmax(Z) with FU overlap
# @TIMEOUT 25000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_perf_vec_rmsnorm_softmax_32x32.py
# @PERF_REPORT
#
# Computes two independent 32×32 BF16 tiles simultaneously:
#
#   Y_x = X / sqrt(mean(X²))                          [RMSNorm]
#   Y_z = exp(Z - rowmax(Z)) / rowsum(exp(Z - rowmax)) [Softmax]
#
# ═══════════════════════════════════════════════════════════════
# FU OVERLAP ANALYSIS
# ═══════════════════════════════════════════════════════════════
#
# RMSNorm chain: VSQUARE → VREDSUM.ROW → VMUL → VSQRT → VRECIP → VMUL
# Softmax chain: VREDMAX.ROW → VSUB → VEXP → VREDSUM.ROW → VRECIP → VMUL
#
# Both chains are RAW-sequential internally. The only place a sole-FU
# op from X and a sole-FU op from Z are simultaneously ready — with no
# double-port op needed by either path — is:
#
#   X: VSQRT  needs mean(X²) from VMUL(X)       → ready after slot 4
#   Z: VEXP   needs (Z - rowmax) from VSUB(Z)   → ready after slot 3
# theoretical cycle count = softmax: vli + rowmax  + sub    + exp   + rowsum + recip + mul on 32x32 tile
#                           rmsnorm: square + rowsum + mul + sqrt + recip + mul
#                           vli + rowmax + square + rowsum + sub + mul + (sqrt/exp) + rowsum + recip + recip + mul + mul
#       throughput per op =     16, 32 , 16 , 32 , 16 , 16 , 32 , 32 , 16 ,16 , 16 , 16
#                               
# optimal cycle count     = 64 + 32 + 64 + 32 + 64 + 64 + 32 + 32 + 64 + 64 + 64 + 64 = 640
#
# actual = 953 (with vload/vstore)
# utilization = 67%
# 
# actual = 707 (without vload/vstore)
# utilization = 89.8%
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 2048

# ─── DMA load all tiles into VMEM ──────────────────────────────
    LI    x6, 0x20000000
    LI    x9, 0x20000400        # Z staging: 1 KiB past X (each tile = 2048 B = 2 KiB,
    LI    x1, 0x90000000  # but VLOAD block banking requires 1 KiB alignment;
    dma.load x6, x1, x12, ch0   # use 0x20000000 for X, 0x20000800 for Z so they
    LI    x1, 0x90000800  # don't overlap: X=2048 B ends at 0x20000800 exactly)
    dma.load x9, x1, x12, ch1
    dma.wait ch0
    dma.wait ch1


# ─── VLOAD all tiles into MREG ─────────────────────────────────
    vload m0, 0(x6)               # m0 ← X low  bank (VMEM 0x20000000)
    DELAY 33

    vload m1, 8(x6)               # m1 ← X high bank
    DELAY 33

    vload m4, 0(x9)               # m4 ← Z low  bank (VMEM 0x20001000)
    DELAY 33

    vload m5, 8(x9)               # m5 ← Z high bank
    DELAY 33
# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS
# ═══════════════════════════════════════════════════════════════
    CSRRS x20, 0xC00, x0        # x20 = cycle count at start

    vli.all m12, 0x3000             # m12 ← 1/32 (pairs with VLOAD)
    DELAY 63

# ── Slot 1 (cycle 0, 34 cy): VREDMAX.ROW m6←m4 ─────────────────────────
    vredmax.row.bf16 m6, m4      # m6,m7 = rowmax(Z) broadcast    (34 cy)
    DELAY 32

# ── Slot 2 (cycle 35, 65 cy): VSQUARE m2←m0 ────────────────────────────
    vsquare.bf16 m2, m0      # m2,m3 = X²                     (65 cy)
    DELAY 64

# ── Slot 3 (cycle 101, 39 cy): VREDSUM.ROW m2←m2 ───────────────────────
    vredsum.row.bf16 m2, m2      # m2,m3 = rowsum(X²)             (39 cy)
    DELAY 37

# ── Slot 4 (cycle 141, 65 cy): VSUB m6←m4,m6 ───────────────────────────
    vsub.bf16 m6, m4, m6   # m6,m7 = Z - rowmax(Z)          (65 cy)
    DELAY 64

# ── Slot 5 (cycle 207, 65 cy): VMUL m2←m2,m12 ──────────────────────────
    vmul.bf16 m2, m2, m12  # m2,m3 = mean(X²)               (65 cy)
    DELAY 64

# Slot 6: VSQRT m2←m2  ∥  VEXP m6←m6   (write exp BACK into m6, not m8)
    vsqrt.bf16 m2, m2      # m2,m3 = rms
    vexp.bf16 m6, m6      # m6,m7 = exp(Z - rowmax)  ← preserve here
    DELAY 64

# Slot 7: VREDSUM.ROW m8←m6             (sum into m8, leaving m6 intact)
    vredsum.row.bf16 m8, m6  # m8,m9 = rowsum(exp)
    DELAY 37

# Slot 8: VRECIP m2←m2
    vrecip.bf16 m2, m2      # m2,m3 = 1/rms
    DELAY 64

# Slot 9: VRECIP m8←m8
    vrecip.bf16 m8, m8      # m8,m9 = 1/rowsum
    DELAY 64

# Slot 10: VMUL m10←m0,m2
    vmul.bf16 m10, m0, m2  # m10,m11 = Y_x
    DELAY 64

# Slot 11: VMUL m8←m6,m8              (m6=exp, m8=1/rowsum — both intact ✓)
    vmul.bf16 m8, m6, m8   # m8,m9 = exp * (1/rowsum) = Y_z
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS
# ═══════════════════════════════════════════════════════════════
    CSRRS x21, 0xC00, x0        # x21 = cycle count at end
    SUB   x22, x21, x20         # x22 = elapsed cycles
    CSRRW x0,  0xC11, x22       # stash in CSR_DBG1

# ═══════════════════════════════════════════════════════════════
# WRITEBACK (not timed)
# ═══════════════════════════════════════════════════════════════

# ─── VSTORE all output tiles to VMEM ───────────────────────────
    LI    x7, 0x20001000        # Y_x staging: 0x20001000–0x200017FF (1 KiB aligned ✓)
    LI    x8, 0x20001800        # Y_z staging: 0x20001800–0x20001FFF (1 KiB aligned ✓)

    vstore m10, 0(x7)            # Y_x low  bank → VMEM 0x20001000
    DELAY 33
    vstore m11, 8(x7)            # Y_x high bank
    DELAY 33
    vstore m8, 0(x8)            # Y_z low  bank → VMEM 0x20001800
    DELAY 33
    vstore m9, 8(x8)            # Y_z high bank
    DELAY 33


# ─── DMA store all tiles to DRAM ───────────────────────────────
    LI    x3, 0x90001000
    dma.store x3, x7, x12, ch0   # VMEM 0x20001000 (Y_x) → DRAM 0x1000
    LI    x3, 0x90001800
    dma.store x3, x8, x12, ch1   # VMEM 0x20001800 (Y_z) → DRAM 0x1800
    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL
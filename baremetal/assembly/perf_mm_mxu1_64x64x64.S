# perf_mm_mxu1_64x64x64.S — single-MXU (M=64, K=64, N=64) tiled matmul perf test
# @TIMEOUT 400000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_perf_mm_mxu1_64x64x64.py 500
# @PERF_REPORT
# @PERF_MACS_MXU1 262144
#
# Full 64×64 × 64×64 = 64×64 FP8 matmul tiled into 32×32 blocks, using
# ONLY MXU1 (IPT). Pass/fail: DRAM output vs software-model golden
# (checks 0x2000–0x2FFF). Util reported (no threshold).
#
# ─────────────────────────────────────────────────────────────────────────
# Cadence — B-stationary tiling
# ─────────────────────────────────────────────────────────────────────────
# The IPT has effectively one push port (p1) usable concurrent with compute
# (p0). To keep p0 continuously feeding matmuls back-to-back, every PUSH
# must (a) go to p1, never p0, and (b) overlap with as much compute as
# possible.
#
# B-stationary tiling makes each pushed wslot serve TWO consecutive matmuls
# (one per A row-block), giving each PUSH a full 64-cycle compute window to
# overlap with. 4 PUSHes total (one per B tile) instead of 8, and 7 of 8
# matmul transitions run at stride 32 (true back-to-back issue).
#
# Matmul order:
#   M0: acc0 = A00·B00     (overwrite, wslot 0 = B00)
#   M1: acc1 = A10·B00     (overwrite, wslot 0 = B00)  ← reuses wslot 0
#   M2: acc0 += A01·B10    (wslot 1 = B10)
#   M3: acc1 += A11·B10    (wslot 1 = B10)             ← reuses wslot 1
#   POP C00 → m8, POP C10 → m10
#   M4: acc0 = A00·B01     (overwrite, wslot 0 = B01)
#   M5: acc1 = A10·B01     (overwrite, wslot 0 = B01)  ← reuses wslot 0
#   M6: acc0 += A01·B11    (wslot 1 = B11)
#   M7: acc1 += A11·B11    (wslot 1 = B11)             ← reuses wslot 1
#   POP C01 → m9, POP C11 → m11
#
# Push schedule (all on p1, never blocking p0):
#   cycle   1: PUSH wslot 0 ← B00   (for M0+M1)
#   cycle  35: PUSH wslot 1 ← B10   (for M2+M3, overlaps M0+M1 feed)
#   cycle  99: PUSH wslot 0 ← B01   (for M4+M5, overlaps M2+M3 feed)
#   cycle 132: PUSH wslot 1 ← B11   (for M6+M7, overlaps M4+M5 feed)
#
# Total measured region: 292 cycles. Util = 256/292 ≈ 87.7%.
# Theoretical floor: 292 cycles for this routine. We're AT the floor.
# Breakdown: initial PUSH (32 cyc, with M0 firing on PUSH's last p1Push
# cycle thanks to the strict pushWActive check) + matmul chain (224 cyc,
# 7 back-to-back stride-32 transitions) + M7 feed (32 cyc, but cmd-wise
# only +1 cycle for M7 dispatch) + WriteP0 drain between final POPs
# (32 cyc) + POP C11 dispatch + CSRR = 292.
#
# Tiling, MREG allocation, DRAM layout — same as MXU0 perf test.
#
# MREG allocation (m0–m11, all in distinct physical banks):
#   m0  A00   m4  B00_T   m8   C00 (final pop dest)
#   m1  A01   m5  B01_T   m9   C01
#   m2  A10   m6  B10_T   m10  C10
#   m3  A11   m7  B11_T   m11  C11
#
# DRAM layout (matches gen_perf_mm_mxu1_64x64x64.py):
#   0x0000  A00     0x1000  B00_T      0x2000  C00
#   0x0400  A01     0x1400  B01_T      0x2400  C01
#   0x0800  A10     0x1800  B10_T      0x2800  C10
#   0x0C00  A11     0x1C00  B11_T      0x2C00  C11

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127                    # scale reg 0 = 0x7F → unit scale

# ═══════════════════════════════════════════════════════════════
# SETUP: load all 8 input tiles DRAM → VMEM → MREG (NOT timed)
# ═══════════════════════════════════════════════════════════════

    # A00 → m0
    ADDI  x6, x0, 0
    LI    x10, 0x0000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33

    # A01 → m1
    ADDI  x6, x0, 0
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 33

    # A10 → m2
    ADDI  x6, x0, 0
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 2, x6, 0
    DELAY 33

    # A11 → m3
    ADDI  x6, x0, 0
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0
    DELAY 33

    # B00_T → m4
    ADDI  x6, x0, 0
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 33

    # B01_T → m5
    ADDI  x6, x0, 0
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 5, x6, 0
    DELAY 33

    # B10_T → m6
    ADDI  x6, x0, 0
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 33

    # B11_T → m7
    ADDI  x6, x0, 0
    LI    x10, 0x1C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 7, x6, 0
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS — snapshot start cycle right before first PUSH
# (matches MXU0 perf-test convention; first PUSH is part of timed region)
# ═══════════════════════════════════════════════════════════════
    CSRR  x20, 0xC00                # cycle 0

# ── Phase A: C00 (acc0) and C10 (acc1), both built from B00, B10 ──

    VMATPUSH.W.MXU1   0, 4          # cyc 1: wslot 0 ← B00 (for M0, M1); p1Push 2..32
    DELAY 30                        # cyc 2..32 — M0 fires on PUSH's last p1Push cycle
                                    # (cyc 33), valid per strict pushWActive check
                                    # (wbuf[col 31] latched at posedge 34, just in time
                                    # for M0's first feed at cyc 34).

    VMATMUL.MXU1      0, 0, 0       # cyc 33: M0  acc0 = A00·B00
    VMATPUSH.W.MXU1   1, 6          # cyc 34: wslot 1 ← B10 (for M2, M3); p1Push 35..66
    DELAY 29                        # cyc 35..64

    VMATMUL.MXU1      1, 2, 0       # cyc 65: M1  acc1 = A10·B00  [stride 32, same wslot]
    DELAY 30                        # cyc 66..96

    VMATMUL.ACC.MXU1  0, 1, 1       # cyc 97: M2  acc0 += A01·B10  [stride 32, wslot 1]
    VMATPUSH.W.MXU1   0, 5          # cyc 98: wslot 0 ← B01 (for M4, M5); p1Push 99..130
    DELAY 29                        # cyc 99..128

    VMATMUL.ACC.MXU1  1, 3, 1       # cyc 129: M3 acc1 += A11·B10  [stride 32, same wslot]
                                    # M3 reads wslot 1 = B10 cycles 130..161.
                                    # PUSH wslot 1 ← B11 must wait until M3 fully releases
                                    # wslot 1 (cycle ≥ 162, after M4 takes p0Comp).
    VMATPOP.FP8.MXU1  8, 0, 0       # cyc 130: m8 ← acc0 (C00 final)
    DELAY 29                        # cyc 131..160

# ── Phase B: C01 (acc0) and C11 (acc1), both built from B01, B11 ──

    VMATMUL.MXU1      0, 0, 0       # cyc 161: M4  acc0 = A00·B01  [stride 32, back-to-back]
    VMATPUSH.W.MXU1   1, 7          # cyc 162: wslot 1 ← B11 (for M6, M7); p1Push 163..194
                                    # Safe — M3 finished reading wslot 1 at cyc 161;
                                    # M4 reads wslot 0, doesn't touch wslot 1.
    VMATPOP.FP8.MXU1  10, 0, 1      # cyc 163: m10 ← acc1 (C10 final); w0Idl at 163
    DELAY 28                        # cyc 164..192

    VMATMUL.MXU1      1, 2, 0       # cyc 193: M5  acc1 = A10·B01  [stride 32, same wslot]
    DELAY 30                        # cyc 194..224

    VMATMUL.ACC.MXU1  0, 1, 1       # cyc 225: M6  acc0 += A01·B11 [stride 32, wslot 1]
    DELAY 30                        # cyc 226..256

    VMATMUL.ACC.MXU1  1, 3, 1       # cyc 257: M7  acc1 += A11·B11 [stride 32, same wslot]
    VMATPOP.FP8.MXU1  9, 0, 0       # cyc 258: m9 ← acc0 (C01 final)
    DELAY 31                        # cyc 259..290 — wait for WriteP0 to drain POP C01

    VMATPOP.FP8.MXU1  11, 0, 1      # cyc 291: m11 ← acc1 (C11 final)

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS — snapshot end cycle, bank delta in DBG1
# Note: CSRR x21 fires right after POP C11 issue, NOT after POP C11 fully
# drains to MREG. Convention matches MXU0 perf test.
# ═══════════════════════════════════════════════════════════════
    CSRR  x21, 0xC00                # cyc 292
    SUB   x22, x21, x20
    CSRW  x22, 0xC11

# ═══════════════════════════════════════════════════════════════
# WRITEBACK (not timed) — MREG → VMEM → DRAM for each output tile
# ═══════════════════════════════════════════════════════════════

    # C00 (m8) → DRAM 0x2000
    LI    x8, 0
    VSTORE 8, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x2000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C01 (m9) → DRAM 0x2400
    LI    x8, 0
    VSTORE 9, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x2400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C10 (m10) → DRAM 0x2800
    LI    x8, 0
    VSTORE 10, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x2800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C11 (m11) → DRAM 0x2C00
    LI    x8, 0
    VSTORE 11, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x2C00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE — host verifies DRAM 0x2000–0x2FFF; reads dbg1_cycles for util.
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

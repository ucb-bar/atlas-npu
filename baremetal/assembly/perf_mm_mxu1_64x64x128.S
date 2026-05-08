# perf_mm_mxu1_64x64x128.S — single-MXU (M=64, N=64, K=128) tiled matmul perf test
# @TIMEOUT 400000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_perf_mm_mxu1_64x64x128.py 500
# @PERF_REPORT
# @PERF_MACS_MXU1 524288
#
# Full 64×128 × 128×64 = 64×64 FP8 matmul tiled into 32×32 blocks, using
# ONLY MXU1 (IPT). Same B-stationary / wslot-pingpong / accSel-alternating
# cadence as perf_mm_mxu1_64x64x64.S, extended from K=64 (8 matmuls,
# 4 PUSHes) to K=128 (16 matmuls, 8 PUSHes). Pass/fail: DRAM output vs
# software-model golden (checks 0x4000–0x4FFF). Util reported (no threshold).
#
# ─────────────────────────────────────────────────────────────────────────
# Cadence — B-stationary tiling, 4 K-tiles per (i,j) output
# ─────────────────────────────────────────────────────────────────────────
# Same single-p1-push trick as the K=64 test: every PUSH goes to p1, never
# p0, and overlaps with as much compute as possible. Each pushed wslot
# serves TWO consecutive matmuls (one per A row-block i∈{0,1}), giving each
# PUSH a full 64-cycle compute window to overlap with. 8 PUSHes total
# (one per B tile), 15 of 16 matmul transitions run at stride 32 (true
# back-to-back issue, alternating accSel 0/1).
#
# Matmul order — Phase A builds C00 (acc0) + C10 (acc1) over k=0..3,
#                Phase B builds C01 (acc0) + C11 (acc1) over k=0..3:
#
#   Phase A:
#     M0:  acc0  = A00·B00     (overwrite, wslot 0 = B00)
#     M1:  acc1  = A10·B00     (overwrite, wslot 0 = B00)  ← reuses wslot 0
#     M2:  acc0 += A01·B10     (wslot 1 = B10)
#     M3:  acc1 += A11·B10     (wslot 1 = B10)             ← reuses wslot 1
#     M4:  acc0 += A02·B20     (wslot 0 = B20)
#     M5:  acc1 += A12·B20     (wslot 0 = B20)             ← reuses wslot 0
#     M6:  acc0 += A03·B30     (wslot 1 = B30)
#     M7:  acc1 += A13·B30     (wslot 1 = B30)             ← reuses wslot 1
#     POP C00 → m16 (mid-Phase B)
#     POP C10 → m18 (mid-Phase B)
#
#   Phase B:
#     M8:  acc0  = A00·B01     (overwrite, wslot 0 = B01)
#     M9:  acc1  = A10·B01     (overwrite, wslot 0 = B01)  ← reuses wslot 0
#     M10: acc0 += A01·B11     (wslot 1 = B11)
#     M11: acc1 += A11·B11     (wslot 1 = B11)             ← reuses wslot 1
#     M12: acc0 += A02·B21     (wslot 0 = B21)
#     M13: acc1 += A12·B21     (wslot 0 = B21)             ← reuses wslot 0
#     M14: acc0 += A03·B31     (wslot 1 = B31)
#     M15: acc1 += A13·B31     (wslot 1 = B31)             ← reuses wslot 1
#     POP C01 → m17
#     POP C11 → m19
#
# Push schedule (all on p1, never blocking p0):
#   cycle   1: PUSH wslot 0 ← B00  (for M0+M1)
#   cycle  34: PUSH wslot 1 ← B10  (for M2+M3, overlaps M0+M1 feed)
#   cycle  98: PUSH wslot 0 ← B20  (for M4+M5; M1 released wslot 0 at cyc 97)
#   cycle 162: PUSH wslot 1 ← B30  (for M6+M7; M3 released wslot 1 at cyc 161)
#   cycle 226: PUSH wslot 0 ← B01  (for M8+M9; M5 released wslot 0 at cyc 225)
#   cycle 290: PUSH wslot 1 ← B11  (for M10+M11; M7 released wslot 1 at cyc 289)
#   cycle 354: PUSH wslot 0 ← B21  (for M12+M13; M9 released wslot 0 at cyc 353)
#   cycle 418: PUSH wslot 1 ← B31  (for M14+M15; M11 released wslot 1 at cyc 417)
#
# Total measured region: 547 cycles. Util = 524288 / (1024 * 547) ≈ 93.6%.
# Theoretical floor: 547 cycles for this routine. We're AT the floor.
# Breakdown: initial PUSH (32 cyc, with M0 firing on PUSH's last p1Push
# cycle thanks to the strict pushWActive check) + 15 stride-32 matmul
# transitions (480 cyc) + M15 feed (32 cyc, but cmd-wise only +1 cyc for
# M15 dispatch) + WriteP0 drain between final POPs (33 cyc, POP→POP stride)
# + POP C11 dispatch + CSRR = 548.
#
# vs K=64 test: 16 matmuls × 32 cyc = 512 cyc compute (vs 256), but the
# fixed startup (PUSH + first-fire) + drain overhead is constant ~35 cyc,
# so util rises from 87.7% (K=64) → 93.6% (K=128). Doubling K halves the
# fractional setup overhead.
#
# MREG allocation (m0–m19, all in distinct physical banks):
#   m0  A00   m4  A02   m8   B00_T   m12  B20_T   m16  C00 (final pop dest)
#   m1  A01   m5  A03   m9   B01_T   m13  B21_T   m17  C01
#   m2  A10   m6  A12   m10  B10_T   m14  B30_T   m18  C10
#   m3  A11   m7  A13   m11  B11_T   m15  B31_T   m19  C11
#
# DRAM layout (matches gen_perf_mm_mxu1_64x64x128.py;
# A is i-major then k, B_T is k-major then j, C is i-major then j):
#   A:                              B_T:                       C:
#     0x0000  A00 (i=0,k=0)          0x2000  B00_T (k=0,j=0)    0x4000  C00
#     0x0400  A01 (i=0,k=1)          0x2400  B01_T (k=0,j=1)    0x4400  C01
#     0x0800  A02 (i=0,k=2)          0x2800  B10_T (k=1,j=0)    0x4800  C10
#     0x0C00  A03 (i=0,k=3)          0x2C00  B11_T (k=1,j=1)    0x4C00  C11
#     0x1000  A10 (i=1,k=0)          0x3000  B20_T (k=2,j=0)
#     0x1400  A11 (i=1,k=1)          0x3400  B21_T (k=2,j=1)
#     0x1800  A12 (i=1,k=2)          0x3800  B30_T (k=3,j=0)
#     0x1C00  A13 (i=1,k=3)          0x3C00  B31_T (k=3,j=1)

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127                    # scale reg 0 = 0x7F → unit scale

# ═══════════════════════════════════════════════════════════════
# SETUP: load all 16 input tiles DRAM → VMEM → MREG (NOT timed)
# ═══════════════════════════════════════════════════════════════

    # A00 → m0
    ADDI  x6, x0, 0
    LI    x10, 0x0000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    # A01 → m1
    ADDI  x6, x0, 0
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 40

    # A02 → m4
    ADDI  x6, x0, 0
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40

    # A03 → m5
    ADDI  x6, x0, 0
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 5, x6, 0
    DELAY 40

    # A10 → m2
    ADDI  x6, x0, 0
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 2, x6, 0
    DELAY 40

    # A11 → m3
    ADDI  x6, x0, 0
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0
    DELAY 40

    # A12 → m6
    ADDI  x6, x0, 0
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 40

    # A13 → m7
    ADDI  x6, x0, 0
    LI    x10, 0x1C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 7, x6, 0
    DELAY 40

    # B00_T → m8
    ADDI  x6, x0, 0
    LI    x10, 0x2000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 8, x6, 0
    DELAY 40

    # B01_T → m9
    ADDI  x6, x0, 0
    LI    x10, 0x2400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 9, x6, 0
    DELAY 40

    # B10_T → m10
    ADDI  x6, x0, 0
    LI    x10, 0x2800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 10, x6, 0
    DELAY 40

    # B11_T → m11
    ADDI  x6, x0, 0
    LI    x10, 0x2C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 11, x6, 0
    DELAY 40

    # B20_T → m12
    ADDI  x6, x0, 0
    LI    x10, 0x3000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 12, x6, 0
    DELAY 40

    # B21_T → m13
    ADDI  x6, x0, 0
    LI    x10, 0x3400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 13, x6, 0
    DELAY 40

    # B30_T → m14
    ADDI  x6, x0, 0
    LI    x10, 0x3800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 14, x6, 0
    DELAY 40

    # B31_T → m15
    ADDI  x6, x0, 0
    LI    x10, 0x3C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 15, x6, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS — snapshot start cycle right before first PUSH
# (matches MXU0/MXU1 perf-test convention; first PUSH is part of timed region)
# ═══════════════════════════════════════════════════════════════
    CSRR  x20, 0xC00                # cycle 0

# ── Phase A: C00 (acc0) and C10 (acc1), built from B00, B10, B20, B30 ──

    VMATPUSH.W.MXU1   0, 8          # cyc 1: wslot 0 ← B00 (for M0,M1); p1Push 2..33
    DELAY 30                        # cyc 2..32 — M0 fires on PUSH's last p1Push cycle
                                    # (cyc 33), valid per strict pushWActive check
                                    # (wbuf[col 31] latched at posedge 34, just in time
                                    # for M0's first feed at cyc 34).

    VMATMUL.MXU1      0, 0, 0       # cyc 33: M0  acc0 = A00·B00
    VMATPUSH.W.MXU1   1, 10         # cyc 34: wslot 1 ← B10 (for M2,M3); p1Push 35..66
    DELAY 29                        # cyc 35..64

    VMATMUL.MXU1      1, 2, 0       # cyc 65: M1  acc1 = A10·B00  [stride 32, same wslot]
    DELAY 30                        # cyc 66..96

    VMATMUL.ACC.MXU1  0, 1, 1       # cyc 97: M2  acc0 += A01·B10  [stride 32, wslot 1]
    VMATPUSH.W.MXU1   0, 12         # cyc 98: wslot 0 ← B20 (for M4,M5); p1Push 99..130
                                    # Safe — M1 released wslot 0 at cyc 97.
    DELAY 29                        # cyc 99..128

    VMATMUL.ACC.MXU1  1, 3, 1       # cyc 129: M3 acc1 += A11·B10  [stride 32, same wslot]
    DELAY 30                        # cyc 130..160

    VMATMUL.ACC.MXU1  0, 4, 0       # cyc 161: M4 acc0 += A02·B20  [stride 32, wslot 0]
    VMATPUSH.W.MXU1   1, 14         # cyc 162: wslot 1 ← B30 (for M6,M7); p1Push 163..194
                                    # Safe — M3 released wslot 1 at cyc 161; M4 reads wslot 0.
    DELAY 29                        # cyc 163..192

    VMATMUL.ACC.MXU1  1, 6, 0       # cyc 193: M5 acc1 += A12·B20  [stride 32, same wslot]
    DELAY 30                        # cyc 194..224

    VMATMUL.ACC.MXU1  0, 5, 1       # cyc 225: M6 acc0 += A03·B30  [stride 32, wslot 1]
    VMATPUSH.W.MXU1   0, 9          # cyc 226: wslot 0 ← B01 (for M8,M9); p1Push 227..258
                                    # Safe — M5 released wslot 0 at cyc 225; M6 reads wslot 1.
    DELAY 29                        # cyc 227..256

    VMATMUL.ACC.MXU1  1, 7, 1       # cyc 257: M7 acc1 += A13·B30  [stride 32, same wslot]
    VMATPOP.FP8.MXU1  16, 0, 0      # cyc 258: m16 ← acc0 (C00 final);
                                    # M6 row-0 acc0 written at cyc 258, accBufRow0Ready[0] OK.
    DELAY 29                        # cyc 259..288

# ── Phase B: C01 (acc0) and C11 (acc1), built from B01, B11, B21, B31 ──

    VMATMUL.MXU1      0, 0, 0       # cyc 289: M8  acc0 = A00·B01  [stride 32 from M7]
                                    # POP C00 still reading acc0 cyc 258..289; M8's first
                                    # acc0 row-0 write at cyc 322 — safe by 33-cyc margin.
    VMATPUSH.W.MXU1   1, 11         # cyc 290: wslot 1 ← B11 (for M10,M11); p1Push 291..322
                                    # Safe — M7 released wslot 1 at cyc 289; M8 reads wslot 0.
    VMATPOP.FP8.MXU1  18, 0, 1      # cyc 291: m18 ← acc1 (C10 final);
                                    # M7 row-0 acc1 written at cyc 290; w0Idl from cyc 290.
    DELAY 28                        # cyc 292..320

    VMATMUL.MXU1      1, 2, 0       # cyc 321: M9  acc1 = A10·B01  [stride 32, same wslot]
    DELAY 30                        # cyc 322..352

    VMATMUL.ACC.MXU1  0, 1, 1       # cyc 353: M10 acc0 += A01·B11  [stride 32, wslot 1]
    VMATPUSH.W.MXU1   0, 13         # cyc 354: wslot 0 ← B21 (for M12,M13); p1Push 355..386
                                    # Safe — M9 released wslot 0 at cyc 353.
    DELAY 29                        # cyc 355..384

    VMATMUL.ACC.MXU1  1, 3, 1       # cyc 385: M11 acc1 += A11·B11  [stride 32, same wslot]
    DELAY 30                        # cyc 386..416

    VMATMUL.ACC.MXU1  0, 4, 0       # cyc 417: M12 acc0 += A02·B21  [stride 32, wslot 0]
    VMATPUSH.W.MXU1   1, 15         # cyc 418: wslot 1 ← B31 (for M14,M15); p1Push 419..450
                                    # Safe — M11 released wslot 1 at cyc 417; M12 reads wslot 0.
    DELAY 29                        # cyc 419..448

    VMATMUL.ACC.MXU1  1, 6, 0       # cyc 449: M13 acc1 += A12·B21  [stride 32, same wslot]
    DELAY 30                        # cyc 450..480

    VMATMUL.ACC.MXU1  0, 5, 1       # cyc 481: M14 acc0 += A03·B31  [stride 32, wslot 1]
    DELAY 30                        # cyc 482..512

    VMATMUL.ACC.MXU1  1, 7, 1       # cyc 513: M15 acc1 += A13·B31  [stride 32, same wslot]
    VMATPOP.FP8.MXU1  17, 0, 0      # cyc 514: m17 ← acc0 (C01 final);
                                    # M14 row-0 acc0 written at cyc 514.
    DELAY 30                        # cyc 515..545 — POP→POP stride 32 (post-w0LastCycle
                                    # collapse: w0Boundary fires on the cycle row 31 writes,
                                    # letting next POP latch back-to-back with no bubble).

    VMATPOP.FP8.MXU1  19, 0, 1      # cyc 546: m19 ← acc1 (C11 final);
                                    # M15 row-0 acc1 written at cyc 546; w0 back-to-back.

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS — snapshot end cycle, bank delta in DBG1
# Note: CSRR x21 fires right after POP C11 issue, NOT after POP C11 fully
# drains to MREG. Convention matches MXU0/MXU1 K=64 perf tests.
# ═══════════════════════════════════════════════════════════════
    CSRR  x21, 0xC00                # cyc 547
    SUB   x22, x21, x20
    CSRW  x22, 0xC11

# ═══════════════════════════════════════════════════════════════
# WRITEBACK (not timed) — MREG → VMEM → DRAM for each output tile
# ═══════════════════════════════════════════════════════════════

    # C00 (m16) → DRAM 0x4000
    LI    x8, 0
    VSTORE 16, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x4000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C01 (m17) → DRAM 0x4400
    LI    x8, 0
    VSTORE 17, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x4400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C10 (m18) → DRAM 0x4800
    LI    x8, 0
    VSTORE 18, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x4800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C11 (m19) → DRAM 0x4C00
    LI    x8, 0
    VSTORE 19, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x4C00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE — host verifies DRAM 0x4000–0x4FFF; reads dbg1_cycles for util.
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

# perf_mm_mxu0_64x64x128.S — single-MXU (M=64, N=64, K=128) tiled matmul perf test
# @TIMEOUT 600000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_perf_mm_mxu0_64x64x128.py 500
# @PERF_REPORT
# @PERF_MACS_MXU0 524288
#
# Full 64×128 × 128×64 = 64×64 FP8 matmul tiled into 32×32 blocks, using
# ONLY MXU0. Compared to the K=64 sibling: K is doubled (4 K-tiles per
# output instead of 2), so the SA runs 16 matmuls + 4 pops. Pop overhead
# is amortized over twice the MAC work, so utilization rises (~91% vs
# ~86% for K=64). Pass/fail comes from comparing DRAM 0x4000–0x4FFF
# against the SA software-model golden.
#
# Timed region (delta written to CSR_DBG1 / MMIO 0x14):
#   starts AFTER every input tile (8×A + 8×B^T) is resident in MREG,
#   with a CSRR snapshot of CSR_CYCLE_COUNTER (0xC00) banked; ends
#   IMMEDIATELY after the last VMATPOP.FP8 lands in MREG, with a second
#   CSRR snapshot. The DRAM writeback (MREG → VMEM → DRAM) is explicitly
#   OUTSIDE the timed region. Util = MACs * 100 / (1024 * dbg1_cycles),
#   where 1024 = peak MAC/cycle for MXU0's 32×32 SA and total MACs for
#   this test = 16 * 32768 = 524288 (declared via @PERF_MACS_MXU0).
#
# Cadence: same 33-cycle stride as the K=64 test — dense same-wslot
# (PUSH+MATMUL+DELAY 30) pairs alternating accSel (acc0 / acc1) and
# wslot (wslot0 / wslot1) every beat. PUSH lane j writes weightSlot
# register at clock edge T_push+1+j; MATMUL's PE(0,j) reads weightSlot
# combinationally at T_M+1+j = T_push+2+j → exactly 1-cycle margin from
# the WeightBuffer's register write→read latency.
#
# Same-accSel reuse spacing is 64 cycles (stride 32 × 2 accSels) — at the
# drainAccConflict floor (T_new ≥ T_M_prev + 64). Post-FSM-collapse, the
# SA accepts back-to-back compute via p0Boundary, allowing the matmul
# chain to run at stride 32 (was 33). POP same-accSel obeys the same
# rule. Final POP→POP also runs at stride 32 (post-w0LastCycle collapse).
#
# Tiling (output-stationary over K, alternating acc0/acc1 each beat):
#   Phase A → C00 (acc0) and C01 (acc1), K=0..3
#     C00 = A00·B00 + A01·B10 + A02·B20 + A03·B30
#     C01 = A00·B01 + A01·B11 + A02·B21 + A03·B31
#   Phase B → C10 (acc0) and C11 (acc1), K=0..3 (after popping Phase A)
#     C10 = A10·B00 + A11·B10 + A12·B20 + A13·B30
#     C11 = A10·B01 + A11·B11 + A12·B21 + A13·B31
# First VMATMUL of each K-chain is overwrite-mode (VMATMUL.MXU0); the
# K=1..3 contributions use accumulate-mode (VMATMUL.ACC.MXU0). No zero-
# bias tile / VMATPUSH.ACC is needed.
#
# Cycle landmarks (M0 at cycle 2 after CSRR + first PUSH):
#   M_i (Phase A) at cycle 2 + 32*i for i in 0..7
#   POP acc0 (C00) at T_M6 + 64 = 258
#   M8  (Phase B C10 K=0) at 259     POP acc1 (C01) at T_M7+64 = 290
#   M9  (Phase B C11 K=0) at 291
#   M_i (Phase B) at cycle 323 + 32*(i-10) for i in 10..15
#   POP acc0 (C10) at T_M14 + 64 = 515
#   POP acc1 (C11) at T_M15 + 64 = 547
# Total measured region: 547 cycles. Util = 512/547 ≈ 93.6%.
#
# Phase A→B optimization: PUSH for next-phase weights fires INSIDE the
# drainAcc-gap DELAY (BEFORE the POP), freeing the slot right after POP
# for back-to-back matmul. Saves 1 cycle on the M7→M8 transition.
#
# Weights are PRE-TRANSPOSED offline by gen_perf_mm_mxu0_64x64x128.py so
# VMATPUSH.W consumes them directly — no VTRPOSE.XLU in the setup path.
#
# MREG allocation (all in distinct physical banks 0–19; no conflicts):
#   m0  A00     m4  A10     m8   B00_T (wslot0 K=0)   m16  C00 (pop dest)
#   m1  A01     m5  A11     m9   B01_T (wslot1 K=0)   m17  C01
#   m2  A02     m6  A12     m10  B10_T (wslot0 K=1)   m18  C10
#   m3  A03     m7  A13     m11  B11_T (wslot1 K=1)   m19  C11
#                           m12  B20_T (wslot0 K=2)
#                           m13  B21_T (wslot1 K=2)
#                           m14  B30_T (wslot0 K=3)
#                           m15  B31_T (wslot1 K=3)
#
# DRAM layout (must match gen_perf_mm_mxu0_64x64x128.py):
#   0x0000 A00     0x0800 A02     0x1000 A10     0x1800 A12   (1024B FP8 / tile)
#   0x0400 A01     0x0C00 A03     0x1400 A11     0x1C00 A13
#   0x2000 B00_T   0x2800 B10_T   0x3000 B20_T   0x3800 B30_T
#   0x2400 B01_T   0x2C00 B11_T   0x3400 B21_T   0x3C00 B31_T
#   0x4000 C00     0x4400 C01     0x4800 C10     0x4C00 C11   (output, checked)
#
# Register usage:
#   x5  DMA DRAM base (0x90000000)   x12  DMA size (1024 B/tile)
#   x6  VMEM input staging offset    x10  DRAM offset (scratch)
#   x8  VMEM output staging offset   x3   DRAM offset for writeback
#   x20 start-cycle snapshot         x21  end-cycle snapshot
#   x22 cycle delta (→ CSR_DBG1)     x30  completion signal

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127                    # scale reg 0 = 0x7F → unit scale

# ═══════════════════════════════════════════════════════════════
# SETUP: load all 16 input tiles DRAM → VMEM → MREG (NOT timed)
# ═══════════════════════════════════════════════════════════════

    # A00 → m0
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    # A01 → m1
    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 40

    # A02 → m2
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 2, x6, 0
    DELAY 40

    # A03 → m3
    ADDI  x6, x0, 0
    LI    x10, 0x90000C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0
    DELAY 40

    # A10 → m4
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40

    # A11 → m5
    ADDI  x6, x0, 0
    LI    x10, 0x90001400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 5, x6, 0
    DELAY 40

    # A12 → m6
    ADDI  x6, x0, 0
    LI    x10, 0x90001800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 40

    # A13 → m7
    ADDI  x6, x0, 0
    LI    x10, 0x90001C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 7, x6, 0
    DELAY 40

    # B00_T → m8
    ADDI  x6, x0, 0
    LI    x10, 0x90002000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 8, x6, 0
    DELAY 40

    # B01_T → m9
    ADDI  x6, x0, 0
    LI    x10, 0x90002400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 9, x6, 0
    DELAY 40

    # B10_T → m10
    ADDI  x6, x0, 0
    LI    x10, 0x90002800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 10, x6, 0
    DELAY 40

    # B11_T → m11
    ADDI  x6, x0, 0
    LI    x10, 0x90002C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 11, x6, 0
    DELAY 40

    # B20_T → m12
    ADDI  x6, x0, 0
    LI    x10, 0x90003000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 12, x6, 0
    DELAY 40

    # B21_T → m13
    ADDI  x6, x0, 0
    LI    x10, 0x90003400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 13, x6, 0
    DELAY 40

    # B30_T → m14
    ADDI  x6, x0, 0
    LI    x10, 0x90003800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 14, x6, 0
    DELAY 40

    # B31_T → m15
    ADDI  x6, x0, 0
    LI    x10, 0x90003C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 15, x6, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS — snapshot start cycle
# ═══════════════════════════════════════════════════════════════
    CSRR  x20, 0xC00                # x20 = cycle count at start

# ── Phase A: compute C00 (→acc0) and C01 (→acc1), K=0..3 ──
# Pattern: (push wK, matmul accN wK, DELAY 29) — alternating accSel/wslot
# every beat. Stride 32 cycles per matmul (post-FSM-collapse: SA gained
# back-to-back compute via p0Boundary). Same-accSel reuse stride is 64
# cycles = drainAccConflict minimum.

    # K=0
    VMATPUSH.W.MXU0   0, 8             # cyc 1: wslot0 ← B00_T
    VMATMUL.MXU0      0, 0, 0          # cyc 2: M0  acc0 = A00·B00      (overwrite)
    DELAY 29                           # cyc 3..32
    VMATPUSH.W.MXU0   1, 9             # cyc 33: wslot1 ← B01_T
    VMATMUL.MXU0      1, 0, 1          # cyc 34: M1  acc1 = A00·B01     [stride 32 from M0]
    DELAY 29                           # cyc 35..64
    # K=1
    VMATPUSH.W.MXU0   0, 10            # cyc 65: wslot0 ← B10_T
    VMATMUL.ACC.MXU0  0, 1, 0          # cyc 66: M2  acc0 += A01·B10    [stride 32, drainAcc 64]
    DELAY 29                           # cyc 67..96
    VMATPUSH.W.MXU0   1, 11            # cyc 97: wslot1 ← B11_T
    VMATMUL.ACC.MXU0  1, 1, 1          # cyc 98: M3  acc1 += A01·B11    [stride 32, drainAcc 64]
    DELAY 29                           # cyc 99..128
    # K=2
    VMATPUSH.W.MXU0   0, 12            # cyc 129: wslot0 ← B20_T
    VMATMUL.ACC.MXU0  0, 2, 0          # cyc 130: M4  acc0 += A02·B20   [stride 32]
    DELAY 29                           # cyc 131..160
    VMATPUSH.W.MXU0   1, 13            # cyc 161: wslot1 ← B21_T
    VMATMUL.ACC.MXU0  1, 2, 1          # cyc 162: M5  acc1 += A02·B21   [stride 32]
    DELAY 29                           # cyc 163..192
    # K=3
    VMATPUSH.W.MXU0   0, 14            # cyc 193: wslot0 ← B30_T
    VMATMUL.ACC.MXU0  0, 3, 0          # cyc 194: M6  acc0 += A03·B30   [stride 32]
    DELAY 29                           # cyc 195..224
    VMATPUSH.W.MXU0   1, 15            # cyc 225: wslot1 ← B31_T
    VMATMUL.ACC.MXU0  1, 3, 1          # cyc 226: M7  acc1 += A03·B31   [stride 32]
    DELAY 29                           # cyc 227..256

# ── Phase A → B: pre-PUSH next-phase weights INSIDE the drainAcc gap, then
#                 POP, then back-to-back matmul. Saves the M7→M8 transition
#                 by 1 cycle vs current POP-then-PUSH ordering.

    VMATPUSH.W.MXU0   0, 8             # cyc 257: wslot0 ← B00_T (Phase B K=0; cd[0]=0 at 257 ✓)
    VMATPOP.FP8.MXU0  16, 0, 0         # cyc 258: m16 ← acc0 (C00); drainAcc T_M6+64=258
    VMATMUL.MXU0      0, 4, 0          # cyc 259: M8  acc0 = A10·B00 (overwrite)
                                       #   stride 33 from M7; PUSH+2=259 ≤ 259 ✓
    DELAY 28                           # cyc 260..288 — POP m17 fires at T_M7+64 = 290
    VMATPUSH.W.MXU0   1, 9             # cyc 289: wslot1 ← B01_T (cd[1]=0 at 289 ✓)
    VMATPOP.FP8.MXU0  17, 0, 1         # cyc 290: m17 ← acc1 (C01); POP→POP stride 32
    VMATMUL.MXU0      1, 4, 1          # cyc 291: M9  acc1 = A10·B01
                                       #   stride 32 from M8 (back-to-back via p0Boundary) ✓
    DELAY 29                           # cyc 292..321

# ── Phase B K=1..3: accumulate ──
    # K=1
    VMATPUSH.W.MXU0   0, 10            # cyc 322: wslot0 ← B10_T
    VMATMUL.ACC.MXU0  0, 5, 0          # cyc 323: M10 acc0 += A11·B10   [stride 32]
    DELAY 29                           # cyc 324..353
    VMATPUSH.W.MXU0   1, 11            # cyc 354: wslot1 ← B11_T
    VMATMUL.ACC.MXU0  1, 5, 1          # cyc 355: M11 acc1 += A11·B11   [stride 32]
    DELAY 29                           # cyc 356..385
    # K=2
    VMATPUSH.W.MXU0   0, 12            # cyc 386: wslot0 ← B20_T
    VMATMUL.ACC.MXU0  0, 6, 0          # cyc 387: M12 acc0 += A12·B20   [stride 32]
    DELAY 29                           # cyc 388..417
    VMATPUSH.W.MXU0   1, 13            # cyc 418: wslot1 ← B21_T
    VMATMUL.ACC.MXU0  1, 6, 1          # cyc 419: M13 acc1 += A12·B21   [stride 32]
    DELAY 29                           # cyc 420..449
    # K=3
    VMATPUSH.W.MXU0   0, 14            # cyc 450: wslot0 ← B30_T
    VMATMUL.ACC.MXU0  0, 7, 0          # cyc 451: M14 acc0 += A13·B30   [stride 32]
    DELAY 29                           # cyc 452..481
    VMATPUSH.W.MXU0   1, 15            # cyc 482: wslot1 ← B31_T
    VMATMUL.ACC.MXU0  1, 7, 1          # cyc 483: M15 acc1 += A13·B31   [stride 32]
    DELAY 30                           # cyc 484..514 — POP m18 fires at T_M14+64 = 515

    VMATPOP.FP8.MXU0  18, 0, 0         # cyc 515: m18 ← acc0 (C10)
    DELAY 30                           # cyc 516..546 — POP→POP stride 32 (post-w0LastCycle
                                       # collapse) AND T_M15+64 = 547 reached at cyc 547
    VMATPOP.FP8.MXU0  19, 0, 1         # cyc 547: m19 ← acc1 (C11)

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS — snapshot end cycle, bank delta in DBG1
# ═══════════════════════════════════════════════════════════════
    CSRR  x21, 0xC00                # x21 = cycle count at end
    SUB   x22, x21, x20             # x22 = cycles inside timed window
    CSRW  x22, 0xC11                # stash delta in CSR_DBG1 (host reads @ 0x14)

# ═══════════════════════════════════════════════════════════════
# WRITEBACK (not timed) — MREG → VMEM → DRAM for each output tile
# ═══════════════════════════════════════════════════════════════

    # C00 (m16) → DRAM 0x4000
    LI    x8, 0
    VSTORE 16, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90004000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C01 (m17) → DRAM 0x4400
    LI    x8, 0
    VSTORE 17, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90004400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C10 (m18) → DRAM 0x4800
    LI    x8, 0
    VSTORE 18, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90004800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C11 (m19) → DRAM 0x4C00
    LI    x8, 0
    VSTORE 19, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90004C00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE — host verifies DRAM 0x4000–0x4FFF against golden; reads
#        mcycles delta (CSR_DBG1 @ 0x14) and computes util_mxu0
#        from the @PERF_MACS_MXU0 directive.
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

# perf_mm_mxu0_64x64x64.S — single-MXU (M=64, N=64, K=64) tiled matmul perf test
# @TIMEOUT 400000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_perf_mm_mxu0_64x64x64.py 500
# @PERF_REPORT
# @PERF_MACS_MXU0 262144
#
# Full 64×64 × 64×64 = 64×64 FP8 matmul tiled into 32×32 blocks, using
# ONLY MXU0. Pass/fail comes from comparing the DRAM output to the
# software-model golden (checks at 0x2000–0x2FFF). Utilization is
# reported for information only — no threshold is enforced.
#
# Timed region (delta written to CSR_DBG1 / MMIO 0x14):
#   starts AFTER every input tile (A00–A11, B00_T–B11_T) is resident in
#   MREG, with a CSRR snapshot of CSR_CYCLE_COUNTER (0xC00) banked;
#   ends IMMEDIATELY after the last VMATPOP.FP8 lands in MREG, with a
#   second CSRR snapshot. The DRAM writeback (MREG → VMEM → DRAM) is
#   explicitly OUTSIDE the timed region. Util = MACs * 100 / (1024 *
#   dbg1_cycles), where 1024 = peak MAC/cycle for MXU0's 32x32 SA and
#   total MACs for this test = 8 * 32768 = 262144 (declared via
#   @PERF_MACS_MXU0).
#
# Cadence: dense same-wslot push+matmul pairs at 32-cycle stride
# (PUSH+MATMUL+DELAY 29; DELAY N stalls N+1 cycles, so 1+1+30 = 32).
# Post-FSM-collapse, the SA accepts back-to-back compute via p0Boundary
# (a new compute fires on the cycle the previous compute finishes its
# last feed beat — no idle bubble). Push lane j writes weightSlot{0,1}(j)
# register at clock edge T_push+1+j; MATMUL's PE(0,j) reads weightSlot
# combinationally at T_M+1+j = T_push+2+j → exactly 1-cycle margin from
# the WeightBuffer's register write→read latency.
#
# Same-accSel MATMUL reuse spacing is 64 cycles (stride 32 × 2 accSels) —
# at the drainAccConflict floor: T_new ≥ T_M_prev + 64 (SA pipeline
# depth 62 + SRAM settle 1 + 1-cycle scalar issue → cmd at T_M+64 reads
# accBuf at T_M+65 which sees the new row 0 settled at edge T_M+63 →
# T_M+64). POP same-accSel obeys the same rule:
#   POP acc0 (C00) at T_M2+64 = 130.
#   POP acc1 (C01) at T_M3+64 = 162  (POP→POP stride 32 from POP acc0,
#                                      post-w0LastCycle collapse).
#   POP acc0 (C10) at T_M6+64 = 259.
#   POP acc1 (C11) at T_M7+64 = 291  (POP→POP stride 32).
# Total measured region: 291 cycles. Util = 256/291 ≈ 88.0%.
#
# Phase A→B optimization: each PUSH for the next-phase weights fires INSIDE
# the drainAcc-gap DELAY (BEFORE the POP) rather than after. This frees the
# slot right after the POP for back-to-back matmul issue, saving 1 cycle on
# the M3→M4 transition (and shifting all of Phase B + final POPs 1 cycle
# earlier).
#
# Tiling (output-stationary over K):
#   C00 = A00·B00 + A01·B10     C01 = A00·B01 + A01·B11
#   C10 = A10·B00 + A11·B10     C11 = A10·B01 + A11·B11
# First VMATMUL of each K-chain is overwrite-mode (VMATMUL.MXU0);
# the K=1 contribution uses accumulate-mode (VMATMUL.ACC.MXU0). No
# zero-bias tile / VMATPUSH.ACC is needed.
#
# Weights are PRE-TRANSPOSED offline by gen_perf_mm_mxu0_64x64x64.py so
# VMATPUSH.W consumes them directly — no VTRPOSE.XLU in the setup path.
#
# MREG allocation (all in distinct physical banks 0–11; no conflicts):
#   m0  A00              m4  B00_T                 m8   C00 result (pop dest)
#   m1  A01              m5  B01_T                 m9   C01 result
#   m2  A10              m6  B10_T                 m10  C10 result
#   m3  A11              m7  B11_T                 m11  C11 result
#
# DRAM layout (must match gen_perf_mm_mxu0_64x64x64.py):
#   0x0000  A00     (1024B FP8, 32×32)
#   0x0400  A01
#   0x0800  A10
#   0x0C00  A11
#   0x1000  B00_T
#   0x1400  B01_T
#   0x1800  B10_T
#   0x1C00  B11_T
#   0x2000  C00     (output, checked)
#   0x2400  C01
#   0x2800  C10
#   0x2C00  C11
#
# Register usage:
#   x5  DMA DRAM base (0x00000001)   x12  DMA size (1024 B/tile)
#   x6  VMEM input staging offset    x10  DRAM offset (scratch)
#   x8  VMEM output staging offset   x3   DRAM offset for writeback
#   x20 start-cycle snapshot         x21  end-cycle snapshot
#   x22 cycle delta (→ CSR_DBG1)     x30  completion signal

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127                    # scale reg 0 = 0x7F → unit scale

# ═══════════════════════════════════════════════════════════════
# SETUP: load all 8 input tiles DRAM → VMEM → MREG (NOT timed)
# ═══════════════════════════════════════════════════════════════

    # A00 → m0
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33

    # A01 → m1
    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 33

    # A10 → m2
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 2, x6, 0
    DELAY 33

    # A11 → m3
    ADDI  x6, x0, 0
    LI    x10, 0x90000C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0
    DELAY 33

    # B00_T → m4
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 33

    # B01_T → m5
    ADDI  x6, x0, 0
    LI    x10, 0x90001400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 5, x6, 0
    DELAY 33

    # B10_T → m6
    ADDI  x6, x0, 0
    LI    x10, 0x90001800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 33

    # B11_T → m7
    ADDI  x6, x0, 0
    LI    x10, 0x90001C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 7, x6, 0
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS — snapshot start cycle
# ═══════════════════════════════════════════════════════════════
    CSRR  x20, 0xC00                # x20 = cycle count at start

# ── Phase A: compute C00 (→acc0) and C01 (→acc1), K-stationary ──
# Pattern: (push wK, matmul accN wK, DELAY 29) — back-to-back matmul stride
# 32 (post-FSM-collapse: SA gained back-to-back compute via p0Boundary).
# Same-accSel reuse stride is 64 cycles (= drainAccConflict minimum).

    # K=0
    VMATPUSH.W.MXU0   0, 4            # cyc 1: wslot0 ← B00_T
    VMATMUL.MXU0      0, 0, 0         # cyc 2: M0  acc0 = A00·B00      (overwrite)
    DELAY 29                          # cyc 3..32
    VMATPUSH.W.MXU0   1, 5            # cyc 33: wslot1 ← B01_T
    VMATMUL.MXU0      1, 0, 1         # cyc 34: M1  acc1 = A00·B01     [stride 32 from M0]
    DELAY 29                          # cyc 35..64
    # K=1
    VMATPUSH.W.MXU0   0, 6            # cyc 65: wslot0 ← B10_T (cd[0]=0 ✓)
    VMATMUL.ACC.MXU0  0, 1, 0         # cyc 66: M2  acc0 += A01·B10    [stride 32, same-accSel
                                      #              stride 64 from M0 = drainAcc minimum]
    DELAY 29                          # cyc 67..96
    VMATPUSH.W.MXU0   1, 7            # cyc 97: wslot1 ← B11_T (cd[1]=0 ✓)
    VMATMUL.ACC.MXU0  1, 1, 1         # cyc 98: M3  acc1 += A01·B11    [stride 32, drainAcc 64]
    DELAY 29                          # cyc 99..128

# ── Phase A → B: pre-PUSH next-phase weights INSIDE the drainAcc gap, then
#                 POP, then back-to-back matmul. Saves the cycle that POP+PUSH
#                 between matmuls would otherwise cost.

    VMATPUSH.W.MXU0   0, 4            # cyc 129: wslot0 ← B00_T (Phase B K=0; cd[0]=0 at 129 ✓)
    VMATPOP.FP8.MXU0  8, 0, 0         # cyc 130: m8 ← acc0 (C00 = M0+M2); drainAcc T_M2+64
    VMATMUL.MXU0      0, 2, 0         # cyc 131: M4  acc0 = A10·B00 (overwrite)
                                      #   stride 33 from M3 (was 34); PUSH+2=131 ≤ 131 ✓
    DELAY 28                          # cyc 132..160 — POP m9 fires at T_M3+64 = 162
    VMATPUSH.W.MXU0   1, 5            # cyc 161: wslot1 ← B01_T (cd[1]=0 at 161 ✓)
    VMATPOP.FP8.MXU0  9, 0, 1         # cyc 162: m9 ← acc1 (C01); POP→POP stride 32 ✓
    VMATMUL.MXU0      1, 2, 1         # cyc 163: M5  acc1 = A10·B01
                                      #   stride 32 from M4 (back-to-back via p0Boundary) ✓
    DELAY 29                          # cyc 164..193

# ── Phase B K=1: accumulate ──
    VMATPUSH.W.MXU0   0, 6            # cyc 194: wslot0 ← B10_T (cd[0]=0 ✓)
    VMATMUL.ACC.MXU0  0, 3, 0         # cyc 195: M6  acc0 += A11·B10   [stride 32, drainAcc[0] 64]
    DELAY 29                          # cyc 196..225
    VMATPUSH.W.MXU0   1, 7            # cyc 226: wslot1 ← B11_T (cd[1]=0 ✓)
    VMATMUL.ACC.MXU0  1, 3, 1         # cyc 227: M7  acc1 += A11·B11   [stride 32, drainAcc[1] 64]
    DELAY 30                          # cyc 228..258 — POP m10 fires at T_M6+64 = 259

    VMATPOP.FP8.MXU0  10, 0, 0        # cyc 259: m10 ← acc0 (C10 = M4+M6)
    DELAY 30                          # cyc 260..290 — POP→POP stride 32 (post-w0LastCycle
                                      # collapse) AND T_M7+64 = 291 reached at cyc 291
    VMATPOP.FP8.MXU0  11, 0, 1        # cyc 291: m11 ← acc1 (C11 = M5+M7)

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS — snapshot end cycle, bank delta in DBG1
# ═══════════════════════════════════════════════════════════════
    CSRR  x21, 0xC00                # x21 = cycle count at end
    SUB   x22, x21, x20             # x22 = cycles inside timed window
    CSRW  x22, 0xC11                # stash delta in CSR_DBG1 (host reads @ 0x14)

# ═══════════════════════════════════════════════════════════════
# WRITEBACK (not timed) — MREG → VMEM → DRAM for each output tile
# ═══════════════════════════════════════════════════════════════

    # C00 (m8) → DRAM 0x2000
    LI    x8, 0
    VSTORE 8, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x90002000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C01 (m9) → DRAM 0x2400
    LI    x8, 0
    VSTORE 9, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x90002400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C10 (m10) → DRAM 0x2800
    LI    x8, 0
    VSTORE 10, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x90002800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C11 (m11) → DRAM 0x2C00
    LI    x8, 0
    VSTORE 11, x8, 0
    DELAY 33
    LI    x8, 0
    LI    x3, 0x90002C00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE — host verifies DRAM 0x2000–0x2FFF against golden; reads
#        mcycles delta (CSR_DBG1 @ 0x14) and computes util_mxu0
#        from the @PERF_MACS_MXU0 directive.
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

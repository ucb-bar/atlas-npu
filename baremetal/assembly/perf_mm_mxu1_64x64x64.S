# perf_mm_mxu1_64x64x64.S — single-MXU (M=64, K=64, N=64) tiled matmul perf test
# @TIMEOUT 400000
# @PYTHON_GEN gen_perf_mm_mxu1_64x64x64.py 500
# @PERF_REPORT
# @PERF_MACS_MXU1 262144
#
# Full 64×64 × 64×64 = 64×64 FP8 matmul tiled into 32×32 blocks, using
# ONLY MXU1. Pass/fail comes from comparing the DRAM output to the
# software-model golden (checks at 0x2000–0x2FFF). Utilization is
# reported for information only — no threshold is enforced.
#
# Timed region (delta written to CSR_DBG1 / MMIO 0x14):
#   starts AFTER every input tile (A00–A11, B00_T–B11_T) is resident in
#   MREG, with a CSRR snapshot of CSR_CYCLE_COUNTER (0xC00) banked;
#   ends IMMEDIATELY after the last VMATPOP.FP8 lands in MREG, with a
#   second CSRR snapshot. The DRAM writeback (MREG → VMEM → DRAM) is
#   explicitly OUTSIDE the timed region. Util = MACs * 100 / (1024 *
#   dbg1_cycles), where 1024 = peak MAC/cycle and total MACs for this
#   test = 8 * 32768 = 262144 (declared via @PERF_MACS_MXU1).
#
# The assembly uses fixed DELAY 32 between MXU ops — one Slot-A
# pipeline depth, the theoretical lower bound. Lower delays cannot
# produce hazards (the scheduler enforces dependencies); they would
# just stall and lower the reported util. No empirical tuning.
#
# Tiling (output-stationary over K):
#   C00 = A00·B00 + A01·B10     C01 = A00·B01 + A01·B11
#   C10 = A10·B00 + A11·B10     C11 = A10·B01 + A11·B11
# First VMATMUL of each K-chain is overwrite-mode (VMATMUL.MXU1);
# the K=1 contribution uses accumulate-mode (VMATMUL.ACC.MXU1). No
# zero-bias tile / VMATPUSH.ACC is needed.
#
# Weights are PRE-TRANSPOSED offline by gen_perf_mm_mxu1_64x64x64.py so
# VMATPUSH.W consumes them directly — no VTRPOSE.XLU in the setup path.
#
# MREG allocation (all in distinct physical banks 0–11; no conflicts):
#   m0  A00              m4  B00_T                 m8   C00 result (pop dest)
#   m1  A01              m5  B01_T                 m9   C01 result
#   m2  A10              m6  B10_T                 m10  C10 result
#   m3  A11              m7  B11_T                 m11  C11 result
#
# DRAM layout (must match gen_perf_mm_mxu1_64x64x64.py):
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
#   x5  DMA DRAM base (0x90000000)   x12  DMA size (1024 B/tile)
#   x6  VMEM input staging offset    x10  DRAM offset (scratch)
#   x8  VMEM output staging offset   x3   DRAM offset for writeback
#   x20 start-cycle snapshot         x21  end-cycle snapshot
#   x22 cycle delta (→ CSR_DBG1)     x30  completion signal

    LI    x5,  0x90000000
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
    DELAY 40

    # A01 → m1
    ADDI  x6, x0, 0
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 40

    # A10 → m2
    ADDI  x6, x0, 0
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 2, x6, 0
    DELAY 40

    # A11 → m3
    ADDI  x6, x0, 0
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0
    DELAY 40

    # B00_T → m4
    ADDI  x6, x0, 0
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40

    # B01_T → m5
    ADDI  x6, x0, 0
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 5, x6, 0
    DELAY 40

    # B10_T → m6
    ADDI  x6, x0, 0
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 40

    # B11_T → m7
    ADDI  x6, x0, 0
    LI    x10, 0x1C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 7, x6, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS — reset counters, snapshot start cycle
# ═══════════════════════════════════════════════════════════════
    CSRW  x0,  0xC20                # zero mxu0/mxu1/run perf counters
    CSRR  x20, 0xC00                # x20 = cycle count at start

# ── Phase A: compute C00 (→m8) and C01 (→m9), output-stationary over K ──

    # K=0: load B0x_T weights
    VMATPUSH.W.MXU1   0, 4          # wslot 0 ← B00_T  (→ Slot B)
    DELAY 32                        # let Slot B drain so push#2 also routes to B
    VMATPUSH.W.MXU1   1, 5          # wslot 1 ← B01_T  (→ Slot B)
    DELAY 32                        # weight buffer write completes before matmul

    VMATMUL.MXU1      0, 0, 0       # acc0 = A00 · B00    (overwrite)
    DELAY 32
    VMATMUL.MXU1      1, 0, 1       # acc1 = A00 · B01    (overwrite, ws=1)
    DELAY 96                        # drain slot A before re-push to wslot 1
                                    # (32-beat stream + 63-cycle pipe = 95 cyc; idle at 96)

    # K=1: overwrite wslots with B1x_T, then accumulate
    VMATPUSH.W.MXU1   0, 6          # wslot 0 ← B10_T
    DELAY 32
    VMATPUSH.W.MXU1   1, 7          # wslot 1 ← B11_T
    DELAY 32

    VMATMUL.ACC.MXU1  0, 1, 0       # acc0 += A01 · B10
    DELAY 32
    VMATMUL.ACC.MXU1  1, 1, 1       # acc1 += A01 · B11   (accSel=1)
    DELAY 96                        # drain accSel 0 and 1 before pops

    VMATPOP.FP8.MXU1  8, 0, 0       # m8  ← acc0  (C00), scale id 0
    DELAY 32
    VMATPOP.FP8.MXU1  9, 0, 1       # m9  ← acc1  (C01)
    DELAY 32

# ── Phase B: compute C10 (→m10) and C11 (→m11) ──

    # K=0: reload B0x_T (wslots got clobbered by Phase A K=1)
    VMATPUSH.W.MXU1   0, 4          # wslot 0 ← B00_T
    DELAY 32
    VMATPUSH.W.MXU1   1, 5          # wslot 1 ← B01_T
    DELAY 32

    VMATMUL.MXU1      0, 2, 0       # acc0 = A10 · B00    (overwrite)
    DELAY 32
    VMATMUL.MXU1      1, 2, 1       # acc1 = A10 · B01    (overwrite, ws=1)
    DELAY 96                        # drain slot A before re-push to wslot 1

    VMATPUSH.W.MXU1   0, 6          # wslot 0 ← B10_T
    DELAY 32
    VMATPUSH.W.MXU1   1, 7          # wslot 1 ← B11_T
    DELAY 32

    VMATMUL.ACC.MXU1  0, 3, 0       # acc0 += A11 · B10
    DELAY 32
    VMATMUL.ACC.MXU1  1, 3, 1       # acc1 += A11 · B11   (accSel=1)
    DELAY 96                        # drain accSel 0 and 1 before pops

    VMATPOP.FP8.MXU1  10, 0, 0      # m10 ← acc0  (C10)
    DELAY 32
    VMATPOP.FP8.MXU1  11, 0, 1      # m11 ← acc1  (C11)

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
    DELAY 40
    LI    x8, 0
    LI    x3, 0x2000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C01 (m9) → DRAM 0x2400
    LI    x8, 0
    VSTORE 9, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x2400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C10 (m10) → DRAM 0x2800
    LI    x8, 0
    VSTORE 10, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x2800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C11 (m11) → DRAM 0x2C00
    LI    x8, 0
    VSTORE 11, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x2C00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE — host verifies DRAM 0x2000–0x2FFF against golden; reads
#        mcycles delta (CSR_DBG1 @ 0x14) and computes util_mxu1
#        from the @PERF_MACS_MXU1 directive.
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

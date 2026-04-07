# mxu0_qk_scores.S — Q=X@WQ+bias, K=X@WK+bias, scores=Q@K^T  (all overlaps)
# @TIMEOUT 300000
# @PYTHON_GEN gen_mxu0_qk_scores.py 900
#
# Overlap patterns demonstrated:
#   [a] weight push + acc push:     prologue (Slot B + Slot A)
#   [b] compute + weight push:      every phase slab 1 (Slot A + Slot B)
#   [c] compute + acc push:         phases 2,4 slab 2 (Slot A compute + Slot B push)
#   [d] acc push + pop:             transitions 1→2, 3→4 (Slot A/B + Slot C)
#   [e] compute + weight push + pop: phase 3 slab 1 (Slot A + B + C)
#       (Phase 5 cannot use triple: popped K_n1 IS the weight input,
#        so pop must complete before weights can be loaded.)
#   [f] acc ping-pong:              phases alternate acc 0,1,0,1,0
#
# Zero bias (FP8 0x00 = 0.0) is pushed to each acc buffer before compute.
# MATMUL.ACC accumulates onto zero, so numerical results match unbiased test.
# This exercises the hardware overlap paths without changing expected values.
#
# Acc ping-pong allocation:
#   Phase 1  Q[:,0:32]  → acc 0
#   Phase 2  Q[:,32:64] → acc 1  │ pop acc 0 → TRF 8
#   Phase 3  K[:,0:32]  → acc 0  │ pop acc 1 → TRF 9
#   Phase 4  K[:,32:64] → acc 1  │ pop acc 0 → TRF 10
#   Phase 5  scores     → acc 0  │ pop acc 1 → TRF 11
#            final pop acc 0 → TRF 2,3 (BF16)
#
# TRF bank map:
#   0    activations (reloaded each slab)
#   4→5  weight tile A (raw→transposed)
#   6→7  weight tile B (raw→transposed)
#   8    Q[:,0:32]  pop result (FP8)
#   9    Q[:,32:64] pop result (FP8)
#   10   K[:,0:32]  pop result (FP8)
#   11   K[:,32:64] pop result (FP8)
#   12   zero bias  (FP8, persistent)
#   2,3  final BF16 scores
#
# DRAM layout (extends original with zero bias region):
#   0x0000  X[:,0:32]      (1024B)
#   0x0400  X[:,32:64]     (1024B)
#   0x0800  WQ_A           (1024B)
#   0x0C00  WQ_B           (1024B)
#   0x1000  WQ_C           (1024B)
#   0x1400  WQ_D           (1024B)
#   0x1800  WK_A           (1024B)
#   0x1C00  WK_B           (1024B)
#   0x2000  WK_C           (1024B)
#   0x2400  WK_D           (1024B)
#   0x2800  Q[:,0:32] store  (1024B)
#   0x2C00  Q[:,32:64] store (1024B)
#   0x3000  K[:,0:32] store  (1024B)
#   0x3400  K[:,32:64] store (1024B)
#   0x3800  (reserved)
#   0x3C00  (reserved)
#   0x4000  scores result lo (1024B)
#   0x4400  scores result hi (1024B)
#   0x4800  zero bias FP8    (1024B, all 0x00)

    LI    x5,  0x60000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    LI    x15, 1024
    SELI  0, 127          # scale reg 0 — identity E8M0 for FP8 pops
    SELI  1, 127          # scale reg 1 — identity E8M0

# ═══════════════════════════════════════════════════════════════
# GLOBAL SETUP: Load zero bias → TRF 12 (persistent for all phases)
# ═══════════════════════════════════════════════════════════════
    LI    x8, 512                   # VMEM staging addr
    LI    x10, 0x4800              # DRAM: zero bias FP8
    DMA.LOAD  x8, x10, x12, 3
    DMA.WAIT  3
    LI    x8, 512
    VLOAD 12, x8, 0

######################################################################
# Phase 1: Q[:,0:32] = X@WQ_A + X@WQ_B + bias → acc 0
#          First phase — nothing to pop yet.
# Overlaps: [a] weight push + acc push (prologue)
#           [b] compute + weight push (slab 1)
######################################################################

    # Load WQ_A → TRF 4 → transpose → TRF 5
    ADDI  x6, x0, 0
    LI    x10, 0x800
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4

    # ── OVERLAP [a]: weight push + acc push ──
    # Slot B: push weights → wbuf 0 from TRF 5
    # Slot A: push zero bias → acc 0 from TRF 12
    # Different push types → different submodule interfaces → safe.
    VMATPUSH.W.MXU0       0, 5
    VMATPUSH.ACC.FP8.MXU0 0, 12

    # Pre-load WQ_B → TRF 6 → transpose → TRF 7
    LI    x7, 256
    LI    x10, 0xC00
    DMA.LOAD  x7, x10, x12, 1
    DMA.WAIT  1
    LI    x7, 256
    VLOAD 6, x7, 0
    VTRPOSE.XLU 7, 6

    # Load X[:,0:32] → TRF 0
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0

    # ── OVERLAP [b]: compute + weight push (slab 1) ──
    # Slot A: MATMUL.ACC acc 0 (accumulates onto zero bias = same as MATMUL)
    # Slot B: push wbuf 1 ← TRF 7
    VMATMUL.ACC.MXU0  0, 0, 0
    VMATPUSH.W.MXU0   1, 7

    # Load X[:,32:64] → TRF 0
    ADDI  x6, x0, 0
    LI    x10, 0x400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0

    # Phase 1, slab 2: compute only (no extra overlap this phase)
    VMATMUL.ACC.MXU0  0, 0, 1
    # → acc 0 now holds Q[:,0:32]

######################################################################
# Transition 1→2: acc push (bias→acc1) + pop (acc0→TRF8)
# Overlap: [d] acc push + pop
######################################################################

    # ── OVERLAP [d]: acc push + pop ──
    # Slot A or B: push zero bias → acc 1
    # Slot C:      pop acc 0 → TRF 8, FP8 scale reg 0
    # Different acc buffers (push → acc 1, pop → acc 0). Safe.
    VMATPUSH.ACC.FP8.MXU0 1, 12
    VMATPOP.FP8.MXU0      8, 0, 0

######################################################################
# Phase 2: Q[:,32:64] → acc 1
# Bias already pushed at transition. Pop already started at transition.
# Overlaps: [b] compute + weight push (slab 1)
#           [c] compute + acc push (slab 2 — preloads bias→acc0 for phase 3)
######################################################################

    # Load WQ_C → TRF 4 → transpose → TRF 5, push to wbuf 0
    ADDI  x6, x0, 0
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    # Pre-load WQ_D → TRF 6 → transpose → TRF 7
    LI    x7, 256
    LI    x10, 0x1400
    DMA.LOAD  x7, x10, x12, 1
    DMA.WAIT  1
    LI    x7, 256
    VLOAD 6, x7, 0
    VTRPOSE.XLU 7, 6

    # Load X[:,0:32] → TRF 0
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0

    # ── OVERLAP [b]: compute + weight push (slab 1) ──
    VMATMUL.ACC.MXU0  1, 0, 0
    VMATPUSH.W.MXU0   1, 7

    # Load X[:,32:64] → TRF 0
    ADDI  x6, x0, 0
    LI    x10, 0x400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0

    # ── OVERLAP [c]: compute + acc push (slab 2) ──
    # Slot A: MATMUL.ACC on acc 1 (compute)
    # Slot B: push zero bias → acc 0 (preload for phase 3)
    # acc 1 ≠ acc 0 → no buffer conflict. Safe.
    VMATMUL.ACC.MXU0      1, 0, 1
    VMATPUSH.ACC.FP8.MXU0 0, 12
    # → acc 1 now holds Q[:,32:64]
    # → acc 0 now holds zero bias (ready for phase 3)

    # Store Q[:,0:32] from TRF 8 (pop completed long ago during DMA loads)
    LI    x8, 512
    VSTORE 8, x8, 0
    LI    x8, 512
    LI    x10, 0x2800
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

######################################################################
# Phase 3: K[:,0:32] → acc 0  |  pop acc 1 → TRF 9
# Bias for acc 0 already loaded during phase 2 slab 2 [c].
# Pop acc 1 deferred to here for triple overlap.
# Overlaps: [e] compute + weight push + pop (slab 1 — triple!)
######################################################################

    # Load WK_A → TRF 4 → transpose → TRF 5, push to wbuf 0
    ADDI  x6, x0, 0
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    LI    x7, 256
    LI    x10, 0x1C00
    DMA.LOAD  x7, x10, x12, 1
    DMA.WAIT  1
    LI    x7, 256
    VLOAD 6, x7, 0
    VTRPOSE.XLU 7, 6

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0

    # ── OVERLAP [e]: compute + weight push + pop (TRIPLE) ──
    # Slot A: MATMUL.ACC → acc 0
    # Slot B: PUSH.W → wbuf 1 ← TRF 7
    # Slot C: POP.FP8 acc 1 → TRF 9, scale reg 1
    # All three slots active. acc 0 ≠ acc 1, wbuf 1 ≠ wbuf 0. Safe.
    VMATMUL.ACC.MXU0  0, 0, 0
    VMATPUSH.W.MXU0   1, 7
    VMATPOP.FP8.MXU0  9, 1, 1

    ADDI  x6, x0, 0
    LI    x10, 0x400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0

    # Phase 3, slab 2: compute only
    VMATMUL.ACC.MXU0  0, 0, 1
    # → acc 0 now holds K[:,0:32]

    # Store Q[:,32:64] from TRF 9
    LI    x8, 512
    VSTORE 9, x8, 0
    LI    x8, 512
    LI    x10, 0x2C00
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

######################################################################
# Transition 3→4: acc push (bias→acc1) + pop (acc0→TRF10)
# Overlap: [d] acc push + pop
######################################################################

    # ── OVERLAP [d]: acc push + pop ──
    VMATPUSH.ACC.FP8.MXU0 1, 12
    VMATPOP.FP8.MXU0      10, 0, 0

######################################################################
# Phase 4: K[:,32:64] → acc 1
# Same structure as phase 2.
# Overlaps: [b] compute + weight push (slab 1)
#           [c] compute + acc push (slab 2 — preloads bias→acc0 for phase 5)
######################################################################

    ADDI  x6, x0, 0
    LI    x10, 0x2000
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    LI    x7, 256
    LI    x10, 0x2400
    DMA.LOAD  x7, x10, x12, 1
    DMA.WAIT  1
    LI    x7, 256
    VLOAD 6, x7, 0
    VTRPOSE.XLU 7, 6

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0

    # ── OVERLAP [b]: compute + weight push ──
    VMATMUL.ACC.MXU0  1, 0, 0
    VMATPUSH.W.MXU0   1, 7

    ADDI  x6, x0, 0
    LI    x10, 0x400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0

    # ── OVERLAP [c]: compute + acc push ──
    # Preload bias→acc 0 for phase 5 scores
    VMATMUL.ACC.MXU0      1, 0, 1
    VMATPUSH.ACC.FP8.MXU0 0, 12
    # → acc 1 holds K[:,32:64]
    # → acc 0 holds zero bias (ready for phase 5)

    # Store K[:,0:32] from TRF 10
    LI    x8, 512
    VSTORE 10, x8, 0
    LI    x8, 512
    LI    x10, 0x3000
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

######################################################################
# Phase 5: scores = Q @ K^T → acc 0
#   K tiles NOT transposed (pushing K directly as weights → Q @ K^T).
#   Bias for acc 0 already loaded during phase 4 slab 2.
#
#   DATA DEPENDENCY: K_n1 must be popped, stored to DRAM, then reloaded
#   as weight input. Triple overlap is NOT possible here because the pop
#   result IS the weight data. (Triple overlap is demonstrated in phase 3.)
#
# Overlaps: [b] compute + weight push (slab 1)
######################################################################

    # Pop K[:,32:64] from acc 1 → TRF 11 (MUST happen before DRAM loads)
    VMATPOP.FP8.MXU0  11, 1, 1

    # Load K[:,0:32] result → TRF 4, push to wbuf 0 (no transpose)
    # Pop runs concurrently with this DMA+push (~50 cycles > 32 cycle pop)
    ADDI  x6, x0, 0
    LI    x10, 0x3000
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VMATPUSH.W.MXU0  0, 4

    # Store K[:,32:64] from TRF 11 → DRAM (pop finished during load above)
    LI    x8, 512
    VSTORE 11, x8, 0
    LI    x8, 512
    LI    x10, 0x3400
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

    # Load K[:,32:64] from DRAM → TRF 6 (now correct data is there)
    ADDI  x6, x0, 0
    LI    x10, 0x3400
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0

    # Load Q[:,0:32] activations → TRF 0
    LI    x7, 256
    LI    x10, 0x2800
    DMA.LOAD  x7, x10, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0

    # ── OVERLAP [b]: compute + weight push ──
    VMATMUL.ACC.MXU0  0, 0, 0
    VMATPUSH.W.MXU0   1, 6

    # Load Q[:,32:64] activations → TRF 0
    LI    x7, 256
    LI    x10, 0x2C00
    DMA.LOAD  x7, x10, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0

    # Phase 5, slab 2: accumulate
    VMATMUL.ACC.MXU0  0, 0, 1

######################################################################
# Final pop: scores from acc 0 (BF16)
######################################################################
    VMATPOP.BF16.MXU0  2, 0

    LI    x8, 512
    VSTORE 2, x8, 0
    LI    x9, 768
    VSTORE 3, x9, 0

    LI    x8, 512
    LI    x10, 0x4000
    DMA.STORE x10, x8, x15, 2
    DMA.WAIT  2

    LI    x9, 768
    LI    x10, 0x4400
    DMA.STORE x10, x9, x15, 3
    DMA.WAIT  3

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL
    
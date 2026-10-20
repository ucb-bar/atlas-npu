# mxu1_qk_scores.S — Q=X@WQ+bias, K=X@WK+bias, scores=Q@K^T  (all overlaps)
# @TIMEOUT 300000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu1_qk_scores.py 900
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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024
    LI    x15, 1024
    seli e0, 127          # scale reg 0 — identity E8M0 for FP8 pops
    seli e1, 127          # scale reg 1 — identity E8M0

# ═══════════════════════════════════════════════════════════════
# GLOBAL SETUP: Load zero bias → TRF 12 (persistent for all phases)
# ═══════════════════════════════════════════════════════════════
    LI    x8, 512                   # VMEM staging addr
    LI    x10, 0x90004800  # DRAM: zero bias FP8
    dma.load x8, x10, x12, ch3
    dma.wait ch3
    LI    x8, 512
    vload m12, 0(x8)

######################################################################
# Phase 1: Q[:,0:32] = X@WQ_A + X@WQ_B + bias → acc 0
#          First phase — nothing to pop yet.
# Overlaps: [a] weight push + acc push (prologue)
#           [b] compute + weight push (slab 1)
######################################################################

    # Load WQ_A → TRF 4 → transpose → TRF 5
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    # Wait for VLOAD to finish writing m4 before XLU reads it for transpose.
    DELAY 33
    vtrpose.xlu m5, m4
    # Wait for XLU transpose to finish writing m5 before MXU reads it.
    DELAY 64

    # ── OVERLAP [a]: weight push + acc push ──
    # Slot B: push weights → wbuf 0 from TRF 5
    # Slot A: push zero bias → acc 0 from TRF 12
    # Different push types → different submodule interfaces → safe.
    vmatpush.weight.mxu1 w0, m5
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpush.acc.fp8.mxu1 acc0, m12

    # Pre-load WQ_B → TRF 6 → transpose → TRF 7
    LI    x7, 256
    LI    x10, 0x90000C00
    dma.load x7, x10, x12, ch1
    dma.wait ch1
    LI    x7, 256
    vload m6, 0(x7)
    # Wait for VLOAD to finish writing m6 before XLU reads it for transpose.
    DELAY 33
    vtrpose.xlu m7, m6
    # Wait for XLU transpose to finish writing m7 before MXU reads it.
    DELAY 64

    # Load X[:,0:32] → TRF 0
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # ── OVERLAP [b]: compute + weight push (slab 1) ──
    # Slot A: MATMUL.ACC acc 0 (accumulates onto zero bias = same as MATMUL)
    # Slot B: push wbuf 1 ← TRF 7
    vmatmul.acc.mxu1 acc0, m0, w0
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpush.weight.mxu1 w1, m7

    # Load X[:,32:64] → TRF 0
    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # Phase 1, slab 2: compute only (no extra overlap this phase)
    vmatmul.acc.mxu1 acc0, m0, w1
    # Wait for matmul pipeline to finish writing acc 0 before pop.
    DELAY 34
    # → acc 0 now holds Q[:,0:32]

######################################################################
# Transition 1→2: acc push (bias→acc1) + pop (acc0→TRF8)
# Overlap: [d] acc push + pop
######################################################################

    # ── OVERLAP [d]: acc push + pop ──
    # Slot A or B: push zero bias → acc 1
    # Slot C:      pop acc 0 → TRF 8, FP8 scale reg 0
    # Different acc buffers (push → acc 1, pop → acc 0). Safe.
    vmatpush.acc.fp8.mxu1 acc1, m12
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpop.fp8.acc.mxu1 m8, acc0, e0

######################################################################
# Phase 2: Q[:,32:64] → acc 1
# Bias already pushed at transition. Pop already started at transition.
# Overlaps: [b] compute + weight push (slab 1)
#           [c] compute + acc push (slab 2 — preloads bias→acc0 for phase 3)
######################################################################

    # Load WQ_C → TRF 4 → transpose → TRF 5, push to wbuf 0
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    # Wait for VLOAD to finish writing m4 before XLU reads it for transpose.
    DELAY 33
    vtrpose.xlu m5, m4
    # Wait for XLU transpose to finish writing m5 before MXU reads it.
    DELAY 64
    vmatpush.weight.mxu1 w0, m5

    # Pre-load WQ_D → TRF 6 → transpose → TRF 7
    LI    x7, 256
    LI    x10, 0x90001400
    dma.load x7, x10, x12, ch1
    dma.wait ch1
    LI    x7, 256
    vload m6, 0(x7)
    # Wait for VLOAD to finish writing m6 before XLU reads it for transpose.
    DELAY 33
    vtrpose.xlu m7, m6
    # Wait for XLU transpose to finish writing m7 before MXU reads it.
    DELAY 64

    # Load X[:,0:32] → TRF 0
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # ── OVERLAP [b]: compute + weight push (slab 1) ──
    vmatmul.acc.mxu1 acc1, m0, w0
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpush.weight.mxu1 w1, m7

    # Load X[:,32:64] → TRF 0
    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # ── OVERLAP [c]: compute + acc push (slab 2) ──
    # Slot A: MATMUL.ACC on acc 1 (compute)
    # Slot B: push zero bias → acc 0 (preload for phase 3)
    # acc 1 ≠ acc 0 → no buffer conflict. Safe.
    vmatmul.acc.mxu1 acc1, m0, w1
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpush.acc.fp8.mxu1 acc0, m12
    # → acc 1 now holds Q[:,32:64]
    # → acc 0 now holds zero bias (ready for phase 3)

    # Store Q[:,0:32] from TRF 8 (pop completed long ago during DMA loads)
    LI    x8, 512
    vstore m8, 0(x8)
    # Wait for VSTORE to finish before DMA reads VMEM for store-back.
    DELAY 33
    LI    x8, 512
    LI    x10, 0x90002800
    dma.store x10, x8, x12, ch2
    dma.wait ch2

######################################################################
# Phase 3: K[:,0:32] → acc 0  |  pop acc 1 → TRF 9
# Bias for acc 0 already loaded during phase 2 slab 2 [c].
# Pop acc 1 deferred to here for triple overlap.
# Overlaps: [e] compute + weight push + pop (slab 1 — triple!)
######################################################################

    # Load WK_A → TRF 4 → transpose → TRF 5, push to wbuf 0
    ADDI  x6, x0, 0
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    # Wait for VLOAD to finish writing m4 before XLU reads it for transpose.
    DELAY 33
    vtrpose.xlu m5, m4
    # Wait for XLU transpose to finish writing m5 before MXU reads it.
    DELAY 64
    vmatpush.weight.mxu1 w0, m5

    LI    x7, 256
    LI    x10, 0x90001C00
    dma.load x7, x10, x12, ch1
    dma.wait ch1
    LI    x7, 256
    vload m6, 0(x7)
    # Wait for VLOAD to finish writing m6 before XLU reads it for transpose.
    DELAY 33
    vtrpose.xlu m7, m6
    # Wait for XLU transpose to finish writing m7 before MXU reads it.
    DELAY 64

    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # ── OVERLAP [e]: compute + weight push + pop (TRIPLE) ──
    # Slot A: MATMUL.ACC → acc 0
    # Slot B: PUSH.W → wbuf 1 ← TRF 7
    # Slot C: POP.FP8 acc 1 → TRF 9, scale reg 1
    # All three slots active. acc 0 ≠ acc 1, wbuf 1 ≠ wbuf 0. Safe.
    vmatmul.acc.mxu1 acc0, m0, w0
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpush.weight.mxu1 w1, m7
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpop.fp8.acc.mxu1 m9, acc1, e1

    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # Phase 3, slab 2: compute only
    vmatmul.acc.mxu1 acc0, m0, w1
    # Wait for matmul pipeline to finish writing acc 0 before pop.
    DELAY 34
    # → acc 0 now holds K[:,0:32]

    # Store Q[:,32:64] from TRF 9
    LI    x8, 512
    vstore m9, 0(x8)
    # Wait for VSTORE to finish before DMA reads VMEM for store-back.
    DELAY 33
    LI    x8, 512
    LI    x10, 0x90002C00
    dma.store x10, x8, x12, ch2
    dma.wait ch2

######################################################################
# Transition 3→4: acc push (bias→acc1) + pop (acc0→TRF10)
# Overlap: [d] acc push + pop
######################################################################

    # ── OVERLAP [d]: acc push + pop ──
    vmatpush.acc.fp8.mxu1 acc1, m12
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpop.fp8.acc.mxu1 m10, acc0, e0

######################################################################
# Phase 4: K[:,32:64] → acc 1
# Same structure as phase 2.
# Overlaps: [b] compute + weight push (slab 1)
#           [c] compute + acc push (slab 2 — preloads bias→acc0 for phase 5)
######################################################################

    ADDI  x6, x0, 0
    LI    x10, 0x90002000
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    # Wait for VLOAD to finish writing m4 before XLU reads it for transpose.
    DELAY 33
    vtrpose.xlu m5, m4
    # Wait for XLU transpose to finish writing m5 before MXU reads it.
    DELAY 64
    vmatpush.weight.mxu1 w0, m5

    LI    x7, 256
    LI    x10, 0x90002400
    dma.load x7, x10, x12, ch1
    dma.wait ch1
    LI    x7, 256
    vload m6, 0(x7)
    # Wait for VLOAD to finish writing m6 before XLU reads it for transpose.
    DELAY 33
    vtrpose.xlu m7, m6
    # Wait for XLU transpose to finish writing m7 before MXU reads it.
    DELAY 64

    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # ── OVERLAP [b]: compute + weight push ──
    vmatmul.acc.mxu1 acc1, m0, w0
    # Back-to-back issue is intentional here; these ops overlap safely on MXU1.
    vmatpush.weight.mxu1 w1, m7

    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # ── OVERLAP [c]: compute + acc push ──
    # Preload bias→acc 0 for phase 5 scores
    vmatmul.acc.mxu1 acc1, m0, w1
    vmatpush.acc.fp8.mxu1 acc0, m12
    # → acc 1 holds K[:,32:64]
    # → acc 0 holds zero bias (ready for phase 5)

    # Store K[:,0:32] from TRF 10
    LI    x8, 512
    vstore m10, 0(x8)
    # Wait for VSTORE to finish before DMA reads VMEM for store-back.
    DELAY 33
    LI    x8, 512
    LI    x10, 0x90003000
    dma.store x10, x8, x12, ch2
    dma.wait ch2

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
    vmatpop.fp8.acc.mxu1 m11, acc1, e1

    # Load K[:,0:32] result → TRF 4, push to wbuf 0 (no transpose)
    # Pop runs concurrently with this DMA+push (~50 cycles > 32 cycle pop)
    ADDI  x6, x0, 0
    LI    x10, 0x90003000
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    # Wait for VLOAD to finish writing m4 before MXU reads weights from it.
    DELAY 33
    vmatpush.weight.mxu1 w0, m4

    # Store K[:,32:64] from TRF 11 → DRAM (pop finished during load above)
    LI    x8, 512
    vstore m11, 0(x8)
    # Wait for VSTORE to finish before DMA reads VMEM for store-back.
    DELAY 33
    LI    x8, 512
    LI    x10, 0x90003400
    dma.store x10, x8, x12, ch2
    dma.wait ch2

    # Load K[:,32:64] from DRAM → TRF 6 (now correct data is there)
    ADDI  x6, x0, 0
    LI    x10, 0x90003400
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m6, 0(x6)
    # Wait for VLOAD to finish writing m6 before MXU reads weights from it.
    DELAY 33

    # Load Q[:,0:32] activations → TRF 0
    LI    x7, 256
    LI    x10, 0x90002800
    dma.load x7, x10, x12, ch0
    dma.wait ch0
    LI    x7, 256
    vload m0, 0(x7)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # ── OVERLAP [b]: compute + weight push ──
    vmatmul.acc.mxu1 acc0, m0, w0
    vmatpush.weight.mxu1 w1, m6

    # Load Q[:,32:64] activations → TRF 0
    LI    x7, 256
    LI    x10, 0x90002C00
    dma.load x7, x10, x12, ch0
    dma.wait ch0
    LI    x7, 256
    vload m0, 0(x7)
    # Wait for VLOAD to finish writing m0 before MXU reads activations.
    DELAY 33

    # Phase 5, slab 2: accumulate
    vmatmul.acc.mxu1 acc0, m0, w1

######################################################################
# Final pop: scores from acc 0 (BF16)
######################################################################
    # Wait for the final matmul to finish before popping BF16 scores.
    DELAY 34
    vmatpop.bf16.acc.mxu1 m2, acc0
    # Wait for pop to finish writing m2/m3 before LSU reads them (VSTORE).
    DELAY 31

    LI    x8, 512
    vstore m2, 0(x8)
    # LSU is busy during VSTORE; delay before issuing the next VSTORE.
    DELAY 33
    LI    x9, 768
    vstore m3, 0(x9)
    # Wait for VSTORE to finish before DMA reads VMEM for store-back.
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90004000
    dma.store x10, x8, x15, ch2
    dma.wait ch2

    LI    x9, 768
    LI    x10, 0x90004400
    dma.store x10, x9, x15, ch3
    dma.wait ch3

    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL

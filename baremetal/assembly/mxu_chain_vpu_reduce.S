# mxu_chain_vpu_reduce.S — MXU0→FP8→MXU1 chain + VPU post-process
# @TIMEOUT 1000000
# @PYTHON_GEN gen_mxu_chain_vpu_reduce.py 333
#
# Attention-like pipeline exercising all three functional units:
#
#   Phase A: MXU0 computes Q = X @ WQ   (32×32, BF16 acc → FP8 pop)
#   Phase B: MXU1 computes K = X @ WK   (32×32, BF16 acc → FP8 pop)
#   Phase C: MXU0 computes S = Q @ K^T  (32×32, BF16 pop — final scores)
#   Phase D: VPU computes bias_add = S + bias_vpu  (BF16 element-wise)
#
# VPU ops run interleaved during DMA/transpose stalls:
#   VPU-1: vmin(A, B) during Phase A weight prep
#   VPU-2: vadd(S, bias) after Phase C (cross-unit dependency)
#
# Overlap patterns:
#   [a] MXU0 w-push + acc-push (Phase A prologue)
#   [b] MXU0 compute + MXU0 w-push (not used — single k-tile)
#   [c] MXU1 w-push + acc-push (Phase B prologue)
#   [d] MXU0 FP8 pop + VPU vmin (concurrent pop + VPU, Phase A→B)
#   [e] MXU1 FP8 pop + DMA store (pop while storing Q, Phase B→C)
#   [f] MXU0 compute (Phase C) + DMA load (K for wbuf)
#   [g] 4-channel DMA parallel (initial load)
#   [h] VPU vadd + DMA store (storing scores while VPU processes)
#
# TRF bank map (all 16 banks used):
#   m0    activations / Q for Phase C (FP8, reloaded)
#   m1    weight raw scratch (XLU input)
#   m2    weight transposed (XLU output)
#   m3    zero bias (persistent)
#   m4    Q pop result (FP8, from MXU0)
#   m5    K pop result (FP8, from MXU1)
#   m6,7  BF16 scores pop (from MXU0 Phase C)
#   m8,9  VPU input A (BF16, for vmin)
#   m10,11 VPU input B (BF16, for vmin)
#   m12,13 VPU vmin output / VPU vadd output
#   m14,15 bias_vpu (BF16, for post-process vadd)
#
# DRAM layout:
#   0x0000  X          (1024B FP8)
#   0x0400  WQ         (1024B FP8)
#   0x0800  WK         (1024B FP8)
#   0x0C00  zero bias  (1024B FP8, all 0x00)
#   0x1000  A_vpu b0   (1024B BF16)
#   0x1400  A_vpu b1   (1024B BF16)
#   0x1800  B_vpu b0   (1024B BF16)
#   0x1C00  B_vpu b1   (1024B BF16)
#   0x2000  bias_vpu b0 (1024B BF16)
#   0x2400  bias_vpu b1 (1024B BF16)
#   — scratch (used for Q, K round-trip) —
#   0x2800  Q store    (1024B FP8)
#   0x2C00  K store    (1024B FP8)
#   — outputs —
#   0x3000  scores lo  (1024B BF16)
#   0x3400  scores hi  (1024B BF16)
#   0x3800  vmin b0    (1024B BF16)
#   0x3C00  vmin b1    (1024B BF16)
#   0x4000  biased b0  (1024B BF16) — vadd(scores, bias_vpu)
#   0x4400  biased b1  (1024B BF16)

    LI    x5,  0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127          # identity scale for FP8 pops
    SELI  1, 127

# ═══════════════════════════════════════════════════════════════
# GLOBAL SETUP: Load zero bias + VPU inputs + bias_vpu
# Use all 4 DMA channels for parallel initial load [g]
# ═══════════════════════════════════════════════════════════════

    # ch0: zero bias → VMEM[0]
    ADDI  x6, x0, 0
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 0

    # ch1: A_vpu b0 → VMEM[256]
    LI    x7, 256
    LI    x10, 0x1000
    DMA.LOAD  x7, x10, x12, 1

    # ch2: B_vpu b0 → VMEM[512]
    LI    x8, 512
    LI    x10, 0x1800
    DMA.LOAD  x8, x10, x12, 2

    # ch3: bias_vpu b0 → VMEM[768]
    LI    x9, 768
    LI    x10, 0x2000
    DMA.LOAD  x9, x10, x12, 3

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    # VLOAD first batch
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0                 # m3 ← zero bias
    DELAY 40
    LI    x7, 256
    VLOAD 8, x7, 0                 # m8 ← A_vpu b0
    DELAY 40
    LI    x8, 512
    VLOAD 10, x8, 0                # m10 ← B_vpu b0
    DELAY 40
    LI    x9, 768
    VLOAD 14, x9, 0                # m14 ← bias_vpu b0
    DELAY 40

    # Second round: bank1 data
    # ch0: A_vpu b1
    ADDI  x6, x0, 0
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 0

    # ch1: B_vpu b1
    LI    x7, 256
    LI    x10, 0x1C00
    DMA.LOAD  x7, x10, x12, 1

    # ch2: bias_vpu b1
    LI    x8, 512
    LI    x10, 0x2400
    DMA.LOAD  x8, x10, x12, 2

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2

    ADDI  x6, x0, 0
    VLOAD 9, x6, 0                 # m9 ← A_vpu b1
    DELAY 40
    LI    x7, 256
    VLOAD 11, x7, 0                # m11 ← B_vpu b1
    DELAY 40
    LI    x8, 512
    VLOAD 15, x8, 0                # m15 ← bias_vpu b1
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE A: MXU0 computes Q = X @ WQ → acc 0 → FP8 pop → m4
# ═══════════════════════════════════════════════════════════════

    # Load WQ → m1 → XLU → m2
    ADDI  x6, x0, 0
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 40
    VTRPOSE.XLU 2, 1
    DELAY 80

    # ── [a] MXU0 w-push + acc-push ──
    VMATPUSH.W.MXU0       0, 2
    VMATPUSH.ACC.FP8.MXU0 0, 3

    # Load X → m0
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    # MXU0 compute Q
    VMATMUL.ACC.MXU0  0, 0, 0
    DELAY 120

    # ── [d] MXU0 FP8 pop + VPU vmin (concurrent) ──
    # MXU0 pops acc 0 → m4 (FP8). VPU computes vmin on m8,m10 → m12.
    # Different functional units, different TRF banks. Safe.
    VMATPOP.FP8.MXU0  4, 0, 0
    VMIN.BF16         12, 8, 10
    DELAY 150

    # Store Q (m4) → DRAM scratch (need it for Phase C)
    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 40
    LI    x8, 512
    LI    x10, 0x2800
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

# ═══════════════════════════════════════════════════════════════
# PHASE B: MXU1 computes K = X @ WK → acc 0 → FP8 pop → m5
# ═══════════════════════════════════════════════════════════════

    # Load WK → m1 → XLU → m2 (reuse scratch banks)
    ADDI  x6, x0, 0
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 40
    VTRPOSE.XLU 2, 1
    DELAY 80

    # ── [c] MXU1 w-push + acc-push ──
    VMATPUSH.W.MXU1       0, 2
    VMATPUSH.ACC.FP8.MXU1 0, 3
    DELAY 35                          # ← wait for Slot A acc push + Slot B weight push

    # X is still in m0 (wasn't overwritten). Reuse for MXU1.
    VMATMUL.ACC.MXU1  0, 0, 0
    DELAY 120

    # ── [e] MXU1 FP8 pop + DMA store of VPU vmin result ──
    VMATPOP.FP8.MXU1  5, 0, 0

    # Store vmin result (m12,m13) while MXU1 pop runs
    LI    x8, 512
    VSTORE 12, x8, 0
    DELAY 40
    LI    x9, 768
    VSTORE 13, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x10, 0x3800
    DMA.STORE x10, x8, x12, 2
    LI    x9, 768
    LI    x10, 0x3C00
    DMA.STORE x10, x9, x12, 3
    DMA.WAIT  2
    DMA.WAIT  3

    # Store K (m5) → DRAM scratch
    LI    x8, 512
    VSTORE 5, x8, 0
    DELAY 40
    LI    x8, 512
    LI    x10, 0x2C00
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

# ═══════════════════════════════════════════════════════════════
# PHASE C: MXU0 computes S = Q @ K^T → acc 0 → BF16 pop → m6,m7
#
# K is pushed directly as weights (no transpose) to get Q @ K^T.
# This mirrors the attention score computation.
# ═══════════════════════════════════════════════════════════════

    # Load K from DRAM → m1, push as weight (NO transpose for K^T)
    ADDI  x6, x0, 0
    LI    x10, 0x2C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 40

    VMATPUSH.W.MXU0       0, 1    # K directly as weights
    VMATPUSH.ACC.FP8.MXU0 0, 3    # zero bias → acc 0

    # ── [f] Load Q from DRAM (DMA) while MXU0 push completes ──
    ADDI  x6, x0, 0
    LI    x10, 0x2800
    DMA.LOAD  x6, x10, x12, 1     # ch1 for Q load
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0                 # m0 ← Q (activations)
    DELAY 40

    # MXU0 compute S = Q @ K^T
    VMATMUL.ACC.MXU0  0, 0, 0
    DELAY 120

    # BF16 pop → m6,m7
    VMATPOP.BF16.MXU0  6, 0
    DELAY 40

    # Store scores m6,m7 → DRAM
    LI    x8, 512
    VSTORE 6, x8, 0
    DELAY 40
    LI    x9, 768
    VSTORE 7, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x10, 0x3000
    DMA.STORE x10, x8, x12, 0

    LI    x9, 768
    LI    x10, 0x3400
    DMA.STORE x10, x9, x12, 1

    DMA.WAIT  0
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# PHASE D: VPU post-process — biased = vadd(S, bias_vpu)
#   S in m6,m7, bias_vpu in m14,m15 → result in m12,m13
# ═══════════════════════════════════════════════════════════════

    VADD.BF16  12, 6, 14          # m12,m13 ← S + bias_vpu
    DELAY 150

    # Store biased result
    LI    x8, 512
    VSTORE 12, x8, 0
    DELAY 40
    LI    x9, 768
    VSTORE 13, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x10, 0x4000
    DMA.STORE x10, x8, x12, 2

    LI    x9, 768
    LI    x10, 0x4400
    DMA.STORE x10, x9, x12, 3

    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

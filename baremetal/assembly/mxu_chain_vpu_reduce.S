# mxu_chain_vpu_reduce.S — MXU1 → FP8 → MXU1 chain + VPU post-process
# @TIMEOUT 1000000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu_chain_vpu_reduce.py 333
#
# Attention-like pipeline exercising all three functional units:
#
#   Phase A: MXU1 computes Q = X @ WQ   (32×32, BF16 acc → FP8 pop)
#   Phase B: MXU1 computes K = X @ WK   (32×32, BF16 acc → FP8 pop)
#   Phase C: MXU1 computes S = Q @ K^T  (32×32, BF16 pop — final scores)
#   Phase D: VPU computes bias_add = S + bias_vpu  (BF16 element-wise)
#
# VPU ops run interleaved during DMA/transpose stalls:
#   VPU-1: vmin(A, B) during Phase A weight prep
#   VPU-2: vadd(S, bias) after Phase C (cross-unit dependency)
#
# Overlap patterns:
#   [a] MXU1 w-push + acc-push (Phase A prologue)
#   [b] MXU1 compute + MXU1 w-push (not used — single k-tile)
#   [c] MXU1 w-push + acc-push (Phase B prologue)
#   [d] MXU1 FP8 pop + VPU vmin (concurrent pop + VPU, Phase A→B)
#   [e] MXU1 FP8 pop + DMA store (pop while storing Q, Phase B→C)
#   [f] MXU1 compute (Phase C) + DMA load (K for wbuf)
#   [g] 4-channel DMA parallel (initial load)
#   [h] VPU vadd + DMA store (storing scores while VPU processes)
#
# TRF bank map (all 16 banks used):
#   m0    activations / Q for Phase C (FP8, reloaded)
#   m1    weight raw scratch (XLU input)
#   m2    weight transposed (XLU output)
#   m3    zero bias (persistent)
#   m4    Q pop result (FP8, from MXU1)
#   m5    K pop result (FP8, from MXU1)
#   m6,7  BF16 scores pop (from MXU1 Phase C)
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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024
    seli e0, 127          # identity scale for FP8 pops
    seli e1, 127

# ═══════════════════════════════════════════════════════════════
# GLOBAL SETUP: Load zero bias + VPU inputs + bias_vpu
# Use all 4 DMA channels for parallel initial load [g]
# ═══════════════════════════════════════════════════════════════

    # ch0: zero bias → VMEM[0]
    ADDI  x6, x0, 0
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch0

    # ch1: A_vpu b0 → VMEM[256]
    LI    x7, 256
    LI    x10, 0x90001000
    dma.load x7, x10, x12, ch1

    # ch2: B_vpu b0 → VMEM[512]
    LI    x8, 512
    LI    x10, 0x90001800
    dma.load x8, x10, x12, ch2

    # ch3: bias_vpu b0 → VMEM[768]
    LI    x9, 768
    LI    x10, 0x90002000
    dma.load x9, x10, x12, ch3

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

    # VLOAD first batch
    ADDI  x6, x0, 0
    vload m3, 0(x6)                 # m3 ← zero bias
    DELAY 33
    LI    x7, 256
    vload m8, 0(x7)                 # m8 ← A_vpu b0
    DELAY 33
    LI    x8, 512
    vload m10, 0(x8)                # m10 ← B_vpu b0
    DELAY 33
    LI    x9, 768
    vload m14, 0(x9)                # m14 ← bias_vpu b0
    DELAY 33

    # Second round: bank1 data
    # ch0: A_vpu b1
    ADDI  x6, x0, 0
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch0

    # ch1: B_vpu b1
    LI    x7, 256
    LI    x10, 0x90001C00
    dma.load x7, x10, x12, ch1

    # ch2: bias_vpu b1
    LI    x8, 512
    LI    x10, 0x90002400
    dma.load x8, x10, x12, ch2

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2

    ADDI  x6, x0, 0
    vload m9, 0(x6)                 # m9 ← A_vpu b1
    DELAY 33
    LI    x7, 256
    vload m11, 0(x7)                # m11 ← B_vpu b1
    DELAY 33
    LI    x8, 512
    vload m15, 0(x8)                # m15 ← bias_vpu b1
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# PHASE A: MXU1 computes Q = X @ WQ → acc 0 → FP8 pop → m4
# ═══════════════════════════════════════════════════════════════

    # Load WQ → m1 → XLU → m2
    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m1, 0(x6)
    DELAY 33
    vtrpose.xlu m2, m1
    DELAY 64

    # ── [a] MXU1 w-push + acc-push ──
    vmatpush.weight.mxu1 w0, m2
    vmatpush.acc.fp8.mxu1 acc0, m3

    # Load X → m0
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

    # MXU1 compute Q
    vmatmul.acc.mxu1 acc0, m0, w0
    DELAY 34

    # ── [d] MXU1 FP8 pop + VPU vmin (concurrent) ──
    # MXU1 pops acc 0 → m4 (FP8). VPU computes vmin on m8,m10 → m12.
    # Different functional units, different TRF banks. Safe.
    vmatpop.fp8.acc.mxu1 m4, acc0, e0
    vminimum.bf16 m12, m8, m10
    DELAY 64

    # Store Q (m4) → DRAM scratch (need it for Phase C)
    LI    x8, 512
    vstore m4, 0(x8)
    DELAY 33
    LI    x8, 512
    LI    x10, 0x90002800
    dma.store x10, x8, x12, ch2
    dma.wait ch2

# ═══════════════════════════════════════════════════════════════
# PHASE B: MXU1 computes K = X @ WK → acc 0 → FP8 pop → m5
# ═══════════════════════════════════════════════════════════════

    # Load WK → m1 → XLU → m2 (reuse scratch banks)
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m1, 0(x6)
    DELAY 33
    vtrpose.xlu m2, m1
    DELAY 64

    # ── [c] MXU1 w-push + acc-push ──
    vmatpush.weight.mxu1 w0, m2
    vmatpush.acc.fp8.mxu1 acc0, m3
    DELAY 31                          # ← wait for Slot A acc push + Slot B weight push

    # X is still in m0 (wasn't overwritten). Reuse for MXU1.
    vmatmul.acc.mxu1 acc0, m0, w0
    DELAY 34

    # ── [e] MXU1 FP8 pop + DMA store of VPU vmin result ──
    vmatpop.fp8.acc.mxu1 m5, acc0, e0

    # Store vmin result (m12,m13) while MXU1 pop runs
    LI    x8, 512
    vstore m12, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m13, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90003800
    dma.store x10, x8, x12, ch2
    LI    x9, 768
    LI    x10, 0x90003C00
    dma.store x10, x9, x12, ch3
    dma.wait ch2
    dma.wait ch3

    # Store K (m5) → DRAM scratch
    LI    x8, 512
    vstore m5, 0(x8)
    DELAY 33
    LI    x8, 512
    LI    x10, 0x90002C00
    dma.store x10, x8, x12, ch2
    dma.wait ch2

# ═══════════════════════════════════════════════════════════════
# PHASE C: MXU1 computes S = Q @ K^T → acc 0 → BF16 pop → m6,m7
#
# K is pushed directly as weights (no transpose) to get Q @ K^T.
# This mirrors the attention score computation.
# ═══════════════════════════════════════════════════════════════

    # Load K from DRAM → m1, push as weight (NO transpose for K^T)
    ADDI  x6, x0, 0
    LI    x10, 0x90002C00
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m1, 0(x6)
    DELAY 33

    vmatpush.weight.mxu1 w0, m1    # K directly as weights
    vmatpush.acc.fp8.mxu1 acc0, m3    # zero bias → acc 0

    # ── [f] Load Q from DRAM (DMA) while MXU1 push completes ──
    ADDI  x6, x0, 0
    LI    x10, 0x90002800
    dma.load x6, x10, x12, ch1     # ch1 for Q load
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m0, 0(x6)                 # m0 ← Q (activations)
    DELAY 33

    # MXU1 compute S = Q @ K^T
    vmatmul.acc.mxu1 acc0, m0, w0
    DELAY 34

    # BF16 pop → m6,m7
    vmatpop.bf16.acc.mxu1 m6, acc0
    DELAY 31

    # Store scores m6,m7 → DRAM
    LI    x8, 512
    vstore m6, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m7, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90003000
    dma.store x10, x8, x12, ch0

    LI    x9, 768
    LI    x10, 0x90003400
    dma.store x10, x9, x12, ch1

    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# PHASE D: VPU post-process — biased = vadd(S, bias_vpu)
#   S in m6,m7, bias_vpu in m14,m15 → result in m12,m13
# ═══════════════════════════════════════════════════════════════

    vadd.bf16 m12, m6, m14          # m12,m13 ← S + bias_vpu
    DELAY 64

    # Store biased result
    LI    x8, 512
    vstore m12, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m13, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90004000
    dma.store x10, x8, x12, ch2

    LI    x9, 768
    LI    x10, 0x90004400
    dma.store x10, x9, x12, ch3

    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL

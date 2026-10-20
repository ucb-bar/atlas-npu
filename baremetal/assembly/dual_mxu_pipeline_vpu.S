# dual_mxu_pipeline_vpu.S — MXU0 + MXU1 pipelined K=64, VPU interleaved
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dual_mxu_pipeline_vpu.py 500
#
# Both MXUs perform 2-k-tile (K=64) matmuls with weight double-buffering
# and acc ping-pong.  VPU binary ops run in the gaps between DMA/transpose
# phases, and again during MXU pops for cross-unit post-processing.
#
# Computations:
#   MXU0 (SA):  C0 = X @ W0  (32×64 @ 64×32 → 32×32 BF16, 2 k-tiles)
#   MXU1 (IPT): C1 = X @ W1  (32×64 @ 64×32 → 32×32 BF16, 2 k-tiles)
#   VPU:        D  = vsub(A, B)   (32×32 BF16)
#               E  = vmax(A, B)   (32×32 BF16)
#               F  = vsub(C0, C1) (cross-unit)
#
# Overlap patterns demonstrated:
#   [a] MXU0 w-push + acc-push (prologue)
#   [b] MXU1 w-push + acc-push (prologue)
#   [c] MXU0 MATMUL.ACC + MXU0 w-push (next k-tile preload)
#   [d] MXU1 MATMUL.ACC + MXU1 w-push (next k-tile preload)
#   [e] MXU0 compute + MXU1 compute (dual MXU concurrent)
#   [f] VPU compute during DMA load stalls
#   [g] MXU0 pop + MXU1 pop (concurrent pops, different acc/banks)
#   [h] 4-channel DMA parallel load
#   [i] DMA store + VPU compute
#
# TRF bank map (all 16 banks used, with phase-based reuse):
#   m0    MXU0 activations slab 0 (FP8)
#   m1    MXU0 activations slab 1 (FP8)
#   m2    MXU0 weight even (raw → XLU staging) → VPU sub bank0 → MXU0 pop lo
#   m3    MXU0 weight even transposed           → VPU sub bank1 → MXU0 pop hi
#   m4    MXU0 weight odd  (raw → XLU staging) → VPU max bank0 → MXU1 pop lo
#   m5    MXU0 weight odd  transposed           → VPU max bank1 → MXU1 pop hi
#   m6    MXU1 weight even (raw → XLU staging) → MXU1 activations slab 0 dup
#   m7    MXU1 weight even transposed           → diff bank1
#   m8    MXU1 weight odd  (raw → XLU staging) → MXU1 activations slab 1 dup
#   m9    MXU1 weight odd  transposed
#   m10   zero bias (persistent)
#   m11   scratch
#   m12   VPU input A bank0
#   m13   VPU input A bank1
#   m14   VPU input B bank0
#   m15   VPU input B bank1
#
# NOTE: Under the 1R1W MREG contract, each active reader needs a distinct
# bank. Before each dual-MXU compute window we duplicate the activation slab
# so MXU0 and MXU1 never read the same bank in the same cycle.
#
# DRAM layout:
#   0x0000  X[:,0:32]      (1024B FP8)
#   0x0400  X[:,32:64]     (1024B FP8)
#   0x0800  W0_A (k0)     (1024B FP8)
#   0x0C00  W0_B (k1)     (1024B FP8)
#   0x1000  W1_A (k0)     (1024B FP8)
#   0x1400  W1_B (k1)     (1024B FP8)
#   0x1800  zero bias     (1024B FP8)
#   0x1C00  A_vpu bank0   (1024B BF16)
#   0x2000  A_vpu bank1   (1024B BF16)
#   0x2400  B_vpu bank0   (1024B BF16)
#   0x2800  B_vpu bank1   (1024B BF16)
#   — outputs —
#   0x2C00  C0 lo          (1024B BF16)
#   0x3000  C0 hi          (1024B BF16)
#   0x3400  C1 lo          (1024B BF16)
#   0x3800  C1 hi          (1024B BF16)
#   0x3C00  sub bank0      (1024B BF16)
#   0x4000  sub bank1      (1024B BF16)
#   0x4400  max bank0      (1024B BF16)
#   0x4800  max bank1      (1024B BF16)
#   0x4C00  diff bank0     (1024B BF16) — vsub(C0,C1)
#   0x5000  diff bank1     (1024B BF16)

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024
    seli e0, 127
    seli e1, 127

# ═══════════════════════════════════════════════════════════════
# SETUP: Load zero bias → m10
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m10, 0(x6)
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# SETUP: Load VPU inputs (4-channel parallel DMA) [h]
#   ch0: A_vpu bank0 → VMEM[0]
#   ch1: A_vpu bank1 → VMEM[256]
#   ch2: B_vpu bank0 → VMEM[512]
#   ch3: B_vpu bank1 → VMEM[768]
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90001C00
    dma.load x6, x10, x12, ch0

    LI    x7, 256
    LI    x10, 0x90002000
    dma.load x7, x10, x12, ch1

    LI    x8, 512
    LI    x10, 0x90002400
    dma.load x8, x10, x12, ch2

    LI    x9, 768
    LI    x10, 0x90002800
    dma.load x9, x10, x12, ch3

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

    ADDI  x6, x0, 0
    vload m12, 0(x6)                # m12 ← A_vpu bank0
    DELAY 33
    LI    x7, 256
    vload m13, 0(x7)                # m13 ← A_vpu bank1
    DELAY 33
    LI    x8, 512
    vload m14, 0(x8)                # m14 ← B_vpu bank0
    DELAY 33
    LI    x9, 768
    vload m15, 0(x9)                # m15 ← B_vpu bank1
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# MXU0 PROLOGUE: Load W0_A (k=0), transpose, push weights + bias
# ═══════════════════════════════════════════════════════════════

    # W0_A → VMEM[0] → m2 → XLU → m3
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m2, 0(x6)
    DELAY 33
    vtrpose.xlu m3, m2
    DELAY 64

    # ── [a] MXU0 w-push + acc-push ──
    vmatpush.weight.mxu0 w0, m3
    vmatpush.acc.fp8.mxu0 acc0, m10

    # Preload W0_B (k=1) → m4 → XLU → m5
    ADDI  x6, x0, 0
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    DELAY 33
    vtrpose.xlu m5, m4
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# MXU1 PROLOGUE: Load W1_A (k=0), transpose, push weights + bias
# ═══════════════════════════════════════════════════════════════

    # W1_A → VMEM[0] → m6 → XLU → m7
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m6, 0(x6)
    DELAY 33
    vtrpose.xlu m7, m6
    DELAY 64

    # ── [b] MXU1 w-push + acc-push ──
    vmatpush.weight.mxu1 w0, m7
    vmatpush.acc.fp8.mxu1 acc0, m10

    # Preload W1_B (k=1) → m8 → XLU → m9
    ADDI  x6, x0, 0
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m8, 0(x6)
    DELAY 33
    vtrpose.xlu m9, m8
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# K-TILE 0: Load X[:,0:32] → m0 and duplicate → m6 for MXU1
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90000000  # DRAM 0x0000: X[:,0:32]
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)                 # MXU0 activations slab 0
    DELAY 33
    vload m6, 0(x6)                 # MXU1 duplicate slab 0
    DELAY 33

    # ── [c]+[e]: MXU0 compute + w-push, MXU1 compute + w-push ──
    # MXU0 reads m0; MXU1 reads duplicated slab m6.
    # This keeps every active MREG reader on a distinct bank.
    # MXU0: matmul acc 0 with X slab 0, push W0_B→wbuf1
    # MXU1: matmul acc 0 with duplicated X slab 0, push W1_B→wbuf1
    vmatmul.acc.mxu0 acc0, m0, w0
    vmatpush.weight.mxu0 w1, m5
    vmatmul.acc.mxu1 acc0, m6, w0
    vmatpush.weight.mxu1 w1, m9

    # ── [f]: VPU vsub during MXU compute ──
    # VPU reads the even/odd source pairs (m12,m13) and (m14,m15) —
    # no conflict with MXU activations (m0, m6) or accumulators
    vsub.bf16 m2, m12, m14          # m2,m3 ← vsub(A_vpu, B_vpu)
    # Note: m2,m3 held MXU0 weight data, now overwritten (weights already pushed)

# ═══════════════════════════════════════════════════════════════
# K-TILE 1: Load X[:,32:64] → m1 and duplicate → m8 for MXU1
# ═══════════════════════════════════════════════════════════════

    # Need tile-0 matmuls to finish before issuing the tile-1 accumulate.
    DELAY 64

    ADDI  x6, x0, 0
    LI    x10, 0x90000400  # DRAM 0x0400: X[:,32:64]
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m1, 0(x6)                 # MXU0 activations slab 1
    DELAY 33
    vload m8, 0(x6)                 # MXU1 duplicate slab 1
    DELAY 33

    # ── [e]: Dual MXU concurrent compute, k-tile 1 ──
    # MXU0 reads m1; MXU1 reads duplicated slab m8.
    vmatmul.acc.mxu0 acc0, m1, w1     # acc 0 += X[:,32:64] @ W0_B
    vmatmul.acc.mxu1 acc0, m8, w1     # acc 0 += X[:,32:64] @ W1_B

    # ── [f]: VPU vmax during MXU compute ──
    # Same source pairs as VSUB: A=(m12,m13), B=(m14,m15). VSUB wrote
    # to m2,m3 so sources m12-m15 are still intact.
    vmaximum.bf16 m4, m12, m14          # m4,m5 ← vmax(A_vpu, B_vpu)
    # m4,m5 held MXU0 W0_B weight data, now overwritten (already pushed)

    # Wait for MXU matmul pipelines to drain
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# STORE VPU vsub result (m2,m3) — overlapped with MXU acc drain
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    vstore m2, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m3, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90003C00
    dma.store x10, x8, x12, ch2
    LI    x9, 768
    LI    x10, 0x90004000
    dma.store x10, x9, x12, ch3
    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# STORE VPU vmax result (m4,m5)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    vstore m4, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m5, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90004400
    dma.store x10, x8, x12, ch2
    LI    x9, 768
    LI    x10, 0x90004800
    dma.store x10, x9, x12, ch3
    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# POP RESULTS: MXU0 pop + MXU1 pop [g]
#   MXU0 BF16 pop → m2,m3 (now free after vsub stored)
#   MXU1 BF16 pop → m4,m5 (now free after vmax stored)
# ═══════════════════════════════════════════════════════════════

    # ── [g]: concurrent pops on different units + acc buffers ──
    vmatpop.bf16.acc.mxu0 m2, acc0
    vmatpop.bf16.acc.mxu1 m4, acc0
    DELAY 31

# ═══════════════════════════════════════════════════════════════
# STORE MXU0 result (m2,m3)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    vstore m2, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m3, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90002C00
    dma.store x10, x8, x12, ch0
    LI    x9, 768
    LI    x10, 0x90003000
    dma.store x10, x9, x12, ch1
    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# STORE MXU1 result (m4,m5)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    vstore m4, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m5, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90003400
    dma.store x10, x8, x12, ch0
    LI    x9, 768
    LI    x10, 0x90003800
    dma.store x10, x9, x12, ch1
    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# CROSS-UNIT: VPU vsub(C0, C1) — m2,m3 = C0, m4,m5 = C1
#   ── [i]: VPU compute ──
# ═══════════════════════════════════════════════════════════════

    vsub.bf16 m6, m2, m4            # m6,m7 ← vsub(C0, C1)
    DELAY 64

    LI    x8, 512
    vstore m6, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m7, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90004C00
    dma.store x10, x8, x12, ch2
    LI    x9, 768
    LI    x10, 0x90005000
    dma.store x10, x9, x12, ch3
    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL
    

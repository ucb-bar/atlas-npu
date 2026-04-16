# dual_mxu_pipeline_vpu.S — MXU0 + MXU1 pipelined K=64, VPU interleaved
# @TIMEOUT 800000
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

    LI    x5,  0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127
    SELI  1, 127

# ═══════════════════════════════════════════════════════════════
# SETUP: Load zero bias → m10
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 10, x6, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# SETUP: Load VPU inputs (4-channel parallel DMA) [h]
#   ch0: A_vpu bank0 → VMEM[0]
#   ch1: A_vpu bank1 → VMEM[256]
#   ch2: B_vpu bank0 → VMEM[512]
#   ch3: B_vpu bank1 → VMEM[768]
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x1C00
    DMA.LOAD  x6, x10, x12, 0

    LI    x7, 256
    LI    x10, 0x2000
    DMA.LOAD  x7, x10, x12, 1

    LI    x8, 512
    LI    x10, 0x2400
    DMA.LOAD  x8, x10, x12, 2

    LI    x9, 768
    LI    x10, 0x2800
    DMA.LOAD  x9, x10, x12, 3

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    ADDI  x6, x0, 0
    VLOAD 12, x6, 0                # m12 ← A_vpu bank0
    DELAY 40
    LI    x7, 256
    VLOAD 13, x7, 0                # m13 ← A_vpu bank1
    DELAY 40
    LI    x8, 512
    VLOAD 14, x8, 0                # m14 ← B_vpu bank0
    DELAY 40
    LI    x9, 768
    VLOAD 15, x9, 0                # m15 ← B_vpu bank1
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# MXU0 PROLOGUE: Load W0_A (k=0), transpose, push weights + bias
# ═══════════════════════════════════════════════════════════════

    # W0_A → VMEM[0] → m2 → XLU → m3
    ADDI  x6, x0, 0
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 2, x6, 0
    DELAY 40
    VTRPOSE.XLU 3, 2
    DELAY 80

    # ── [a] MXU0 w-push + acc-push ──
    VMATPUSH.W.MXU0       0, 3
    VMATPUSH.ACC.FP8.MXU0 0, 10

    # Preload W0_B (k=1) → m4 → XLU → m5
    ADDI  x6, x0, 0
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40
    VTRPOSE.XLU 5, 4
    DELAY 80

# ═══════════════════════════════════════════════════════════════
# MXU1 PROLOGUE: Load W1_A (k=0), transpose, push weights + bias
# ═══════════════════════════════════════════════════════════════

    # W1_A → VMEM[0] → m6 → XLU → m7
    ADDI  x6, x0, 0
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 40
    VTRPOSE.XLU 7, 6
    DELAY 80

    # ── [b] MXU1 w-push + acc-push ──
    VMATPUSH.W.MXU1       0, 7
    VMATPUSH.ACC.FP8.MXU1 0, 10

    # Preload W1_B (k=1) → m8 → XLU → m9
    ADDI  x6, x0, 0
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 8, x6, 0
    DELAY 40
    VTRPOSE.XLU 9, 8
    DELAY 80

# ═══════════════════════════════════════════════════════════════
# K-TILE 0: Load X[:,0:32] → m0 and duplicate → m6 for MXU1
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0              # DRAM 0x0000: X[:,0:32]
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0                 # MXU0 activations slab 0
    DELAY 40
    VLOAD 6, x6, 0                 # MXU1 duplicate slab 0
    DELAY 40

    # ── [c]+[e]: MXU0 compute + w-push, MXU1 compute + w-push ──
    # MXU0 reads m0; MXU1 reads duplicated slab m6.
    # This keeps every active MREG reader on a distinct bank.
    # MXU0: matmul acc 0 with X slab 0, push W0_B→wbuf1
    # MXU1: matmul acc 0 with duplicated X slab 0, push W1_B→wbuf1
    VMATMUL.ACC.MXU0  0, 0, 0
    VMATPUSH.W.MXU0   1, 5
    VMATMUL.ACC.MXU1  0, 6, 0
    VMATPUSH.W.MXU1   1, 9

    # ── [f]: VPU vsub during MXU compute ──
    # VPU reads the even/odd source pairs (m12,m13) and (m14,m15) —
    # no conflict with MXU activations (m0, m6) or accumulators
    VSUB.BF16  2, 12, 14          # m2,m3 ← vsub(A_vpu, B_vpu)
    # Note: m2,m3 held MXU0 weight data, now overwritten (weights already pushed)

# ═══════════════════════════════════════════════════════════════
# K-TILE 1: Load X[:,32:64] → m1 and duplicate → m8 for MXU1
# ═══════════════════════════════════════════════════════════════

    # Need tile-0 matmuls to finish before issuing the tile-1 accumulate.
    DELAY 120

    ADDI  x6, x0, 0
    LI    x10, 0x0400             # DRAM 0x0400: X[:,32:64]
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0                 # MXU0 activations slab 1
    DELAY 40
    VLOAD 8, x6, 0                 # MXU1 duplicate slab 1
    DELAY 40

    # ── [e]: Dual MXU concurrent compute, k-tile 1 ──
    # MXU0 reads m1; MXU1 reads duplicated slab m8.
    VMATMUL.ACC.MXU0  0, 1, 1     # acc 0 += X[:,32:64] @ W0_B
    VMATMUL.ACC.MXU1  0, 8, 1     # acc 0 += X[:,32:64] @ W1_B

    # ── [f]: VPU vmax during MXU compute ──
    # Same source pairs as VSUB: A=(m12,m13), B=(m14,m15). VSUB wrote
    # to m2,m3 so sources m12-m15 are still intact.
    VMAX.BF16  4, 12, 14          # m4,m5 ← vmax(A_vpu, B_vpu)
    # m4,m5 held MXU0 W0_B weight data, now overwritten (already pushed)

    # Wait for MXU matmul pipelines to drain
    DELAY 120

# ═══════════════════════════════════════════════════════════════
# STORE VPU vsub result (m2,m3) — overlapped with MXU acc drain
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40
    LI    x9, 768
    VSTORE 3, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x10, 0x3C00
    DMA.STORE x10, x8, x12, 2
    LI    x9, 768
    LI    x10, 0x4000
    DMA.STORE x10, x9, x12, 3
    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# STORE VPU vmax result (m4,m5)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 40
    LI    x9, 768
    VSTORE 5, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x10, 0x4400
    DMA.STORE x10, x8, x12, 2
    LI    x9, 768
    LI    x10, 0x4800
    DMA.STORE x10, x9, x12, 3
    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# POP RESULTS: MXU0 pop + MXU1 pop [g]
#   MXU0 BF16 pop → m2,m3 (now free after vsub stored)
#   MXU1 BF16 pop → m4,m5 (now free after vmax stored)
# ═══════════════════════════════════════════════════════════════

    # ── [g]: concurrent pops on different units + acc buffers ──
    VMATPOP.BF16.MXU0  2, 0
    VMATPOP.BF16.MXU1  4, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# STORE MXU0 result (m2,m3)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40
    LI    x9, 768
    VSTORE 3, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x10, 0x2C00
    DMA.STORE x10, x8, x12, 0
    LI    x9, 768
    LI    x10, 0x3000
    DMA.STORE x10, x9, x12, 1
    DMA.WAIT  0
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# STORE MXU1 result (m4,m5)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 40
    LI    x9, 768
    VSTORE 5, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x10, 0x3400
    DMA.STORE x10, x8, x12, 0
    LI    x9, 768
    LI    x10, 0x3800
    DMA.STORE x10, x9, x12, 1
    DMA.WAIT  0
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# CROSS-UNIT: VPU vsub(C0, C1) — m2,m3 = C0, m4,m5 = C1
#   ── [i]: VPU compute ──
# ═══════════════════════════════════════════════════════════════

    VSUB.BF16  6, 2, 4            # m6,m7 ← vsub(C0, C1)
    DELAY 150

    LI    x8, 512
    VSTORE 6, x8, 0
    DELAY 40
    LI    x9, 768
    VSTORE 7, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x10, 0x4C00
    DMA.STORE x10, x8, x12, 2
    LI    x9, 768
    LI    x10, 0x5000
    DMA.STORE x10, x9, x12, 3
    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL
    

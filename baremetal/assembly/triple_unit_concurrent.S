# triple_unit_concurrent.S — MXU0 + MXU1 + VPU concurrent execution
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_triple_unit_concurrent.py 777
# @PMU_TEST
# These IDs follow the Atlas PMU's sorted event list for the default config.
# @PMU_EVENT dma.beats_completed 2
# @PMU_EVENT dma.commands 4
# @PMU_EVENT mxu0.commands 139
# @PMU_EVENT mxu0.compute_cycles 140
# @PMU_EVENT mxu1.commands 142
# @PMU_EVENT mxu1.compute_cycles 143
# @PMU_EVENT vpu.active_cycles 163
# @PMU_EVENT vpu.commands 164
#
# Three independent functional units compute simultaneously:
#   MXU0 (SA):  C0 = X0 @ W0      (32×32, single k-tile, BF16 out)
#   MXU1 (IPT): C1 = X1 @ W1      (32×32, single k-tile, BF16 out)
#   VPU:        D  = vadd(A, B)    (32×32 BF16 element-wise)
#               E  = vmul(A, B)    (32×32 BF16 element-wise)
#
# After MXU pops, cross-unit data dependency:
#   F  = vadd(C0, C1)              (VPU processes MXU outputs)
#
# Overlap patterns demonstrated:
#   [a] MXU0 weight push + MXU0 acc push (prologue)
#   [b] MXU1 weight push + MXU1 acc push (prologue)
#   [c] MXU0 compute + MXU1 compute + VPU add (TRIPLE UNIT)
#   [d] MXU0 pop + VPU mul (pop on MXU0 + VPU on different banks)
#   [e] MXU1 pop + VPU store (pop on MXU1 + LSU store of VPU result)
#   [f] 4-channel DMA parallel load during setup
#   [g] DMA store + VPU compute (cross-unit post-processing)
#
# TRF bank map (all 16 banks used):
#   m0    MXU0 activations (X0, FP8)
#   m1    weight raw scratch (shared staging for XLU)
#   m2    MXU0 weight transposed
#   m3    zero bias (shared, persistent)
#   m4,5  MXU0 BF16 result (pop)
#   m6    MXU1 activations (X1, FP8)
#   m7    MXU1 weight transposed
#   m8,9  VPU input A (BF16, 2 banks)
#   m10,11 VPU input B (BF16, 2 banks)
#   m12,13 VPU output / scratch (BF16, 2 banks)
#   m14,15 MXU1 BF16 result (pop)
#
# VMEM staging map (word addresses, 256 words = 1024B per tile):
#   [0..255]      general tile staging (DMA ch 0)
#   [256..511]    general tile staging (DMA ch 1)
#   [512..767]    store staging area 1
#   [768..1023]   store staging area 2
#
# DRAM layout:
#   0x0000  X0           (1024B FP8)  — MXU0 activations
#   0x0400  W0           (1024B FP8)  — MXU0 weights
#   0x0800  X1           (1024B FP8)  — MXU1 activations
#   0x0C00  W1           (1024B FP8)  — MXU1 weights
#   0x1000  zero bias    (1024B FP8, all 0x00)
#   0x1400  A_vpu bank0  (1024B BF16) — VPU input A lanes 0-15
#   0x1800  A_vpu bank1  (1024B BF16) — VPU input A lanes 16-31
#   0x1C00  B_vpu bank0  (1024B BF16) — VPU input B lanes 0-15
#   0x2000  B_vpu bank1  (1024B BF16) — VPU input B lanes 16-31
#   — outputs —
#   0x2400  C0 lo        (1024B BF16) — MXU0 result lanes 0-15
#   0x2800  C0 hi        (1024B BF16) — MXU0 result lanes 16-31
#   0x2C00  C1 lo        (1024B BF16) — MXU1 result lanes 0-15
#   0x3000  C1 hi        (1024B BF16) — MXU1 result lanes 16-31
#   0x3400  add bank0    (1024B BF16) — VPU vadd result lanes 0-15
#   0x3800  add bank1    (1024B BF16) — VPU vadd result lanes 16-31
#   0x3C00  mul bank0    (1024B BF16) — VPU vmul result lanes 0-15
#   0x4000  mul bank1    (1024B BF16) — VPU vmul result lanes 16-31
#   0x4400  sum bank0    (1024B BF16) — VPU vadd(C0,C1) lanes 0-15
#   0x4800  sum bank1    (1024B BF16) — VPU vadd(C0,C1) lanes 16-31

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024
    seli e0, 127          # scale reg 0 — identity E8M0
    seli e1, 127          # scale reg 1 — identity E8M0

# ═══════════════════════════════════════════════════════════════
# PHASE 0: Load zero bias → m3 (persistent, shared by both MXUs)
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90001000  # DRAM: zero bias
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m3, 0(x6)
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Prepare MXU0 weights — load W0, transpose, push
# ═══════════════════════════════════════════════════════════════

    # W0 → VMEM[0] → m1 (raw) → XLU transpose → m2
    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m1, 0(x6)
    DELAY 33
    vtrpose.xlu m2, m1
    DELAY 64

    # ── OVERLAP [a]: MXU0 weight push + acc push ──
    # Slot B: push W0 transposed → MXU0 wbuf 0
    # Slot A: push zero bias → MXU0 acc 0
    vmatpush.weight.mxu0 w0, m2
    vmatpush.acc.fp8.mxu0 acc0, m3

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Prepare MXU1 weights — load W1, transpose, push
#          (m1 is free now; reuse as raw staging)
# ═══════════════════════════════════════════════════════════════

    # W1 → VMEM[0] → m1 (raw) → XLU transpose → m7
    ADDI  x6, x0, 0
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m1, 0(x6)
    DELAY 33
    vtrpose.xlu m7, m1
    DELAY 64

    # ── OVERLAP [b]: MXU1 weight push + acc push ──
    vmatpush.weight.mxu1 w0, m7
    vmatpush.acc.fp8.mxu1 acc0, m3

# ═══════════════════════════════════════════════════════════════
# PHASE 3: 4-channel parallel DMA load of activations + VPU data
#   Channel 0: X0 → VMEM[0]
#   Channel 1: X1 → VMEM[256]
#   Channel 2: A_vpu bank0 → VMEM[512]
#   Channel 3: B_vpu bank0 → VMEM[768]
# ═══════════════════════════════════════════════════════════════

    # ── OVERLAP [f]: 4-channel DMA parallel load ──
    ADDI  x6, x0, 0
    LI    x10, 0x90000000  # DRAM 0x0000: X0
    dma.load x6, x10, x12, ch0

    LI    x7, 256
    LI    x10, 0x90000800  # DRAM 0x0800: X1
    dma.load x7, x10, x12, ch1

    LI    x8, 512
    LI    x10, 0x90001400  # DRAM 0x1400: A_vpu bank0
    dma.load x8, x10, x12, ch2

    LI    x9, 768
    LI    x10, 0x90001C00  # DRAM 0x1C00: B_vpu bank0
    dma.load x9, x10, x12, ch3

    # Wait for all 4 channels
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

    # VLOAD X0 → m0, X1 → m6, A_vpu_b0 → m8, B_vpu_b0 → m10
    ADDI  x6, x0, 0
    vload m0, 0(x6)                 # m0 ← X0
    DELAY 33
    LI    x7, 256
    vload m6, 0(x7)                 # m6 ← X1
    DELAY 33
    LI    x8, 512
    vload m8, 0(x8)                 # m8 ← A_vpu bank0
    DELAY 33
    LI    x9, 768
    vload m10, 0(x9)                # m10 ← B_vpu bank0
    DELAY 33

    # Second round: load VPU bank1 data
    # Channel 0: A_vpu bank1 → VMEM[0]
    # Channel 1: B_vpu bank1 → VMEM[256]
    ADDI  x6, x0, 0
    LI    x10, 0x90001800  # DRAM 0x1800: A_vpu bank1
    dma.load x6, x10, x12, ch0

    LI    x7, 256
    LI    x10, 0x90002000  # DRAM 0x2000: B_vpu bank1
    dma.load x7, x10, x12, ch1

    dma.wait ch0
    dma.wait ch1

    ADDI  x6, x0, 0
    vload m9, 0(x6)                 # m9 ← A_vpu bank1
    DELAY 33
    LI    x7, 256
    vload m11, 0(x7)                # m11 ← B_vpu bank1
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# PHASE 4: TRIPLE UNIT CONCURRENT COMPUTE
#   MXU0: MATMUL.ACC acc 0, X0 (m0), wbuf 0
#   MXU1: MATMUL.ACC acc 0, X1 (m6), wbuf 0
#   VPU:  VADD.BF16 m12, m8, m10
#
#   Bank conflicts: MXU0 reads m0, MXU1 reads m6, VPU reads m8/m10
#                   writes: MXU0→acc0, MXU1→acc0, VPU→m12/m13
#                   All distinct banks. Safe.
# ═══════════════════════════════════════════════════════════════

    # ── OVERLAP [c]: MXU0 compute + MXU1 compute + VPU add ──
    vmatmul.acc.mxu0 acc0, m0, w0
    vmatmul.acc.mxu1 acc0, m6, w0
    vadd.bf16 m12, m8, m10

    # Wait for all three to complete
    # MXU matmul pipeline: ~120 cycles
    # VPU add: ~150 cycles
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# PHASE 5: MXU0 pop + VPU mul (overlapped)
#   MXU0 pop BF16 → m4,m5
#   VPU:  VMUL.BF16 m8, m8, m10
#         BF16 writes consume an even-start register pair, so the mul result
#         overwrites m8,m9 after both add and mul have already consumed A_vpu.
#
#   MXU0 pop writes m4,m5, so there is no bank conflict with the VPU writeback.
# ═══════════════════════════════════════════════════════════════

    # ── OVERLAP [d]: MXU0 pop + VPU mul ──
    vmatpop.bf16.acc.mxu0 m4, acc0
    vmul.bf16 m8, m8, m10      # overwrite m8,m9 with mul result
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# PHASE 6: MXU1 pop + store VPU add result
#   MXU1 pop BF16 → m14,m15
#   Store VPU add (m12,m13) to DRAM while pop runs
# ═══════════════════════════════════════════════════════════════

    # ── OVERLAP [e]: MXU1 pop + VPU add store ──
    vmatpop.bf16.acc.mxu1 m14, acc0

    # Store VPU add result: m12 → VMEM[512], m13 → VMEM[768]
    LI    x8, 512
    vstore m12, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m13, 0(x9)
    DELAY 33

    # DMA store VPU add bank0 and bank1
    LI    x8, 512
    LI    x10, 0x90003400  # DRAM: add bank0
    dma.store x10, x8, x12, ch2

    LI    x9, 768
    LI    x10, 0x90003800  # DRAM: add bank1
    dma.store x10, x9, x12, ch3

    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# PHASE 7: Store VPU mul result (m8,m9)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    vstore m8, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m9, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90003C00  # DRAM: mul bank0
    dma.store x10, x8, x12, ch2

    LI    x9, 768
    LI    x10, 0x90004000  # DRAM: mul bank1
    dma.store x10, x9, x12, ch3

    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# PHASE 8: Store MXU0 result (m4,m5)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    vstore m4, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m5, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90002400  # DRAM: C0 lo
    dma.store x10, x8, x12, ch0

    LI    x9, 768
    LI    x10, 0x90002800  # DRAM: C0 hi
    dma.store x10, x9, x12, ch1

    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# PHASE 9: Store MXU1 result (m14,m15)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    vstore m14, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m15, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90002C00  # DRAM: C1 lo
    dma.store x10, x8, x12, ch0

    LI    x9, 768
    LI    x10, 0x90003000  # DRAM: C1 hi
    dma.store x10, x9, x12, ch1

    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# PHASE 10: Cross-unit post-processing — F = vadd(C0, C1)
#   C0 is in m4,m5 (BF16). C1 is in m14,m15 (BF16).
#   VPU: VADD.BF16 m12, m4, m14 → result in m12,m13
# ═══════════════════════════════════════════════════════════════

    # ── OVERLAP [g]: VPU cross-unit compute ──
    vadd.bf16 m12, m4, m14
    DELAY 64

    # Store F = vadd(C0, C1)
    LI    x8, 512
    vstore m12, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m13, 0(x9)
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90004400  # DRAM: sum bank0
    dma.store x10, x8, x12, ch2

    LI    x9, 768
    LI    x10, 0x90004800  # DRAM: sum bank1
    dma.store x10, x9, x12, ch3

    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL
    

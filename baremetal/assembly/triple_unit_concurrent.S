# triple_unit_concurrent.S — MXU0 + MXU1 + VPU concurrent execution
# @TIMEOUT 500000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_triple_unit_concurrent.py 777
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

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127          # scale reg 0 — identity E8M0
    SELI  1, 127          # scale reg 1 — identity E8M0

# ═══════════════════════════════════════════════════════════════
# PHASE 0: Load zero bias → m3 (persistent, shared by both MXUs)
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x1000              # DRAM: zero bias
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Prepare MXU0 weights — load W0, transpose, push
# ═══════════════════════════════════════════════════════════════

    # W0 → VMEM[0] → m1 (raw) → XLU transpose → m2
    ADDI  x6, x0, 0
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 33
    VTRPOSE.XLU 2, 1
    DELAY 64

    # ── OVERLAP [a]: MXU0 weight push + acc push ──
    # Slot B: push W0 transposed → MXU0 wbuf 0
    # Slot A: push zero bias → MXU0 acc 0
    VMATPUSH.W.MXU0       0, 2
    VMATPUSH.ACC.FP8.MXU0 0, 3

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Prepare MXU1 weights — load W1, transpose, push
#          (m1 is free now; reuse as raw staging)
# ═══════════════════════════════════════════════════════════════

    # W1 → VMEM[0] → m1 (raw) → XLU transpose → m7
    ADDI  x6, x0, 0
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 33
    VTRPOSE.XLU 7, 1
    DELAY 64

    # ── OVERLAP [b]: MXU1 weight push + acc push ──
    VMATPUSH.W.MXU1       0, 7
    VMATPUSH.ACC.FP8.MXU1 0, 3

# ═══════════════════════════════════════════════════════════════
# PHASE 3: 4-channel parallel DMA load of activations + VPU data
#   Channel 0: X0 → VMEM[0]
#   Channel 1: X1 → VMEM[256]
#   Channel 2: A_vpu bank0 → VMEM[512]
#   Channel 3: B_vpu bank0 → VMEM[768]
# ═══════════════════════════════════════════════════════════════

    # ── OVERLAP [f]: 4-channel DMA parallel load ──
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0              # DRAM 0x0000: X0
    DMA.LOAD  x6, x10, x12, 0

    LI    x7, 256
    LI    x10, 0x0800             # DRAM 0x0800: X1
    DMA.LOAD  x7, x10, x12, 1

    LI    x8, 512
    LI    x10, 0x1400             # DRAM 0x1400: A_vpu bank0
    DMA.LOAD  x8, x10, x12, 2

    LI    x9, 768
    LI    x10, 0x1C00             # DRAM 0x1C00: B_vpu bank0
    DMA.LOAD  x9, x10, x12, 3

    # Wait for all 4 channels
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

    # VLOAD X0 → m0, X1 → m6, A_vpu_b0 → m8, B_vpu_b0 → m10
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0                 # m0 ← X0
    DELAY 33
    LI    x7, 256
    VLOAD 6, x7, 0                 # m6 ← X1
    DELAY 33
    LI    x8, 512
    VLOAD 8, x8, 0                 # m8 ← A_vpu bank0
    DELAY 33
    LI    x9, 768
    VLOAD 10, x9, 0                # m10 ← B_vpu bank0
    DELAY 33

    # Second round: load VPU bank1 data
    # Channel 0: A_vpu bank1 → VMEM[0]
    # Channel 1: B_vpu bank1 → VMEM[256]
    ADDI  x6, x0, 0
    LI    x10, 0x1800             # DRAM 0x1800: A_vpu bank1
    DMA.LOAD  x6, x10, x12, 0

    LI    x7, 256
    LI    x10, 0x2000             # DRAM 0x2000: B_vpu bank1
    DMA.LOAD  x7, x10, x12, 1

    DMA.WAIT  0
    DMA.WAIT  1

    ADDI  x6, x0, 0
    VLOAD 9, x6, 0                 # m9 ← A_vpu bank1
    DELAY 33
    LI    x7, 256
    VLOAD 11, x7, 0                # m11 ← B_vpu bank1
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
    VMATMUL.ACC.MXU0  0, 0, 0
    VMATMUL.ACC.MXU1  0, 6, 0
    VADD.BF16         12, 8, 10

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
    VMATPOP.BF16.MXU0 4, 0
    VMUL.BF16         8, 8, 10      # overwrite m8,m9 with mul result
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# PHASE 6: MXU1 pop + store VPU add result
#   MXU1 pop BF16 → m14,m15
#   Store VPU add (m12,m13) to DRAM while pop runs
# ═══════════════════════════════════════════════════════════════

    # ── OVERLAP [e]: MXU1 pop + VPU add store ──
    VMATPOP.BF16.MXU1 14, 0

    # Store VPU add result: m12 → VMEM[512], m13 → VMEM[768]
    LI    x8, 512
    VSTORE 12, x8, 0
    DELAY 33
    LI    x9, 768
    VSTORE 13, x9, 0
    DELAY 33

    # DMA store VPU add bank0 and bank1
    LI    x8, 512
    LI    x10, 0x3400              # DRAM: add bank0
    DMA.STORE x10, x8, x12, 2

    LI    x9, 768
    LI    x10, 0x3800              # DRAM: add bank1
    DMA.STORE x10, x9, x12, 3

    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# PHASE 7: Store VPU mul result (m8,m9)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    VSTORE 8, x8, 0
    DELAY 33
    LI    x9, 768
    VSTORE 9, x9, 0
    DELAY 33

    LI    x8, 512
    LI    x10, 0x3C00              # DRAM: mul bank0
    DMA.STORE x10, x8, x12, 2

    LI    x9, 768
    LI    x10, 0x4000              # DRAM: mul bank1
    DMA.STORE x10, x9, x12, 3

    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# PHASE 8: Store MXU0 result (m4,m5)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 33
    LI    x9, 768
    VSTORE 5, x9, 0
    DELAY 33

    LI    x8, 512
    LI    x10, 0x2400              # DRAM: C0 lo
    DMA.STORE x10, x8, x12, 0

    LI    x9, 768
    LI    x10, 0x2800              # DRAM: C0 hi
    DMA.STORE x10, x9, x12, 1

    DMA.WAIT  0
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# PHASE 9: Store MXU1 result (m14,m15)
# ═══════════════════════════════════════════════════════════════

    LI    x8, 512
    VSTORE 14, x8, 0
    DELAY 33
    LI    x9, 768
    VSTORE 15, x9, 0
    DELAY 33

    LI    x8, 512
    LI    x10, 0x2C00              # DRAM: C1 lo
    DMA.STORE x10, x8, x12, 0

    LI    x9, 768
    LI    x10, 0x3000              # DRAM: C1 hi
    DMA.STORE x10, x9, x12, 1

    DMA.WAIT  0
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# PHASE 10: Cross-unit post-processing — F = vadd(C0, C1)
#   C0 is in m4,m5 (BF16). C1 is in m14,m15 (BF16).
#   VPU: VADD.BF16 m12, m4, m14 → result in m12,m13
# ═══════════════════════════════════════════════════════════════

    # ── OVERLAP [g]: VPU cross-unit compute ──
    VADD.BF16  12, 4, 14
    DELAY 64

    # Store F = vadd(C0, C1)
    LI    x8, 512
    VSTORE 12, x8, 0
    DELAY 33
    LI    x9, 768
    VSTORE 13, x9, 0
    DELAY 33

    LI    x8, 512
    LI    x10, 0x4400              # DRAM: sum bank0
    DMA.STORE x10, x8, x12, 2

    LI    x9, 768
    LI    x10, 0x4800              # DRAM: sum bank1
    DMA.STORE x10, x9, x12, 3

    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL
    

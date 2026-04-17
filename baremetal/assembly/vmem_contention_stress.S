# vmem_contention_stress.S — VMEM bank arbitration + DMA pipeline stress test
# @TIMEOUT 1500000
# @PYTHON_GEN gen_vmem_contention_stress.py 42
#
# Exercises every VMEM contention path:
#   [WR-BP] DMA.LOAD responses + VSTORE → write-port backpressure
#   [RD-BP] DMA.STORE pre-reads + VLOAD → read-port backpressure
#   [PIPE]  4 DMA channels queued back-to-back before any WAIT
#   [3WAY]  MXU0 + MXU1 + VPU compute during DMA quiescence
#   [CHAIN] MXU0 result popped, stored to VMEM, DMA-stored to DRAM,
#           new data DMA-loaded and VLOAD-ed during contention,
#           then re-computed on MXU0 → end-to-end pipeline
#
# VMEM bank conflict analysis (8 banks, 32 KiB block banking):
#   Each aligned 1024-byte tile occupies one tensor slot inside a single bank.
#   In this test, the contending VMEM regions are intentionally packed into
#   bank 0, so same-port overlaps hold on that bank for the full 32-cycle
#   transfer and force the lower-priority client to stall.
#
# TRF bank map (all 16 used):
#   m0    activations for MXU0 (reloaded: X0, then X2)
#   m1    weight raw staging (XLU input, reused)
#   m2    weight transposed for MXU0 (XLU output)
#   m3    zero bias (FP8, persistent)
#   m4,5  MXU0 BF16 result (C0, then C2)
#   m6    activations for MXU1 (X1)
#   m7    weight transposed for MXU1
#   m8,9  VPU input A (BF16 pair)
#   m10,11 VPU input B (BF16 pair)
#   m12,13 VPU result (BF16 pair, vadd output)
#   m14,15 MXU1 BF16 result (C1)
#
# VMEM word-address regions (256 words = 1024 bytes per slot):
#   [0..255]       slot 0 — multi-purpose staging
#   [256..511]     slot 1 — multi-purpose staging
#   [512..767]     slot 2 — output staging
#   [768..1023]    slot 3 — output staging
#   [1024..1279]   slot 4 — C0 lo for DMA.STORE readback
#   [1280..1535]   slot 5 — C0 hi for DMA.STORE readback
#
# DRAM layout:
#   0x0000  X0         (1024B FP8)
#   0x0400  W0         (1024B FP8)
#   0x0800  X1         (1024B FP8)
#   0x0C00  W1         (1024B FP8)
#   0x1000  bias FP8   (1024B, all 0x00)
#   0x1400  A_vpu b0   (1024B BF16)
#   0x1800  A_vpu b1   (1024B BF16)
#   0x1C00  B_vpu b0   (1024B BF16)
#   0x2000  B_vpu b1   (1024B BF16)
#   0x2400  X2         (1024B FP8)
#   0x2800  W2         (1024B FP8)
#   — outputs —
#   0x2C00  C0 lo      (1024B BF16)
#   0x3000  C0 hi      (1024B BF16)
#   0x3400  C1 lo      (1024B BF16)
#   0x3800  C1 hi      (1024B BF16)
#   0x3C00  add b0     (1024B BF16)
#   0x4000  add b1     (1024B BF16)
#   0x4400  C2 lo      (1024B BF16)
#   0x4800  C2 hi      (1024B BF16)

    LI    x5,  0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127
    SELI  1, 127

# ═══════════════════════════════════════════════════════════════
# PHASE 0: Load persistent data — zero bias + VPU inputs
#          Sequential DMA, each tile staged through VMEM[0]
# ═══════════════════════════════════════════════════════════════

    # bias → m3
    ADDI  x6, x0, 0
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0
    DELAY 40

    # A_vpu b0 → m8, b1 → m9 (register pair via flag 0/8)
    ADDI  x6, x0, 0
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    LI    x10, 0x1800
    DMA.LOAD  x7, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 8, x6, 0
    DELAY 40
    VLOAD 9, x6, 8
    DELAY 40

    # B_vpu b0 → m10, b1 → m11
    ADDI  x6, x0, 0
    LI    x10, 0x1C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    LI    x10, 0x2000
    DMA.LOAD  x7, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 10, x6, 0
    DELAY 40
    VLOAD 11, x6, 8
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Pipelined DMA load — 4 channels, no intermediate waits
#          [PIPE] Enqueue all 4 commands, then wait for each.
#          DMA engine processes them in FIFO order through TileLink.
#
#   ch0: X0 → VMEM[0]
#   ch1: W0 → VMEM[256]
#   ch2: X1 → VMEM[512]
#   ch3: W1 → VMEM[768]
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0     # ch0: X0 → VMEM[0]

    LI    x7, 256
    LI    x10, 0x0400
    DMA.LOAD  x7, x10, x12, 1     # ch1: W0 → VMEM[256]

    LI    x8, 512
    LI    x10, 0x0800
    DMA.LOAD  x8, x10, x12, 2     # ch2: X1 → VMEM[512]

    LI    x9, 768
    LI    x10, 0x0C00
    DMA.LOAD  x9, x10, x12, 3     # ch3: W1 → VMEM[768]

    # Commands are queued; DMA processes them sequentially on TileLink.
    # Responses may interleave.  Wait for all.
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# PHASE 2: VLOAD data + set up MXU0 and MXU1 weights
# ═══════════════════════════════════════════════════════════════

    # X0 → m0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    # W0 (raw) → m1 → XLU transpose → m2
    LI    x7, 256
    VLOAD 1, x7, 0
    DELAY 40
    VTRPOSE.XLU 2, 1
    DELAY 80

    # MXU0: push weights + zero bias.
    # Use non-accumulating MATMUL for this single-tile op so the compute
    # always starts from a fresh accum buffer.
    VMATPUSH.W.MXU0       0, 2
    VMATPUSH.ACC.FP8.MXU0 0, 3

    # X1 → m6
    LI    x8, 512
    VLOAD 6, x8, 0
    DELAY 40

    # W1 (raw) → m1 (reuse) → XLU transpose → m7
    LI    x9, 768
    VLOAD 1, x9, 0
    DELAY 40
    VTRPOSE.XLU 7, 1
    DELAY 80

    # MXU1: push weights + zero bias.
    # As above, the compute itself uses non-accumulating MATMUL.
    VMATPUSH.W.MXU1       0, 7
    VMATPUSH.ACC.FP8.MXU1 0, 3
    DELAY 40 

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Triple-unit compute — MXU0 + MXU1 + VPU concurrent
#          [3WAY] All three functional units fire on independent
#          TRF banks with no resource conflicts.
#
#   MXU0: acc0 = X0 (m0) @ W0_t (wbuf0)
#   MXU1: acc0 = X1 (m6) @ W1_t (wbuf0)
#   VPU:  m12,m13 = vadd(m8,m10) = vadd(A_vpu, B_vpu)
# ═══════════════════════════════════════════════════════════════

    VMATMUL.MXU0      0, 0, 0
    VMATMUL.MXU1      0, 6, 0
    VADD.BF16         12, 8, 10
    DELAY 150

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Pop MXU results
#   MXU0 → m4,m5 (C0)
#   MXU1 → m14,m15 (C1)
# ═══════════════════════════════════════════════════════════════

    VMATPOP.BF16.MXU0  4, 0
    VMATPOP.BF16.MXU1  14, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE 5: WRITE-PORT BACKPRESSURE [WR-BP]
#
#   Issue DMA.LOAD for X2 and W2 (writes to VMEM on response path),
#   then IMMEDIATELY start VSTORE of VPU result (also writes to VMEM).
#
#   DMA.LOAD responses arrive after TileLink round-trip (~10-20 cyc).
#   VSTORE runs for 33 cycles.  During overlap, both try to write
#   VMEM.  VSTORE wins (LSU priority), DMA stalls via vmemWriteGrant.
#
#   DMA writes to VMEM[0..255] and VMEM[256..511] in bank 0.
#   VSTORE writes to VMEM[512..767] and VMEM[768..1023] in bank 0.
#   All writes target the same bank-0 write port → maximum backpressure.
# ═══════════════════════════════════════════════════════════════

    # Enqueue DMA loads (start TileLink traffic immediately)
    ADDI  x6, x0, 0
    LI    x10, 0x2400
    DMA.LOAD  x6, x10, x12, 0     # ch0: X2 → VMEM[0]

    LI    x7, 256
    LI    x10, 0x2800
    DMA.LOAD  x7, x10, x12, 1     # ch1: W2 → VMEM[256]

    # While DMA responses are in flight, VSTORE VPU result → VMEM
    # VSTORE writes m12 → VMEM[512], m13 → VMEM[768] (via flag 0/8)
    LI    x8, 512
    VSTORE 12, x8, 0              # m12 (add b0) → VMEM[512] — 33 cycles
    DELAY 40
    VSTORE 13, x8, 8              # m13 (add b1) → VMEM[768] — 33 cycles
    DELAY 40

    # DMA may still be running (extended by backpressure).  Wait.
    DMA.WAIT  0
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# PHASE 6: Store VPU add result → DRAM (from VMEM[512], VMEM[768])
#          Also store C0 to VMEM for later DMA.STORE + VLOAD overlap.
# ═══════════════════════════════════════════════════════════════

    # DMA.STORE add result: VMEM[512] → DRAM, VMEM[768] → DRAM
    LI    x8, 512
    LI    x10, 0x3C00
    DMA.STORE x10, x8, x12, 0     # ch0: VMEM[512] → DRAM[add_b0]
    LI    x9, 768
    LI    x10, 0x4000
    DMA.STORE x10, x9, x12, 1     # ch1: VMEM[768] → DRAM[add_b1]
    DMA.WAIT  0
    DMA.WAIT  1

    # Store C0 (m4,m5) to VMEM[1024], VMEM[1280] for next phase
    LI    x8, 1024
    VSTORE 4, x8, 0               # m4 (C0 lo) → VMEM[1024]
    DELAY 40
    LI    x9, 1280
    VSTORE 5, x9, 0               # m5 (C0 hi) → VMEM[1280]
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE 7: READ-PORT BACKPRESSURE [RD-BP]
#
#   Issue DMA.STORE (reads VMEM → DRAM) for C0, then IMMEDIATELY
#   start VLOAD (reads VMEM → TRF) for X2.
#
#   DMA.STORE pre-reads VMEM lines into storeDataQueue before
#   issuing TileLink puts.  VLOAD also reads VMEM.  Both hit
#   the read port — VLOAD wins (LSU priority), DMA read sequencer
#   stalls via vmemReadGrant denial.
#
#   DMA reads VMEM[1024..1279] in bank 0.
#   VLOAD reads VMEM[0..255] in bank 0.
#   Both target the same bank-0 read port → DMA read sequencer stalls.
# ═══════════════════════════════════════════════════════════════

    # Issue DMA.STORE first — DMA starts reading VMEM immediately
    LI    x8, 1024
    LI    x10, 0x2C00
    DMA.STORE x10, x8, x12, 2     # ch2: VMEM[1024] → DRAM[C0_lo]

    # Immediately start VLOAD — both read VMEM concurrently!
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0                 # m0 ← X2 from VMEM[0] — 33 cycles
    DELAY 40

    # Second pair: DMA.STORE C0 hi + VLOAD W2
    LI    x9, 1280
    LI    x10, 0x3000
    DMA.STORE x10, x9, x12, 3     # ch3: VMEM[1280] → DRAM[C0_hi]

    LI    x7, 256
    VLOAD 1, x7, 0                 # m1 ← W2 (raw) from VMEM[256]
    DELAY 40

    # Wait for DMA stores (may be delayed by read-port backpressure)
    DMA.WAIT  2
    DMA.WAIT  3

# ═══════════════════════════════════════════════════════════════
# PHASE 8: Second MXU0 compute — C2 = X2 @ W2  [CHAIN]
#          X2 now in m0 (VLOAD-ed during contention).
#          W2 raw in m1, needs transpose + push.
# ═══════════════════════════════════════════════════════════════

    # Transpose W2: m1 → XLU → m2
    VTRPOSE.XLU 2, 1
    DELAY 80

    # MXU0: push new weights + zero bias.
    # The chained second pass also uses non-accumulating MATMUL so it cannot
    # inherit stale psums from the earlier C0 run.
    VMATPUSH.W.MXU0       0, 2
    VMATPUSH.ACC.FP8.MXU0 0, 3
    DELAY 40 

    # MXU0 compute C2 = X2 @ W2
    VMATMUL.MXU0      0, 0, 0
    DELAY 120

    # Pop C2 → m4,m5
    VMATPOP.BF16.MXU0  4, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE 9: Store remaining results — C1, C2 → DRAM
#          Reuse VMEM[512..767] and VMEM[768..1023] as staging.
# ═══════════════════════════════════════════════════════════════

    # Store C1 (m14,m15) → VMEM → DRAM
    LI    x8, 512
    VSTORE 14, x8, 0              # m14 (C1 lo) → VMEM[512]
    DELAY 40
    LI    x9, 768
    VSTORE 15, x9, 0              # m15 (C1 hi) → VMEM[768]
    DELAY 40

    LI    x8, 512
    LI    x10, 0x3400
    DMA.STORE x10, x8, x12, 0     # ch0: VMEM[512] → DRAM[C1_lo]
    LI    x9, 768
    LI    x10, 0x3800
    DMA.STORE x10, x9, x12, 1     # ch1: VMEM[768] → DRAM[C1_hi]
    DMA.WAIT  0
    DMA.WAIT  1

    # Store C2 (m4,m5) → VMEM → DRAM
    LI    x8, 512
    VSTORE 4, x8, 0               # m4 (C2 lo) → VMEM[512]
    DELAY 40
    LI    x9, 768
    VSTORE 5, x9, 0               # m5 (C2 hi) → VMEM[768]
    DELAY 40

    LI    x8, 512
    LI    x10, 0x4400
    DMA.STORE x10, x8, x12, 0     # ch0: VMEM[512] → DRAM[C2_lo]
    LI    x9, 768
    LI    x10, 0x4800
    DMA.STORE x10, x9, x12, 1     # ch1: VMEM[768] → DRAM[C2_hi]
    DMA.WAIT  0
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

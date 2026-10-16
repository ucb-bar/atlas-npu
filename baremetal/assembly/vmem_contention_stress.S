# vmem_contention_stress.S — VMEM bank arbitration + DMA pipeline stress test
# @TIMEOUT 1500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vmem_contention_stress.py 42
#
# Exercises every VMEM contention path:
#   [WW-BP] DMA.LOAD responses + VSTORE → shared 1RW port backpressure
#   [RW-BP] DMA.STORE pre-reads + VSTORE → shared 1RW port backpressure
#   [RR-BP] DMA.STORE pre-reads + VLOAD → shared 1RW port backpressure
#   [PIPE]  4 DMA channels queued back-to-back before any WAIT
#   [3WAY]  MXU0 + MXU1 + VPU compute during DMA quiescence
#   [CHAIN] MXU0 result popped, stored to VMEM, DMA-stored to DRAM,
#           new data DMA-loaded and VLOAD-ed during contention,
#           then re-computed on MXU0 → end-to-end pipeline
#
# VMEM bank conflict analysis (6 banks, 256 KiB / 65 536-word banking):
#   Each aligned 1024-byte tile occupies one tensor slot inside a single bank.
#   In this test, the contending VMEM regions are intentionally packed into
#   bank 0.  The low VMEM slots below are all < 65536 words, which is the
#   word span of one bank in the 1.5 MiB / 6-bank configuration.  Same-bank
#   overlaps therefore contend for the single 1RW port and force the
#   lower-priority DMA client to stall.
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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024
    seli e0, 127
    seli e1, 127

# ═══════════════════════════════════════════════════════════════
# PHASE 0: Load persistent data — zero bias + VPU inputs
#          Sequential DMA, each tile staged through VMEM[0]
# ═══════════════════════════════════════════════════════════════

    # bias → m3
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m3, 0(x6)
    DELAY 33

    # A_vpu b0 → m8, b1 → m9 (register pair via flag 0/8)
    ADDI  x6, x0, 0
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI    x7, 256
    LI    x10, 0x90001800
    dma.load x7, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m8, 0(x6)
    DELAY 33
    vload m9, 8(x6)
    DELAY 33

    # B_vpu b0 → m10, b1 → m11
    ADDI  x6, x0, 0
    LI    x10, 0x90001C00
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI    x7, 256
    LI    x10, 0x90002000
    dma.load x7, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m10, 0(x6)
    DELAY 33
    vload m11, 8(x6)
    DELAY 33

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
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0     # ch0: X0 → VMEM[0]

    LI    x7, 256
    LI    x10, 0x90000400
    dma.load x7, x10, x12, ch1     # ch1: W0 → VMEM[256]

    LI    x8, 512
    LI    x10, 0x90000800
    dma.load x8, x10, x12, ch2     # ch2: X1 → VMEM[512]

    LI    x9, 768
    LI    x10, 0x90000C00
    dma.load x9, x10, x12, ch3     # ch3: W1 → VMEM[768]

    # Commands are queued; DMA processes them sequentially on TileLink.
    # Responses may interleave.  Wait for all.
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# PHASE 2: VLOAD data + set up MXU0 and MXU1 weights
# ═══════════════════════════════════════════════════════════════

    # X0 → m0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

    # W0 (raw) → m1 → XLU transpose → m2
    LI    x7, 256
    vload m1, 0(x7)
    DELAY 33
    vtrpose.xlu m2, m1
    DELAY 64

    # MXU0: push weights + zero bias.
    # Use non-accumulating MATMUL for this single-tile op so the compute
    # always starts from a fresh accum buffer.
    vmatpush.weight.mxu0 w0, m2
    vmatpush.acc.fp8.mxu0 acc0, m3

    # X1 → m6
    LI    x8, 512
    vload m6, 0(x8)
    DELAY 33

    # W1 (raw) → m1 (reuse) → XLU transpose → m7
    LI    x9, 768
    vload m1, 0(x9)
    DELAY 33
    vtrpose.xlu m7, m1
    DELAY 64

    # MXU1: push weights + zero bias.
    # As above, the compute itself uses non-accumulating MATMUL.
    vmatpush.weight.mxu1 w0, m7
    vmatpush.acc.fp8.mxu1 acc0, m3
    DELAY 31 

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Triple-unit compute — MXU0 + MXU1 + VPU concurrent
#          [3WAY] All three functional units fire on independent
#          TRF banks with no resource conflicts.
#
#   MXU0: acc0 = X0 (m0) @ W0_t (wbuf0)
#   MXU1: acc0 = X1 (m6) @ W1_t (wbuf0)
#   VPU:  m12,m13 = vadd(m8,m10) = vadd(A_vpu, B_vpu)
# ═══════════════════════════════════════════════════════════════

    vmatmul.mxu0 acc0, m0, w0
    vmatmul.mxu1 acc0, m6, w0
    vadd.bf16 m12, m8, m10
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# PHASE 4: Pop MXU results
#   MXU0 → m4,m5 (C0)
#   MXU1 → m14,m15 (C1)
# ═══════════════════════════════════════════════════════════════

    vmatpop.bf16.acc.mxu0 m4, acc0
    vmatpop.bf16.acc.mxu1 m14, acc0
    DELAY 31

# ═══════════════════════════════════════════════════════════════
# PHASE 5: WRITE/WRITE SHARED-PORT BACKPRESSURE [WW-BP]
#
#   Issue DMA.LOAD for X2 and W2 (writes to VMEM on response path),
#   then IMMEDIATELY start VSTORE of VPU result (also writes to VMEM).
#
#   DMA.LOAD responses arrive after TileLink round-trip (~10-20 cyc).
#   VSTORE runs for 33 cycles.  During overlap, both need the same 1RW
#   bank port.  VSTORE wins (LSU priority), DMA stalls via vmemWriteGrant.
#
#   DMA writes to VMEM[0..255] and VMEM[256..511] in bank 0.
#   VSTORE writes to VMEM[512..767] and VMEM[768..1023] in bank 0.
#   All writes target the same bank-0 1RW port → maximum backpressure.
# ═══════════════════════════════════════════════════════════════

    # Enqueue DMA loads (start TileLink traffic immediately)
    ADDI  x6, x0, 0
    LI    x10, 0x90002400
    dma.load x6, x10, x12, ch0     # ch0: X2 → VMEM[0]

    LI    x7, 256
    LI    x10, 0x90002800
    dma.load x7, x10, x12, ch1     # ch1: W2 → VMEM[256]

    # While DMA responses are in flight, VSTORE VPU result → VMEM
    # VSTORE writes m12 → VMEM[512], m13 → VMEM[768] (via flag 0/8)
    LI    x8, 512
    vstore m12, 0(x8)              # m12 (add b0) → VMEM[512] — 33 cycles
    DELAY 33
    vstore m13, 8(x8)              # m13 (add b1) → VMEM[768] — 33 cycles
    DELAY 33

    # DMA may still be running (extended by backpressure).  Wait.
    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# PHASE 6: READ/WRITE SHARED-PORT BACKPRESSURE [RW-BP]
#
#          Store VPU add result → DRAM (DMA reads VMEM[512], VMEM[768]),
#          then immediately store C0 to VMEM[1024], VMEM[1280].
#
#          DMA.STORE pre-reads and VSTORE writes hit different addresses in
#          the same bank.  Under 1RW banking they still contend for the one
#          bank port, so VSTORE wins and DMA stalls via vmemReadGrant.
#          The C0 stores also set up the next DMA.STORE + VLOAD overlap.
# ═══════════════════════════════════════════════════════════════

    # DMA.STORE add result: VMEM[512] → DRAM, VMEM[768] → DRAM.
    # Do not wait yet; overlap the DMA pre-reads with the C0 VSTOREs below.
    LI    x8, 512
    LI    x10, 0x90003C00
    dma.store x10, x8, x12, ch0     # ch0: VMEM[512] → DRAM[add_b0]
    LI    x9, 768
    LI    x10, 0x90004000
    dma.store x10, x9, x12, ch1     # ch1: VMEM[768] → DRAM[add_b1]

    # Store C0 (m4,m5) to VMEM[1024], VMEM[1280] for next phase
    LI    x8, 1024
    vstore m4, 0(x8)               # m4 (C0 lo) → VMEM[1024]
    DELAY 33
    LI    x9, 1280
    vstore m5, 0(x9)               # m5 (C0 hi) → VMEM[1280]
    DELAY 33

    # DMA stores may have been stretched by the C0 VSTOREs.
    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# PHASE 7: READ/READ SHARED-PORT BACKPRESSURE [RR-BP]
#
#   Issue DMA.STORE (reads VMEM → DRAM) for C0, then IMMEDIATELY
#   start VLOAD (reads VMEM → TRF) for X2.
#
#   DMA.STORE pre-reads VMEM lines into storeDataQueue before
#   issuing TileLink puts.  VLOAD also reads VMEM.  Both hit
#   the same 1RW bank port.  VLOAD wins (LSU priority), DMA read sequencer
#   stalls via vmemReadGrant denial.
#
#   DMA reads VMEM[1024..1279] in bank 0.
#   VLOAD reads VMEM[0..255] in bank 0.
#   Both target the same bank-0 1RW port → DMA read sequencer stalls.
# ═══════════════════════════════════════════════════════════════

    # Issue DMA.STORE first — DMA starts reading VMEM immediately
    LI    x8, 1024
    LI    x10, 0x90002C00
    dma.store x10, x8, x12, ch2     # ch2: VMEM[1024] → DRAM[C0_lo]

    # Immediately start VLOAD — both read VMEM concurrently!
    ADDI  x6, x0, 0
    vload m0, 0(x6)                 # m0 ← X2 from VMEM[0] — 33 cycles
    DELAY 33

    # Second pair: DMA.STORE C0 hi + VLOAD W2
    LI    x9, 1280
    LI    x10, 0x90003000
    dma.store x10, x9, x12, ch3     # ch3: VMEM[1280] → DRAM[C0_hi]

    LI    x7, 256
    vload m1, 0(x7)                 # m1 ← W2 (raw) from VMEM[256]
    DELAY 33

    # Wait for DMA stores (may be delayed by shared-port backpressure)
    dma.wait ch2
    dma.wait ch3

# ═══════════════════════════════════════════════════════════════
# PHASE 8: Second MXU0 compute — C2 = X2 @ W2  [CHAIN]
#          X2 now in m0 (VLOAD-ed during contention).
#          W2 raw in m1, needs transpose + push.
# ═══════════════════════════════════════════════════════════════

    # Transpose W2: m1 → XLU → m2
    vtrpose.xlu m2, m1
    DELAY 64

    # MXU0: push new weights + zero bias.
    # The chained second pass also uses non-accumulating MATMUL so it cannot
    # inherit stale psums from the earlier C0 run.
    vmatpush.weight.mxu0 w0, m2
    vmatpush.acc.fp8.mxu0 acc0, m3
    DELAY 31 

    # MXU0 compute C2 = X2 @ W2
    vmatmul.mxu0 acc0, m0, w0
    DELAY 95

    # Pop C2 → m4,m5
    vmatpop.bf16.acc.mxu0 m4, acc0
    DELAY 31

# ═══════════════════════════════════════════════════════════════
# PHASE 9: Store remaining results — C1, C2 → DRAM
#          Reuse VMEM[512..767] and VMEM[768..1023] as staging.
# ═══════════════════════════════════════════════════════════════

    # Store C1 (m14,m15) → VMEM → DRAM
    LI    x8, 512
    vstore m14, 0(x8)              # m14 (C1 lo) → VMEM[512]
    DELAY 33
    LI    x9, 768
    vstore m15, 0(x9)              # m15 (C1 hi) → VMEM[768]
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90003400
    dma.store x10, x8, x12, ch0     # ch0: VMEM[512] → DRAM[C1_lo]
    LI    x9, 768
    LI    x10, 0x90003800
    dma.store x10, x9, x12, ch1     # ch1: VMEM[768] → DRAM[C1_hi]
    dma.wait ch0
    dma.wait ch1

    # Store C2 (m4,m5) → VMEM → DRAM
    LI    x8, 512
    vstore m4, 0(x8)               # m4 (C2 lo) → VMEM[512]
    DELAY 33
    LI    x9, 768
    vstore m5, 0(x9)               # m5 (C2 hi) → VMEM[768]
    DELAY 33

    LI    x8, 512
    LI    x10, 0x90004400
    dma.store x10, x8, x12, ch0     # ch0: VMEM[512] → DRAM[C2_lo]
    LI    x9, 768
    LI    x10, 0x90004800
    dma.store x10, x9, x12, ch1     # ch1: VMEM[768] → DRAM[C2_hi]
    dma.wait ch0
    dma.wait ch1

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL

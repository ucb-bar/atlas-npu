# dma_8chan_burst.S — All 8 DMA channels in flight simultaneously
# @TIMEOUT 1500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_8chan_burst.py 42
#
# Exercises:
#   [8LD]  8 DMA.LOAD commands queued back-to-back before any WAIT
#   [8ST]  8 DMA.STORE commands queued back-to-back before any WAIT
#   [ECHO] 4 tiles echoed through DMA LOAD→STORE to verify data integrity
#   [COMP] MXU1 + VPU compute on DMA-loaded data verifies functional path
#
# The DMA engine's circular command queue (8 slots) fills completely.
# Commands process sequentially through requestIdx/responseIdx, but all
# 8 channelBusy flags are set simultaneously.  The per-ID bitmap in
# TileLinkAdapter prevents source-ID reuse across pipelined TileLink beats.
#
# DRAM layout — inputs (8 tiles, channels 0–7):
#   0x0000  X         (1024B FP8) — MXU1 activations
#   0x0400  W         (1024B FP8) — MXU1 weights
#   0x0800  bias      (1024B FP8, all 0x00)
#   0x0C00  A_vpu b0  (1024B BF16) — VPU input A lanes 0-15
#   0x1000  A_vpu b1  (1024B BF16) — VPU input A lanes 16-31
#   0x1400  B_vpu b0  (1024B BF16) — VPU input B lanes 0-15
#   0x1800  B_vpu b1  (1024B BF16) — VPU input B lanes 16-31
#   0x1C00  echo_tile (1024B FP8) — extra tile for echo verification
#
# DRAM layout — outputs:
#   0x2000  C0 lo     (1024B BF16) — MXU1 result lanes 0-15
#   0x2400  C0 hi     (1024B BF16) — MXU1 result lanes 16-31
#   0x2800  add b0    (1024B BF16) — VPU vadd result lanes 0-15
#   0x2C00  add b1    (1024B BF16) — VPU vadd result lanes 16-31
#   0x3000  echo0     (1024B) — echo of A_vpu b1
#   0x3400  echo1     (1024B) — echo of B_vpu b0
#   0x3800  echo2     (1024B) — echo of B_vpu b1
#   0x3C00  echo3     (1024B) — echo of echo_tile
#
# VMEM slot map (256 words = 1024B per slot):
#   [0..255]       slot 0 — X  (then C0 lo after VSTORE)
#   [256..511]     slot 1 — W  (then C0 hi after VSTORE)
#   [512..767]     slot 2 — bias (then add b0 after VSTORE)
#   [768..1023]    slot 3 — A_vpu b0 (then add b1 after VSTORE)
#   [1024..1279]   slot 4 — A_vpu b1 (echo)
#   [1280..1535]   slot 5 — B_vpu b0 (echo)
#   [1536..1791]   slot 6 — B_vpu b1 (echo)
#   [1792..2047]   slot 7 — echo_tile (echo)
#
# TRF banks:
#   m0  X activations     m8,9   A_vpu pair
#   m1  W raw staging     m10,11 B_vpu pair
#   m2  W transposed      m12,13 VPU result
#   m3  bias (zero)       m4,5   MXU1 pop result

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024
    seli e0, 127
    seli e1, 127

# ═══════════════════════════════════════════════════════════════
# PHASE 1: 8-CHANNEL DMA.LOAD BURST [8LD]
#
# All 8 commands enqueued before any DMA.WAIT.  The command queue
# fills completely (enqueueIdx wraps 0→1→…→7→0).  requestIdx
# processes them sequentially via TileLink.
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0     # ch0: X → VMEM[0]

    LI    x6, 256
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1     # ch1: W → VMEM[256]

    LI    x6, 512
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2     # ch2: bias → VMEM[512]

    LI    x6, 768
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch3     # ch3: A_vpu b0 → VMEM[768]

    LI    x6, 1024
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch4     # ch4: A_vpu b1 → VMEM[1024]

    LI    x6, 1280
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch5     # ch5: B_vpu b0 → VMEM[1280]

    LI    x6, 1536
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch6     # ch6: B_vpu b1 → VMEM[1536]

    LI    x6, 1792
    LI    x10, 0x90001C00
    dma.load x6, x10, x12, ch7     # ch7: echo_tile → VMEM[1792]

    # All 8 channelBusy flags are set.  Wait for each.
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

# ═══════════════════════════════════════════════════════════════
# PHASE 2: VLOAD data + MXU1 setup
# ═══════════════════════════════════════════════════════════════

    # X → m0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

    # W → m1 → XLU → m2
    LI    x6, 256
    vload m1, 0(x6)
    DELAY 33
    vtrpose.xlu m2, m1
    DELAY 64

    # bias → m3
    LI    x6, 512
    vload m3, 0(x6)
    DELAY 33

    # Push MXU1 weights + bias
    vmatpush.weight.mxu1 w0, m2
    vmatpush.acc.fp8.mxu1 acc0, m3

    # A_vpu pair → m8,m9 (from VMEM[768], flag 0/8)
    LI    x6, 768
    vload m8, 0(x6)
    DELAY 33
    vload m9, 8(x6)
    DELAY 33

    # B_vpu pair → m10,m11 (from VMEM[1280], flag 0/8)
    LI    x6, 1280
    vload m10, 0(x6)
    DELAY 33
    vload m11, 8(x6)
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# PHASE 3: MXU1 + VPU compute
# ═══════════════════════════════════════════════════════════════

    vmatmul.acc.mxu1 acc0, m0, w0
    vadd.bf16 m12, m8, m10
    DELAY 64

    vmatpop.bf16.acc.mxu1 m4, acc0
    DELAY 31

# ═══════════════════════════════════════════════════════════════
# PHASE 4: VSTORE results to VMEM (overwrite input slots 0–3)
# ═══════════════════════════════════════════════════════════════

    # C0 lo → VMEM[0], C0 hi → VMEM[256]
    ADDI  x6, x0, 0
    vstore m4, 0(x6)
    DELAY 33
    LI    x6, 256
    vstore m5, 0(x6)
    DELAY 33

    # add b0 → VMEM[512], add b1 → VMEM[768]
    LI    x6, 512
    vstore m12, 0(x6)
    DELAY 33
    vstore m13, 8(x6)
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# PHASE 5: 8-CHANNEL DMA.STORE BURST [8ST]
#
# VMEM slots 0–3 now hold compute results.
# VMEM slots 4–7 still hold original DMA-loaded data (echoes).
# All 8 DMA.STORE commands enqueued before any DMA.WAIT.
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90002000
    dma.store x10, x6, x12, ch0     # ch0: VMEM[0] → DRAM[C0 lo]

    LI    x6, 256
    LI    x10, 0x90002400
    dma.store x10, x6, x12, ch1     # ch1: VMEM[256] → DRAM[C0 hi]

    LI    x6, 512
    LI    x10, 0x90002800
    dma.store x10, x6, x12, ch2     # ch2: VMEM[512] → DRAM[add b0]

    LI    x6, 768
    LI    x10, 0x90002C00
    dma.store x10, x6, x12, ch3     # ch3: VMEM[768] → DRAM[add b1]

    LI    x6, 1024
    LI    x10, 0x90003000
    dma.store x10, x6, x12, ch4     # ch4: VMEM[1024] → DRAM[echo0]

    LI    x6, 1280
    LI    x10, 0x90003400
    dma.store x10, x6, x12, ch5     # ch5: VMEM[1280] → DRAM[echo1]

    LI    x6, 1536
    LI    x10, 0x90003800
    dma.store x10, x6, x12, ch6     # ch6: VMEM[1536] → DRAM[echo2]

    LI    x6, 1792
    LI    x10, 0x90003C00
    dma.store x10, x6, x12, ch7     # ch7: VMEM[1792] → DRAM[echo3]

    # All 8 channelBusy flags set.  Wait.
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL
    
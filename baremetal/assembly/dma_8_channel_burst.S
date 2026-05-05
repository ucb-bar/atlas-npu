# dma_8chan_burst.S — All 8 DMA channels in flight simultaneously
# @TIMEOUT 1500000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_dma_8chan_burst.py 42
#
# Exercises:
#   [8LD]  8 DMA.LOAD commands queued back-to-back before any WAIT
#   [8ST]  8 DMA.STORE commands queued back-to-back before any WAIT
#   [ECHO] 4 tiles echoed through DMA LOAD→STORE to verify data integrity
#   [COMP] MXU0 + VPU compute on DMA-loaded data verifies functional path
#
# The DMA engine's circular command queue (8 slots) fills completely.
# Commands process sequentially through requestIdx/responseIdx, but all
# 8 channelBusy flags are set simultaneously.  The per-ID bitmap in
# TileLinkAdapter prevents source-ID reuse across pipelined TileLink beats.
#
# DRAM layout — inputs (8 tiles, channels 0–7):
#   0x0000  X         (1024B FP8) — MXU0 activations
#   0x0400  W         (1024B FP8) — MXU0 weights
#   0x0800  bias      (1024B FP8, all 0x00)
#   0x0C00  A_vpu b0  (1024B BF16) — VPU input A lanes 0-15
#   0x1000  A_vpu b1  (1024B BF16) — VPU input A lanes 16-31
#   0x1400  B_vpu b0  (1024B BF16) — VPU input B lanes 0-15
#   0x1800  B_vpu b1  (1024B BF16) — VPU input B lanes 16-31
#   0x1C00  echo_tile (1024B FP8) — extra tile for echo verification
#
# DRAM layout — outputs:
#   0x2000  C0 lo     (1024B BF16) — MXU0 result lanes 0-15
#   0x2400  C0 hi     (1024B BF16) — MXU0 result lanes 16-31
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
#   m3  bias (zero)       m4,5   MXU0 pop result

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127
    SELI  1, 127

# ═══════════════════════════════════════════════════════════════
# PHASE 1: 8-CHANNEL DMA.LOAD BURST [8LD]
#
# All 8 commands enqueued before any DMA.WAIT.  The command queue
# fills completely (enqueueIdx wraps 0→1→…→7→0).  requestIdx
# processes them sequentially via TileLink.
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0     # ch0: X → VMEM[0]

    LI    x6, 256
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 1     # ch1: W → VMEM[256]

    LI    x6, 512
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 2     # ch2: bias → VMEM[512]

    LI    x6, 768
    LI    x10, 0x0C00
    DMA.LOAD  x6, x10, x12, 3     # ch3: A_vpu b0 → VMEM[768]

    LI    x6, 1024
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 4     # ch4: A_vpu b1 → VMEM[1024]

    LI    x6, 1280
    LI    x10, 0x1400
    DMA.LOAD  x6, x10, x12, 5     # ch5: B_vpu b0 → VMEM[1280]

    LI    x6, 1536
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 6     # ch6: B_vpu b1 → VMEM[1536]

    LI    x6, 1792
    LI    x10, 0x1C00
    DMA.LOAD  x6, x10, x12, 7     # ch7: echo_tile → VMEM[1792]

    # All 8 channelBusy flags are set.  Wait for each.
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4
    DMA.WAIT  5
    DMA.WAIT  6
    DMA.WAIT  7

# ═══════════════════════════════════════════════════════════════
# PHASE 2: VLOAD data + MXU0 setup
# ═══════════════════════════════════════════════════════════════

    # X → m0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    # W → m1 → XLU → m2
    LI    x6, 256
    VLOAD 1, x6, 0
    DELAY 40
    VTRPOSE.XLU 2, 1
    DELAY 80

    # bias → m3
    LI    x6, 512
    VLOAD 3, x6, 0
    DELAY 40

    # Push MXU0 weights + bias
    VMATPUSH.W.MXU0       0, 2
    VMATPUSH.ACC.FP8.MXU0 0, 3

    # A_vpu pair → m8,m9 (from VMEM[768], flag 0/8)
    LI    x6, 768
    VLOAD 8, x6, 0
    DELAY 40
    VLOAD 9, x6, 8
    DELAY 40

    # B_vpu pair → m10,m11 (from VMEM[1280], flag 0/8)
    LI    x6, 1280
    VLOAD 10, x6, 0
    DELAY 40
    VLOAD 11, x6, 8
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE 3: MXU0 + VPU compute
# ═══════════════════════════════════════════════════════════════

    VMATMUL.ACC.MXU0  0, 0, 0
    VADD.BF16         12, 8, 10
    DELAY 150

    VMATPOP.BF16.MXU0  4, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE 4: VSTORE results to VMEM (overwrite input slots 0–3)
# ═══════════════════════════════════════════════════════════════

    # C0 lo → VMEM[0], C0 hi → VMEM[256]
    ADDI  x6, x0, 0
    VSTORE 4, x6, 0
    DELAY 40
    LI    x6, 256
    VSTORE 5, x6, 0
    DELAY 40

    # add b0 → VMEM[512], add b1 → VMEM[768]
    LI    x6, 512
    VSTORE 12, x6, 0
    DELAY 40
    VSTORE 13, x6, 8
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# PHASE 5: 8-CHANNEL DMA.STORE BURST [8ST]
#
# VMEM slots 0–3 now hold compute results.
# VMEM slots 4–7 still hold original DMA-loaded data (echoes).
# All 8 DMA.STORE commands enqueued before any DMA.WAIT.
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 0     # ch0: VMEM[0] → DRAM[C0 lo]

    LI    x6, 256
    LI    x10, 0x2400
    DMA.STORE x10, x6, x12, 1     # ch1: VMEM[256] → DRAM[C0 hi]

    LI    x6, 512
    LI    x10, 0x2800
    DMA.STORE x10, x6, x12, 2     # ch2: VMEM[512] → DRAM[add b0]

    LI    x6, 768
    LI    x10, 0x2C00
    DMA.STORE x10, x6, x12, 3     # ch3: VMEM[768] → DRAM[add b1]

    LI    x6, 1024
    LI    x10, 0x3000
    DMA.STORE x10, x6, x12, 4     # ch4: VMEM[1024] → DRAM[echo0]

    LI    x6, 1280
    LI    x10, 0x3400
    DMA.STORE x10, x6, x12, 5     # ch5: VMEM[1280] → DRAM[echo1]

    LI    x6, 1536
    LI    x10, 0x3800
    DMA.STORE x10, x6, x12, 6     # ch6: VMEM[1536] → DRAM[echo2]

    LI    x6, 1792
    LI    x10, 0x3C00
    DMA.STORE x10, x6, x12, 7     # ch7: VMEM[1792] → DRAM[echo3]

    # All 8 channelBusy flags set.  Wait.
    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4
    DMA.WAIT  5
    DMA.WAIT  6
    DMA.WAIT  7

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL
    
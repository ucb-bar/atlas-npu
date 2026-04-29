# sas_8dma_rvv_vmem.S — 8-channel DMA echo + Saturn RVV in VMEM bank 7
#
# T2 of the Saturn↔Atlas memory stress suite. Atlas drives all 8 DMA
# channels through banks 0-6 (channel 7 reuses bank 0 at a different
# slot). VMEM bank 7 is reserved for Saturn — the host driver runs an
# RVV stride-1 sweep there during the Atlas DMA window.
#
# Stress shape: SBUS xbar fairness (Saturn TL master vs Atlas DMA TL
# master) + per-bank arbiter independence (8 banks, 7 used by DMA).
# Pure DMA echo (no VPU/MXU compute) — keeps the test focused on
# memory traffic, not arithmetic. The compute path is exercised by
# T1 (LSU traffic) and T5 (full-court press), so leaving it out here
# is intentional.
#
# Bank index = word_idx >> 15 (8 banks of 32 768 words each).
# Each tile is 1024 B = 32 lines = 256 words. Tile slots:
#   ch0 → word 0          (bank 0, slot 0)
#   ch1 → word 32768      (bank 1)
#   ch2 → word 65536      (bank 2)
#   ch3 → word 98304      (bank 3)
#   ch4 → word 131072     (bank 4)
#   ch5 → word 163840     (bank 5)
#   ch6 → word 196608     (bank 6)
#   ch7 → word 256        (bank 0, slot 1)
#
# DRAM (DMA base 0x90000000):
#   IN  tiles  at 0x0000, 0x0400, ..., 0x1C00 (8 × 1024 B)
#   OUT tiles  at 0x2000, 0x2400, ..., 0x3C00 (8 × 1024 B, expected = IN)
#
# Outer round loop: each iteration does a full LOAD-burst + STORE-burst.
# N_ROUNDS rounds give Saturn a wider overlap window so the host's
# iters_done is large enough to demonstrate sustained concurrent
# RVV-into-VMEM traffic, not a one-shot.
#
# End: dbg0 = 1, halt_reason = ECALL.

    LI    x5,  0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 1024            # transfer size

    LI    x29, 8               # N_ROUNDS
    LI    x28, 0               # round counter

    # ── Phase marker 10: configured, round_loop entry ──
    ADDI  x1, x0, 10
    CSRW  x1, 0xC10            # dbg0 = 10

round_loop:
    # dbg1 = current round number (host can read post-mortem)
    CSRW  x28, 0xC11

# ── Phase 1: 8-channel DMA.LOAD burst (DRAM IN → VMEM banks) ──────────

    LI    x6,  0
    LI    x10, 0x0000
    DMA.LOAD x6, x10, x12, 0    # ch0 → bank 0 slot 0

    LI    x6,  32768
    LI    x10, 0x0400
    DMA.LOAD x6, x10, x12, 1    # ch1 → bank 1

    LI    x6,  65536
    LI    x10, 0x0800
    DMA.LOAD x6, x10, x12, 2    # ch2 → bank 2

    LI    x6,  98304
    LI    x10, 0x0C00
    DMA.LOAD x6, x10, x12, 3    # ch3 → bank 3

    LI    x6,  131072
    LI    x10, 0x1000
    DMA.LOAD x6, x10, x12, 4    # ch4 → bank 4

    LI    x6,  163840
    LI    x10, 0x1400
    DMA.LOAD x6, x10, x12, 5    # ch5 → bank 5

    LI    x6,  196608
    LI    x10, 0x1800
    DMA.LOAD x6, x10, x12, 6    # ch6 → bank 6

    LI    x6,  256
    LI    x10, 0x1C00
    DMA.LOAD x6, x10, x12, 7    # ch7 → bank 0 slot 1

    # ── Phase marker 20: 8 DMA.LOADs enqueued, before WAITs ──
    ADDI  x1, x0, 20
    CSRW  x1, 0xC10

    # Drain all 8 channels.
    DMA.WAIT 0
    DMA.WAIT 1
    DMA.WAIT 2
    DMA.WAIT 3
    DMA.WAIT 4
    DMA.WAIT 5
    DMA.WAIT 6
    DMA.WAIT 7

    # ── Phase marker 30: all DMA.LOADs drained ──
    ADDI  x1, x0, 30
    CSRW  x1, 0xC10

# ── Phase 2: 8-channel DMA.STORE burst (VMEM banks → DRAM OUT) ────────

    LI    x6,  0
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 0

    LI    x6,  32768
    LI    x10, 0x2400
    DMA.STORE x10, x6, x12, 1

    LI    x6,  65536
    LI    x10, 0x2800
    DMA.STORE x10, x6, x12, 2

    LI    x6,  98304
    LI    x10, 0x2C00
    DMA.STORE x10, x6, x12, 3

    LI    x6,  131072
    LI    x10, 0x3000
    DMA.STORE x10, x6, x12, 4

    LI    x6,  163840
    LI    x10, 0x3400
    DMA.STORE x10, x6, x12, 5

    LI    x6,  196608
    LI    x10, 0x3800
    DMA.STORE x10, x6, x12, 6

    LI    x6,  256
    LI    x10, 0x3C00
    DMA.STORE x10, x6, x12, 7

    # ── Phase marker 40: 8 DMA.STOREs enqueued, before WAITs ──
    ADDI  x1, x0, 40
    CSRW  x1, 0xC10

    DMA.WAIT 0
    DMA.WAIT 1
    DMA.WAIT 2
    DMA.WAIT 3
    DMA.WAIT 4
    DMA.WAIT 5
    DMA.WAIT 6
    DMA.WAIT 7

    # ── Phase marker 50: all DMA.STOREs drained, end of round ──
    ADDI  x1, x0, 50
    CSRW  x1, 0xC10

    # Outer round loop counter.
    ADDI  x28, x28, 1
    BLT   x28, x29, round_loop
    NOP
    NOP

# ── Done: dbg0 = 1, ECALL ─────────────────────────────────────────────
# (host expects dbg0 == 1 for PASS; intermediate phase markers above
# overwrite dbg0 during execution — final value 1 means kernel ran to
# completion. dbg1 holds the last round counter for further diagnostics.)

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

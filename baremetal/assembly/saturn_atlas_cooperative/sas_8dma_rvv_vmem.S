# sas_8dma_rvv_vmem.S — 8-channel DMA echo + Saturn RVV in the highest VMEM bank
#
# T2 of the Saturn↔Atlas memory stress suite. Atlas drives all 8 DMA
# channels through a compact 6-bank layout. Under the current 256 KiB /
# 65 536-word banking, those DMA bases decode as banks 0, 0, 1, 2, 2, 3, 4,
# and 0. The highest-numbered VMEM bank (bank 5) is reserved for Saturn on the
# host side, and the driver runs an RVV stride-1 sweep there during the Atlas
# DMA window.
#
# Stress shape: SBUS xbar fairness (Saturn TL master vs Atlas DMA TL
# master) + per-bank arbiter independence between the DMA-touched banks above
# and Saturn's disjoint highest-bank region.
# Pure DMA echo (no VPU/MXU compute) — keeps the test focused on
# memory traffic, not arithmetic. The compute path is exercised by
# T1 (LSU traffic) and T5 (full-court press), so leaving it out here
# is intentional.
#
# Bank index = word_idx / 65 536 (6 banks of 65 536 words = 256 KiB each).
# Each tile is 1024 B = 32 lines = 256 words. Tile slots:
#   ch0 → word 0          (bank 0, slot 0)
#   ch1 → word 32768      (bank 0)
#   ch2 → word 65536      (bank 1)
#   ch3 → word 131072     (bank 2)
#   ch4 → word 163840     (bank 2)
#   ch5 → word 196608     (bank 3)
#   ch6 → word 262144     (bank 4)
#   ch7 → word 256        (bank 0, slot 1)
#   Saturn host RVV region starts at current bank 5 base: word 327680
#   (byte offset 0x140000 from ATLAS_VMEM_BASE).
#
# DRAM (DMA base 0x00000001):
#   IN  tiles  at 0x0000, 0x0400, ..., 0x1C00 (8 × 1024 B)
#   OUT tiles  at 0x2000, 0x2400, ..., 0x3C00 (8 × 1024 B, expected = IN)
#
# Outer round loop: each iteration does a full LOAD-burst + STORE-burst.
# N_ROUNDS rounds give Saturn a wider overlap window so the host's
# iters_done is large enough to demonstrate sustained concurrent
# RVV-into-VMEM traffic, not a one-shot.
#
# End: dbg0 = 1, halt_reason = ECALL.

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024            # transfer size

    LI    x29, 8               # N_ROUNDS
    LI    x28, 0               # round counter

    # ── Phase marker 10: configured, round_loop entry ──
    ADDI  x1, x0, 10
    csrrw x0, 0x800, x1            # dbg0 = 10

round_loop:
    # dbg1 = current round number (host can read post-mortem)
    csrrw x0, 0x801, x28

# ── Phase 1: 8-channel DMA.LOAD burst (DRAM IN → VMEM banks) ──────────

    LI    x6,  0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0    # ch0 → bank 0 slot 0

    LI    x6,  32768
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1    # ch1 → bank 0

    LI    x6,  65536
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2    # ch2 → bank 1

    LI    x6,  131072
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch3    # ch3 → bank 2

    LI    x6,  163840
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch4    # ch4 → bank 2

    LI    x6,  196608
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch5    # ch5 → bank 3

    LI    x6,  262144
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch6    # ch6 → bank 4

    LI    x6,  256
    LI    x10, 0x90001C00
    dma.load x6, x10, x12, ch7    # ch7 → bank 0 slot 1

    # ── Phase marker 20: 8 DMA.LOADs enqueued, before WAITs ──
    ADDI  x1, x0, 20
    csrrw x0, 0x800, x1

    # Drain all 8 channels.
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

    # ── Phase marker 30: all DMA.LOADs drained ──
    ADDI  x1, x0, 30
    csrrw x0, 0x800, x1

# ── Phase 2: 8-channel DMA.STORE burst (VMEM banks → DRAM OUT) ────────

    LI    x6,  0
    LI    x10, 0x90002000
    dma.store x10, x6, x12, ch0

    LI    x6,  32768
    LI    x10, 0x90002400
    dma.store x10, x6, x12, ch1

    LI    x6,  65536
    LI    x10, 0x90002800
    dma.store x10, x6, x12, ch2

    LI    x6,  131072
    LI    x10, 0x90002C00
    dma.store x10, x6, x12, ch3

    LI    x6,  163840
    LI    x10, 0x90003000
    dma.store x10, x6, x12, ch4

    LI    x6,  196608
    LI    x10, 0x90003400
    dma.store x10, x6, x12, ch5

    LI    x6,  262144
    LI    x10, 0x90003800
    dma.store x10, x6, x12, ch6

    LI    x6,  256
    LI    x10, 0x90003C00
    dma.store x10, x6, x12, ch7

    # ── Phase marker 40: 8 DMA.STOREs enqueued, before WAITs ──
    ADDI  x1, x0, 40
    csrrw x0, 0x800, x1

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

    # ── Phase marker 50: all DMA.STOREs drained, end of round ──
    ADDI  x1, x0, 50
    csrrw x0, 0x800, x1

    # Outer round loop counter.
    ADDI  x28, x28, 1
    BLT   x28, x29, round_loop

# ── Done: dbg0 = 1, ECALL ─────────────────────────────────────────────
# (host expects dbg0 == 1 for PASS; intermediate phase markers above
# overwrite dbg0 during execution — final value 1 means kernel ran to
# completion. dbg1 holds the last round counter for further diagnostics.)

    LI    x30, 1
    csrrw x0, 0x800, x30
    ECALL

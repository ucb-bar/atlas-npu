# sas_vmem_3way_arb.S — single-bank 3-way VMEM priority ladder
#
# T1 of the Saturn↔Atlas memory stress suite. Three VMEM client
# classes pin VMEM bank 3 simultaneously: Atlas vector LSU
# (VLOAD/VSTORE, the highest priority class this test exercises),
# DMA (mid), and Saturn TileLink (lowest). Scalar LSU traffic is
# NOT exercised here — adding scalar SW/LW to bank 3 would risk
# tripping the per-bank same-cycle scalar/vector conflict assertions.
#
# Bank 3 is split into three DISJOINT subregions so each writer's golden
# is deterministic. Word indices into VMEM (bank index = word_idx >> 15;
# bank 3 starts at word 98 304):
#
#   LSU sub-region:    word 98304..98559 (256 words = 1024 B)
#   DMA sub-region:    word 98560..98815 (256 words = 1024 B)
#   Saturn sub-region: word 98816..99071 (256 words = 1024 B)
#                      (host accesses Saturn region via SBUS at
#                       ATLAS_VMEM_BASE + 0x60800)
#
# DRAM (DMA base 0x00000001):
#   IN  tile at 0x0000 (1024 B, fixed pattern from generator)
#   OUT tile at 0x1000 (1024 B, expected = IN after echo)
#
# Each loop iteration:
#   1. Issue DMA.LOAD on ch0 (DRAM IN → DMA sub-region).
#   2. While DMA is in flight: VLOAD m0 from LSU sub-region, VSTORE m0
#      back. The VPU register m0 is the transit; net effect on the LSU
#      sub-region is identity, but the LSU port is hot.
#   3. DMA.WAIT 0.
#   4. Issue DMA.STORE on ch1 (DMA sub-region → DRAM OUT).
#   5. More VLOAD/VSTORE on LSU sub-region (m1) while DMA store is in
#      flight.
#   6. DMA.WAIT 1.
#   7. Decrement loop counter, branch.
#
# After N iterations, dbg0 = 1, ECALL.
#
# The Saturn sub-region is untouched by Atlas; the host driver runs an
# RVV stride-1 sweep over it during the same window.

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024              # transfer size

    # Inner-loop count × outer rounds: 128 × 4 = 512 LSU+DMA cycles for
    # a sustained overlap window. Each iteration runs ~1K Atlas cycles;
    # 512 iterations ≈ 200K cycles. Saturn's TARGET_ITERS is 4096 so
    # the sweep does not exhaust before Atlas halts.
    LI    x27, 4                 # outer round cap
    LI    x26, 0                 # outer round counter

outer:
    LI    x13, 128               # inner iter cap
    LI    x11, 0                 # inner iter counter

loop:
    # ── DMA.LOAD: DRAM[0x0000] → VMEM bank 3 word 256 ──
    LI    x7,  98560
    LI    x9,  0x0000
    DMA.LOAD x7, x9, x12, 0

    # ── LSU work overlap: VLOAD/VSTORE on bank 3 word 0 ──
    LI    x6,  98304
    VLOAD 0, x6, 0
    DELAY 40
    VSTORE 0, x6, 0
    DELAY 40

    DMA.WAIT 0

    # ── DMA.STORE: VMEM bank 3 word 256 → DRAM[0x1000] ──
    LI    x7,  98560
    LI    x10, 0x1000
    DMA.STORE x10, x7, x12, 1

    # ── More LSU work overlap (different VPU reg to dodge any RAW) ──
    LI    x6,  98304
    VLOAD 1, x6, 0
    DELAY 40
    VSTORE 1, x6, 0
    DELAY 40

    DMA.WAIT 1

    ADDI  x11, x11, 1
    BLT   x11, x13, loop
    NOP

    ADDI  x26, x26, 1
    BLT   x26, x27, outer
    NOP

# ── Done: dbg0 = 1, ECALL ─────────────────────────────────────────────
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

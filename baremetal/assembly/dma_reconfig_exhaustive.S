# dma_reconfig_exhaustive.S — exhaustive DMA.CONFIG reissue, single upper base (FPGA)
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_reconfig_exhaustive.py
#
# FPGA variant of the exhaustive DMA.CONFIG sweep.  The original test
# walked DMA.CONFIG.upper through all 8 base values (0x0..0x7) — those
# upper bases address regions above the FPGA's 0x9000_0000 DRAM window
# and so cannot be exercised on real silicon.  This version keeps the
# upper base at 0x00000000 and instead sweeps the lower-32 DRAM offset
# across 8 distinct sub-windows within 0x9000_0000.
#
# Transfer layout (each transfer is 128 bytes = 4 beats of 32 bytes):
#   For window W (0..7):
#     LOAD:  DRAM[0x9000_0000 + W * 0x10000 + 0x0000] → VMEM[W * 256]
#     STORE: VMEM[W * 256]                            → DRAM[0x9000_0000 + W * 0x10000 + 0x4000]
#
# Channels 0..7 are reused for loads and stores; each pair runs through
# DMA.WAIT before re-using a channel, so the channel-busy logic is still
# exercised across all 8 channels.
#
# Register usage:
#   x1  — constant: transfer size (128 bytes)
#   x2  — constant: VMEM stride between groups (256 = 8 * 32 bytes)
#   x4  — scratch: DRAM byte offset for loads
#   x5  — scratch: DRAM byte offset for stores
#   x6  — scratch: VMEM byte offset for current group
#   x7  — scratch: DMA.CONFIG operand (upper-32 base, kept at 0)
#   x30 — sentinel value (1) written to CSR_DBG0 at end
# ─────────────────────────────────────────────────────────────────────────────

    LI    x1, 128
    LI    x7, 0x00000000     # DMA.CONFIG.upper stays at 0 for FPGA

# ─── WINDOW 0 ────────────────────────────────────────────────────────────────
# vmem_start = 0 * 256 = 0
    DMA.CONFIG x7, 0
    LI    x4, 0x90000000
    LI    x5, 0x90004000
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x4, x1, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    DMA.STORE x5, x6, x1, 1
    DMA.WAIT  1

# ─── WINDOW 1 ────────────────────────────────────────────────────────────────
# vmem_start = 1 * 256
    DMA.CONFIG x7, 0
    LI    x4, 0x90010000
    LI    x5, 0x90014000
    LI    x6, 0x0100
    DMA.LOAD  x6, x4, x1, 2
    DMA.WAIT  2
    LI    x6, 0x0100
    DMA.STORE x5, x6, x1, 3
    DMA.WAIT  3

# ─── WINDOW 2 ────────────────────────────────────────────────────────────────
    DMA.CONFIG x7, 0
    LI    x4, 0x90020000
    LI    x5, 0x90024000
    LI    x6, 0x0200
    DMA.LOAD  x6, x4, x1, 4
    DMA.WAIT  4
    LI    x6, 0x0200
    DMA.STORE x5, x6, x1, 5
    DMA.WAIT  5

# ─── WINDOW 3 ────────────────────────────────────────────────────────────────
    DMA.CONFIG x7, 0
    LI    x4, 0x90030000
    LI    x5, 0x90034000
    LI    x6, 0x0300
    DMA.LOAD  x6, x4, x1, 6
    DMA.WAIT  6
    LI    x6, 0x0300
    DMA.STORE x5, x6, x1, 7
    DMA.WAIT  7

# ─── WINDOW 4 ────────────────────────────────────────────────────────────────
    DMA.CONFIG x7, 0
    LI    x4, 0x90040000
    LI    x5, 0x90044000
    LI    x6, 0x0400
    DMA.LOAD  x6, x4, x1, 0
    DMA.WAIT  0
    LI    x6, 0x0400
    DMA.STORE x5, x6, x1, 1
    DMA.WAIT  1

# ─── WINDOW 5 ────────────────────────────────────────────────────────────────
    DMA.CONFIG x7, 0
    LI    x4, 0x90050000
    LI    x5, 0x90054000
    LI    x6, 0x0500
    DMA.LOAD  x6, x4, x1, 2
    DMA.WAIT  2
    LI    x6, 0x0500
    DMA.STORE x5, x6, x1, 3
    DMA.WAIT  3

# ─── WINDOW 6 ────────────────────────────────────────────────────────────────
    DMA.CONFIG x7, 0
    LI    x4, 0x90060000
    LI    x5, 0x90064000
    LI    x6, 0x0600
    DMA.LOAD  x6, x4, x1, 4
    DMA.WAIT  4
    LI    x6, 0x0600
    DMA.STORE x5, x6, x1, 5
    DMA.WAIT  5

# ─── WINDOW 7 ────────────────────────────────────────────────────────────────
    DMA.CONFIG x7, 0
    LI    x4, 0x90070000
    LI    x5, 0x90074000
    LI    x6, 0x0700
    DMA.LOAD  x6, x4, x1, 6
    DMA.WAIT  6
    LI    x6, 0x0700
    DMA.STORE x5, x6, x1, 7
    DMA.WAIT  7

# ─── Signal completion ────────────────────────────────────────────────────────
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

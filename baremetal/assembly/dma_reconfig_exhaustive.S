# dma_reconfig_exhaustive.S — exhaustive DMA.CONFIG base register coverage
# @TIMEOUT 500000
# @PYTHON_GEN gen_dma_reconfig_exhaustive.py
#
# Tests every possible dmaBaseReg value (0x0–0x7) with both a LOAD and a STORE
# for each base. The upper 32 bits of every DMA address are asserted against
# the expected base value via the golden data checks.
#
# Transfer layout (all transfers are 128 bytes = 4 beats of 32 bytes):
#   For base B (0..7):
#     LOAD:  DRAM[B : 0x8000_0000 .. 0x8000_007F] → VMEM[B * 256 .. B * 256 + 127]
#     STORE: VMEM[B * 256 .. B * 256 + 127]        → DRAM[B : 0x8001_0000 .. 0x8001_007F]
#
# The LOAD fills VMEM from a unique per-base preloaded payload; the STORE
# writes that payload back to a different DRAM offset so the golden checker
# can confirm the upper address bits were encoded correctly for both directions.
#
# Channel assignment: each (base, direction) pair uses a unique channel
# (channels 0..7) to allow independent DMA.WAIT without cross-channel
# interference, and to stress the channel-busy logic across all 8 channels.
#
# Register usage:
#   x1  — constant: transfer size (128 bytes)
#   x2  — constant: VMEM stride between base-groups (256 = 8 * 32 bytes)
#   x3  — loop induction / scratch
#   x4  — scratch: DRAM offset for loads (always 0x8000_0000)
#   x5  — scratch: DRAM offset for stores (always 0x8001_0000)
#   x6  — scratch: vmem base address for current group
#   x7  — scratch: base index counter (0..7)
#   x8  — scratch: channel index for loads  (= base * 2)
#   x9  — scratch: channel index for stores (= base * 2 + 1)
#   x30 — sentinel value (1) written to CSR_DBG0 at end
#
# NOTE: Software-scheduled; programmer guarantees no bank conflicts and
# DMA.WAIT before reusing a channel.  VMEM bank layout must accommodate
# 8 * 256 = 2048 bytes (64 lines at 32 bytes each) starting at VMEM line 0.
# ─────────────────────────────────────────────────────────────────────────────

    # x1 = 128 (transfer size in bytes for all operations)
    LI    x1, 128

    # x4 = 0x80000000 (DRAM byte offset for loads — lower 32b only)
    LI    x4, 0x80000000

    # x5 = 0x80010000 (DRAM byte offset for stores — lower 32b only)
    LI    x5, 0x80010000

# ─── BASE 0x0 ────────────────────────────────────────────────────────────────
# Set base = 0x00000000, channel 0 = load, channel 1 = store.
# vmem_start = 0 * 256 = 0x0000 (byte offset) → line addr 0
    LI    x6, 0x00000000
    DMA.CONFIG x6, 0

    # Load: DRAM[0x0:0x80000000..0x8000007F] → VMEM[0]
    ADDI  x6, x0, 0          # vmem byte offset 0
    DMA.LOAD  x6, x4, x1, 0
    DMA.WAIT  0

    # Store: VMEM[0] → DRAM[0x0:0x80010000..0x8001007F]
    ADDI  x6, x0, 0          # vmem byte offset 0
    DMA.STORE x5, x6, x1, 1
    DMA.WAIT  1

# ─── BASE 0x1 ────────────────────────────────────────────────────────────────
# vmem_start = 1 * 256 = 0x0100 (byte offset) → line addr 8
    LI    x6, 0x00000001
    DMA.CONFIG x6, 0

    LI    x6, 0x0100          # vmem byte offset 256
    DMA.LOAD  x6, x4, x1, 2
    DMA.WAIT  2

    LI    x6, 0x0100
    DMA.STORE x5, x6, x1, 3
    DMA.WAIT  3

# ─── BASE 0x2 ────────────────────────────────────────────────────────────────
# vmem_start = 2 * 256 = 0x0200 → line addr 16
    LI    x6, 0x00000002
    DMA.CONFIG x6, 0

    LI    x6, 0x0200
    DMA.LOAD  x6, x4, x1, 4
    DMA.WAIT  4

    LI    x6, 0x0200
    DMA.STORE x5, x6, x1, 5
    DMA.WAIT  5

# ─── BASE 0x3 ────────────────────────────────────────────────────────────────
# vmem_start = 3 * 256 = 0x0300 → line addr 24
    LI    x6, 0x00000003
    DMA.CONFIG x6, 0

    LI    x6, 0x0300
    DMA.LOAD  x6, x4, x1, 6
    DMA.WAIT  6

    LI    x6, 0x0300
    DMA.STORE x5, x6, x1, 7
    DMA.WAIT  7

# ─── At this point channels 0-7 are exhausted.
# The engine supports 8 channels (dmaP.numChannels).  Re-use channels 0-7
# for bases 4-7 after confirming all previous waits have completed.
# All DMA.WAITs above guarantee the hardware is idle on those channels.

# ─── BASE 0x4 ────────────────────────────────────────────────────────────────
# vmem_start = 4 * 256 = 0x0400 → line addr 32
    LI    x6, 0x00000004
    DMA.CONFIG x6, 0

    LI    x6, 0x0400
    DMA.LOAD  x6, x4, x1, 0
    DMA.WAIT  0

    LI    x6, 0x0400
    DMA.STORE x5, x6, x1, 1
    DMA.WAIT  1

# ─── BASE 0x5 ────────────────────────────────────────────────────────────────
# vmem_start = 5 * 256 = 0x0500 → line addr 40
    LI    x6, 0x00000005
    DMA.CONFIG x6, 0

    LI    x6, 0x0500
    DMA.LOAD  x6, x4, x1, 2
    DMA.WAIT  2

    LI    x6, 0x0500
    DMA.STORE x5, x6, x1, 3
    DMA.WAIT  3

# ─── BASE 0x6 ────────────────────────────────────────────────────────────────
# vmem_start = 6 * 256 = 0x0600 → line addr 48
    LI    x6, 0x00000006
    DMA.CONFIG x6, 0

    LI    x6, 0x0600
    DMA.LOAD  x6, x4, x1, 4
    DMA.WAIT  4

    LI    x6, 0x0600
    DMA.STORE x5, x6, x1, 5
    DMA.WAIT  5

# ─── BASE 0x7 ────────────────────────────────────────────────────────────────
# vmem_start = 7 * 256 = 0x0700 → line addr 56
    LI    x6, 0x00000007
    DMA.CONFIG x6, 0

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

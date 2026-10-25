# sas_dmem_sbus_dma.S — sustained 8-channel DMA pump on disjoint DRAM
#
# T4 of the Saturn↔Atlas memory stress suite. Atlas runs an 8-channel
# DMA echo pump for N_ROUNDS rounds. The host driver runs an RVV sweep
# over its own large DRAM-backed scratch buffer concurrently, putting
# both TL masters (Saturn dcache → SBUS, Atlas DMA → SBUS) on the same
# SBUS xbar at the same time. Saturn's region and Atlas's region are
# disjoint at the DRAM level so neither side touches the other's data;
# the contention is at the SBUS xbar, not at the destination memory.
#
# Atlas uses the same 8-channel layout as T2: the DMA word bases now decode as
# banks 0, 0, 1, 2, 2, 3, 4, and 0 under the current 256 KiB banking.
# Pure echo, no compute.
#
# DRAM layout (DMA base 0x00000001):
#   IN  tiles  at 0x0000, 0x0400, ..., 0x1C00 (8 × 1024 B)
#   OUT tiles  at 0x2000, 0x2400, ..., 0x3C00 (8 × 1024 B, expected = IN)
#
# N_ROUNDS = 16: long enough to fully overlap a Saturn 64-KiB sweep at
# typical Saturn-dcache miss rates. Bump up if the host-side cycle
# comparison shows insufficient overlap.
#
# End: dbg0 = 1, halt_reason = ECALL.

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024            # transfer size

    LI    x29, 16              # N_ROUNDS
    LI    x28, 0               # round counter

round_loop:
# ── 8-channel DMA.LOAD burst ──────────────────────────────────────────

    LI    x6,  0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0

    LI    x6,  32768
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1

    LI    x6,  65536
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2

    LI    x6,  131072
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch3

    LI    x6,  163840
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch4

    LI    x6,  196608
    LI    x10, 0x90001400
    dma.load x6, x10, x12, ch5

    LI    x6,  262144
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch6

    LI    x6,  256
    LI    x10, 0x90001C00
    dma.load x6, x10, x12, ch7

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

# ── 8-channel DMA.STORE burst ─────────────────────────────────────────

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

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    dma.wait ch7

    ADDI  x28, x28, 1
    BLT   x28, x29, round_loop

# ── Done: dbg0 = 1, ECALL ─────────────────────────────────────────────
    LI    x30, 1
    csrrw x0, 0x800, x30
    ECALL

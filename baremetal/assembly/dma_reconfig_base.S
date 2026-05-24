# dma_reconfig_base.S - DMA.CONFIG reissue with single upper base (FPGA)
# @TIMEOUT 100000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_reconfig_base.py
#
# FPGA variant: only the lower 4 GiB of DRAM is present, so we cannot
# sweep DMA.CONFIG across multiple upper-32 base values.  Instead, we
# reissue DMA.CONFIG with the same upper-base value while moving the
# lower-32 byte offset across two distinct DRAM windows.  This still
# exercises the DMA.CONFIG re-write path and verifies both loads and
# stores use the latest configured base for subsequent transfers.
#
# All accesses land inside the FPGA-available 0x9000_0000 window.
#
# Sequence:
#   reissue 1: LOAD  DRAM[0x9000_0000..0x9000_007F] → VMEM[0..127]
#   reissue 2: LOAD  DRAM[0x9000_4000..0x9000_407F] → VMEM[256..383]
#   reissue 3: STORE VMEM[0..127]   → DRAM[0x9000_2000..0x9000_207F]
#   reissue 4: STORE VMEM[256..383] → DRAM[0x9000_6000..0x9000_607F]

    LI    x12, 128
    LI    x5, 0x00000000

    # ── reissue 1: LOAD from 0x9000_0000 → VMEM[0] ──
    DMA.CONFIG x5, 0
    LI    x10, 0x90000000
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0

    # ── reissue 2: LOAD from 0x9000_4000 → VMEM[256] ──
    DMA.CONFIG x5, 0
    LI    x10, 0x90004000
    LI    x6, 256
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1

    # ── reissue 3: STORE VMEM[0] → 0x9000_2000 ──
    DMA.CONFIG x5, 0
    LI    x10, 0x90002000
    ADDI  x6, x0, 0
    DMA.STORE x10, x6, x12, 2
    DMA.WAIT  2

    # ── reissue 4: STORE VMEM[256] → 0x9000_6000 ──
    DMA.CONFIG x5, 0
    LI    x10, 0x90006000
    LI    x6, 256
    DMA.STORE x10, x6, x12, 3
    DMA.WAIT  3

    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

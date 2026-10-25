# sac_reduction_sum_to_high.S — smolvla_reduction_sum with JAL-to-upper-IMEM tail.
#
# Identical to smolvla_reduction_sum.S through the final DMA.STORE + DMA.WAIT,
# then replaces the pass: {ADDI x1,x0,1 ; CSRRW x0,0x800,x1 ; ECALL} block
# with a JALR to IMEM byte 0x10000 so the next kernel
# staged into the upper IMEM half runs without halting.
#
# Used by tests/saturn_atlas_tri_phase.c: the pass-path caller loads a stock
# smolvla_elementwise_add at IMEM byte 0x10000 and relies on this kernel
# flowing into it via JAL. dbg0 is written by the downstream kernel, not here.
#
# The fail: ECALL block is dead code in this kernel — nothing in the pre-JAL
# body branches to it, so it is effectively unreachable. It is retained
# (past the JALR) only as a safety net in case someone later adds an
# explicit jump to `fail:` from the reduce body.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
#
# DRAM layout:
#   [0x0000..0x03FF]  Input X bank 0    (1024 B)
#   [0x0400..0x07FF]  Input X bank 1    (1024 B)
#   [0x0800..0x0FFF]  Output            (2048 B, both banks identical)
#
# Mreg allocation:  m0,m1 = X    m4,m5 = row sums (broadcast)

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load Input X → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 256
    LI    x1, 0x90000400
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 40
    vload m1, 8(x6)
    DELAY 40

    # ── (m4,m5) = row-sum broadcast over (m0,m1) ──
    vredsum.row.bf16 m4, m0
    DELAY 150

    LI    x8, 512
    vstore m4, 0(x8)
    DELAY 40
    vstore m5, 8(x8)
    DELAY 40

    LI    x8, 512
    LI    x3, 0x90000800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 768
    LI    x3, 0x90000C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

to_high:
    # Hop to the kernel staged at byte 0x10000 without halting. The Atlas PC
    # is a byte address, and the JALR drops the `fail:` CSRRW after it (no
    # delay slot).
    LI    x1, 0x10000
    JALR  x0, x1, 0

fail:
    CSRRW x0, 0x800, x28
    ECALL

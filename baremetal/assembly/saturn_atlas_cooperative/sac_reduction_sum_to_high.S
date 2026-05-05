# sac_reduction_sum_to_high.S — smolvla_reduction_sum with JAL-to-upper-IMEM tail.
#
# Identical to smolvla_reduction_sum.S through the final DMA.STORE + DMA.WAIT,
# then replaces the pass: {ADDI x1,x0,1 ; CSRRW x0,0xC10,x1 ; ECALL} block
# with a word-indexed JALR to byte 0x10000 (word 0x4000) so the next kernel
# staged into the upper IMEM half runs without halting.
#
# Used by tests/saturn_atlas_tri_phase.c: the pass-path caller loads a stock
# smolvla_elementwise_add at IMEM byte 0x10000 and relies on this kernel
# flowing into it via JAL. dbg0 is written by the downstream kernel, not here.
#
# The fail: ECALL block is dead code in this kernel — nothing in the pre-JAL
# body branches to it, so it is effectively unreachable. It is retained
# (past the JALR + NOP delay slot) only as a safety net in case someone
# later adds an explicit jump to `fail:` from the reduce body.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x00000001
#
# DRAM layout:
#   [0x0000..0x03FF]  Input X bank 0    (1024 B)
#   [0x0400..0x07FF]  Input X bank 1    (1024 B)
#   [0x0800..0x0FFF]  Output            (2048 B, both banks identical)
#
# Mreg allocation:  m0,m1 = X    m4,m5 = row sums (broadcast)

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input X → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ── (m4,m5) = row-sum broadcast over (m0,m1) ──
    VREDSUM.ROW.BF16 4, 0
    DELAY 150

    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40

    LI    x8, 512
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x0C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

to_high:
    # Hop to the kernel staged at byte 0x10000 (word 0x4000) without halting.
    # Atlas PC is a word index; JALR uses rs1+imm as-is (no byte-to-word shift).
    # Atlas has 1 delay slot after JAL/JALR: the instruction
    # immediately following the JALR are fetched and executed before the
    # redirect takes effect. They MUST be harmless NOPs; otherwise the
    # `fail:` CSRRW+ECALL below would fire in the delay slot, halting
    # the core with dbg0=1 / halt_reason=ECALL BEFORE the redirect to
    # kernel_add lands. (See sab_a_to_high.S:26-28 for the same pattern.)
    LI    x1, 0x4000
    JALR  x0, x1, 0
    NOP

fail:
    CSRRW x0, 0xC10, x28
    ECALL

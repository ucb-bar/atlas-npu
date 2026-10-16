# smolvla_elementwise_add_to_high.S — JAL-to-upper-IMEM variant of
# smolvla_elementwise_add.S
#
# Identical body to smolvla_elementwise_add.S through the final DMA.STORE +
# DMA.WAIT, then replaces the pass: {ADDI x1,x0,1 ; CSRRW x0,0xC10,x1 ; ECALL}
# block with a word-indexed JALR to byte 0x10000 (word 0x4000) so the next
# kernel staged into the upper IMEM half runs without halting.
#
# Used by tests/saturn_atlas_quad_phase_large.c. Atlas has 1 delay slot
# after JAL/JALR; the NOP after JALR absorbs it. The fail: ECALL block
# is dead code, retained for safety.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
#
# DRAM layout (same as smolvla_elementwise_add):
#   [0x0000..0x03FF]  Input A bank 0    (1024 B)
#   [0x0400..0x07FF]  Input A bank 1    (1024 B)
#   [0x0800..0x0BFF]  Input B bank 0    (1024 B)
#   [0x0C00..0x0FFF]  Input B bank 1    (1024 B)
#   [0x1000..0x17FF]  Output            (2048 B)
#
# Mreg allocation:  m0,m1 = A    m2,m3 = B    m4,m5 = result

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load Input A → VMEM[0..511] ──
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

    # ── Load Input B → VMEM[512..1023] ──
    LI    x7, 512
    LI    x1, 0x90000800
    dma.load x7, x1, x2, ch0
    dma.wait ch0
    LI    x7, 768
    LI    x1, 0x90000C00
    dma.load x7, x1, x2, ch0
    dma.wait ch0

    LI    x7, 512
    vload m2, 0(x7)
    DELAY 40
    vload m3, 8(x7)
    DELAY 40

    # ── Single pair op: (m4,m5) = (m0,m1) + (m2,m3) ──
    vadd.bf16 m4, m0, m2
    DELAY 150

    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 40
    vstore m5, 8(x8)
    DELAY 40

    LI    x8, 1024
    LI    x3, 0x90001000
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90001400
    dma.store x3, x8, x2, ch1
    dma.wait ch1

to_high:
    # Hop to the kernel staged at byte 0x10000 (word 0x4000) without halting.
    # One NOP delay slot covers Atlas's JAL/JALR latency.
    LI    x1, 0x4000
    JALR  x0, x1, 0
    NOP

fail:
    CSRRW x0, 0xC10, x28
    ECALL

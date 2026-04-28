# smolvla_elementwise_add_to_high.S — JAL-to-upper-IMEM variant of
# smolvla_elementwise_add.S
#
# Identical body to smolvla_elementwise_add.S through the final DMA.STORE +
# DMA.WAIT, then replaces the pass: {ADDI x1,x0,1 ; CSRRW x0,0xC10,x1 ; ECALL}
# block with a word-indexed JALR to byte 0x10000 (word 0x4000) so the next
# kernel staged into the upper IMEM half runs without halting.
#
# Used by tests/saturn_atlas_quad_phase_large.c. Atlas has 2-cycle delay slots
# after JAL/JALR; the two NOPs after JALR absorb them. The fail: ECALL block
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

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input A → VMEM[0..511] ──
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

    # ── Load Input B → VMEM[512..1023] ──
    LI    x7, 512
    LI    x1, 2048
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 768
    LI    x1, 3072
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0

    LI    x7, 512
    VLOAD 2, x7, 0
    DELAY 40
    VLOAD 3, x7, 8
    DELAY 40

    # ── Single pair op: (m4,m5) = (m0,m1) + (m2,m3) ──
    VADD.BF16  4, 0, 2
    DELAY 150

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40

    LI    x8, 1024
    LI    x3, 0x1000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x1400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

to_high:
    # Hop to the kernel staged at byte 0x10000 (word 0x4000) without halting.
    # Two NOP delay slots cover Atlas's 2-cycle JAL/JALR latency.
    LI    x1, 0x4000
    JALR  x0, x1, 0
    NOP
    NOP

fail:
    CSRRW x0, 0xC10, x28
    ECALL

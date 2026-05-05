# smolvla_reduction_sum_at_2000.S — variant of smolvla_reduction_sum.S with
# DRAM offsets shifted by +0x2000.
#
# Used by tests/saturn_atlas_quad_phase_large.c as the upper-half kernel of
# the JAL chain in phase A1: lower-half (smolvla_matmul_mxu1_to_high) writes
# its BF16 result at DRAM[0x0800/0x0C00], so to avoid trampling matmul I/O,
# the reduction reads its own BF16 input from DRAM[0x2000/0x2400] (host-staged
# before the phase) and writes the row-sum broadcast to DRAM[0x2800/0x2C00].
# VMEM offsets are unchanged — VMEM is Atlas-local and disjoint from DRAM.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x00000001
#
# DRAM layout:
#   [0x2000..0x23FF]  Input X bank 0    (1024 B)
#   [0x2400..0x27FF]  Input X bank 1    (1024 B)
#   [0x2800..0x2FFF]  Output            (2048 B, both banks identical)
#
# Mreg allocation:  m0,m1 = X    m4,m5 = row sums (broadcast)

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input X → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    LI    x1, 0x2000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 0x2400
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
    LI    x3, 0x2800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x2C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

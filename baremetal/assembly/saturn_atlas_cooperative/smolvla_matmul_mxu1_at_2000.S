# smolvla_matmul_mxu1_at_2000.S — variant of smolvla_matmul_mxu1.S with DRAM
# offsets shifted by +0x2000.
#
# Used by tests/saturn_atlas_quad_phase_large.c as the upper-half kernel of
# the JAL chain in phase A3: lower-half (smolvla_elementwise_add_to_high)
# writes its BF16 result at DRAM[0x1000/0x1400], so to avoid trampling add I/O,
# the matmul reads its own FP8 inputs from DRAM[0x2000/0x2400] (host-staged
# before the phase) and writes BF16 output at DRAM[0x2800/0x2C00].
# VMEM offsets are unchanged.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
#
# DRAM layout:
#   [0x2000..0x23FF]  A FP8 (32×32)             (1024 B)
#   [0x2400..0x27FF]  B FP8 (32×32)             (1024 B)
#   [0x2800..0x2BFF]  Output cols  0-15  BF16    (1024 B)
#   [0x2C00..0x2FFF]  Output cols 16-31  BF16    (1024 B)
#
# Mreg allocation:  m0 = A   m4 = B   m5 = B^T (XLU)   m2,m3 = C

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load B DRAM[0x2400] → VMEM[0] → TRF m4 ──
    ADDI  x6, x0, 0
    LI    x1, 0x2400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40

    # ── Load A DRAM[0x2000] → VMEM[256] → TRF m0 ──
    LI    x7, 256
    LI    x1, 0x2000
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    DELAY 40

    # ── Transpose B so MXU computes A @ B, not A @ B^T ──
    VTRPOSE.XLU 5, 4
    DELAY 80

    VMATPUSH.W.MXU1  0, 5
    DELAY 40
    VMATMUL.MXU1  0, 0, 0
    DELAY 120
    VMATPOP.BF16.MXU1  2, 0
    DELAY 40

    # ── Store m2 (cols 0-15) → VMEM[512] → DRAM[0x2800] ──
    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40

    # ── Store m3 (cols 16-31) → VMEM[768] → DRAM[0x2C00] ──
    LI    x9, 768
    VSTORE 3, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x3, 0x2800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    LI    x9, 768
    LI    x3, 0x2C00
    DMA.STORE x3, x9, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

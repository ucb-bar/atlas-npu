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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load B DRAM[0x2400] → VMEM[0] → TRF m4 ──
    ADDI  x6, x0, 0
    LI    x1, 0x90002400
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m4, 0(x6)
    DELAY 40

    # ── Load A DRAM[0x2000] → VMEM[256] → TRF m0 ──
    LI    x7, 256
    LI    x1, 0x90002000
    dma.load x7, x1, x2, ch0
    dma.wait ch0
    LI    x7, 256
    vload m0, 0(x7)
    DELAY 40

    # ── Transpose B so MXU computes A @ B, not A @ B^T ──
    vtrpose.xlu m5, m4
    DELAY 80

    vmatpush.weight.mxu1 w0, m5
    DELAY 40
    vmatmul.mxu1 acc0, m0, w0
    DELAY 120
    vmatpop.bf16.acc.mxu1 m2, acc0
    DELAY 40

    # ── Store m2 (cols 0-15) → VMEM[512] → DRAM[0x2800] ──
    LI    x8, 512
    vstore m2, 0(x8)
    DELAY 40

    # ── Store m3 (cols 16-31) → VMEM[768] → DRAM[0x2C00] ──
    LI    x9, 768
    vstore m3, 0(x9)
    DELAY 40

    LI    x8, 512
    LI    x3, 0x90002800
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    LI    x9, 768
    LI    x3, 0x90002C00
    dma.store x3, x9, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

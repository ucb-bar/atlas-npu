# smolvla_matmul_mxu1.S (MXU1 / IPT) — port of npu-model smolvla_matmul
# Tests: C = A @ B on seeded random FP8 32x32 tiles
#   (single-tile FP8 matmul → BF16 output, split across m2/m3)
# @TIMEOUT 30000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_matmul_mxu1.py
#
# Atlas MXU loads weights in transposed orientation, so assembly
# VTRPOSE.XLUs B (m4 → m5) before VMATPUSH.W.MXU1. VMATPOP.BF16
# splits BF16 lanes 0–15 into m2 and lanes 16–31 into m3.
#
# DRAM layout:
#   [0x0000..0x03FF]  A FP8 (32×32)             (1024 B, 32 beats)
#   [0x0400..0x07FF]  B FP8 (32×32)             (1024 B, 32 beats)
#   [0x0800..0x0BFF]  Output cols  0-15  BF16    (1024 B, 32 beats)
#   [0x0C00..0x0FFF]  Output cols 16-31  BF16    (1024 B, 32 beats)
#
# Mreg allocation:  m0 = A   m4 = B   m5 = B^T (XLU)   m2,m3 = C

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load B DRAM[0x400] → VMEM[0] → TRF m4 ──
    ADDI  x6, x0, 0
    LI    x1, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40

    # ── Load A DRAM[0x000] → VMEM[256] → TRF m0 ──
    LI    x7, 256
    ADDI  x1, x0, 0
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

    # ── Store m2 (cols 0-15) → VMEM[512] → DRAM[0x800] ──
    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40

    # ── Store m3 (cols 16-31) → VMEM[768] → DRAM[0xC00] ──
    LI    x9, 768
    VSTORE 3, x9, 0
    DELAY 40

    LI    x8, 512
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    LI    x9, 768
    LI    x3, 0x0C00
    DMA.STORE x3, x9, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

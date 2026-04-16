# smolvla_softmax.S — port of npu-model smolvla_softmax
# Tests: row-wise stable softmax on seeded random BF16 32x32 tile
#   y = exp(X - rowmax(X)) / rowsum(exp(...))
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_softmax.py
#
# Atlas's VREDMAX.ROW / VREDSUM.ROW reduce the full 32-lane (m, m+1)
# pair in one op, so npu-model's per-bank reduce + combine collapses
# into one reduction per stage. Six vector ops total.
#
# DRAM layout:
#   [0x0000..0x03FF]  Input X bank 0   (1024 B)
#   [0x0400..0x07FF]  Input X bank 1   (1024 B)
#   [0x0800..0x0FFF]  Output Y         (2048 B)
#
# Mreg allocation (7 pair-pairs):
#   m0,m1   = X                       m2,m3   = rowmax(X)
#   m4,m5   = X - rowmax              m6,m7   = exp(X - rowmax)
#   m8,m9   = rowsum(exp)             m10,m11 = 1/rowsum
#   m12,m13 = exp * 1/rowsum = Y

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
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

    # ── (m2, m3) = rowmax(X) broadcast ──
    VREDMAX.ROW.BF16 2, 0
    DELAY 150

    # ── (m4, m5) = X - rowmax ──
    VSUB.BF16 4, 0, 2
    DELAY 150

    # ── (m6, m7) = exp(X - rowmax) ──
    VEXP 6, 4
    DELAY 150

    # ── (m8, m9) = rowsum(exp) broadcast ──
    VREDSUM.ROW.BF16 8, 6
    DELAY 150

    # ── (m10, m11) = 1 / rowsum ──
    VRECIP.BF16 10, 8
    DELAY 150

    # ── (m12, m13) = exp * inv_rowsum = Y ──
    VMUL.BF16 12, 6, 10
    DELAY 150

    LI    x8, 512
    VSTORE 12, x8, 0
    DELAY 40
    VSTORE 13, x8, 8
    DELAY 40

    LI    x8, 512
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x0C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

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
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ── (m2, m3) = rowmax(X) broadcast ──
    vredmax.row.bf16 m2, m0
    DELAY 32

    # ── (m4, m5) = X - rowmax ──
    vsub.bf16 m4, m0, m2
    DELAY 64

    # ── (m6, m7) = exp(X - rowmax) ──
    vexp.bf16 m6, m4
    DELAY 64

    # ── (m8, m9) = rowsum(exp) broadcast ──
    vredsum.row.bf16 m8, m6
    DELAY 37

    # ── (m10, m11) = 1 / rowsum ──
    vrecip.bf16 m10, m8
    DELAY 64

    # ── (m12, m13) = exp * inv_rowsum = Y ──
    vmul.bf16 m12, m6, m10
    DELAY 64

    LI    x8, 512
    vstore m12, 0(x8)
    DELAY 33
    vstore m13, 8(x8)
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90000800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 768
    LI    x3, 0x90000C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

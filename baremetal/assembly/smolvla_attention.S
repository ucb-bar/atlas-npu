# smolvla_attention.S — dual-MXU scaled dot-product attention (MXU0 + MXU1)
# Tests: single-tile scaled dot-product attention pipeline
#   scores = Q @ K             (MXU0, FP8 → BF16)
#   scaled = scores * scale    (VPU BF16 pair op)
#   probs  = softmax(scaled)   (rowmax → sub → exp → rowsum → recip → mul)
#   packed = VFP8PACK(probs)   (atlas 2→1 phased pack, permuted layout)
#   out    = packed @ V        (MXU1, FP8 → BF16)
# @TIMEOUT 80000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_attention.py
#
# Atlas's MXU computes A @ W^T, so K and V are both XLU-transposed
# before VMATPUSH so the matmuls compute Q @ K and packed @ V rather
# than Q @ K^T / packed @ V^T.
#
# Atlas's VFP8PACK is a 2→1 phased lane box: it streams all 32 rows of
# the low bank then all 32 rows of the high bank, pairing (2i, 2i+1) of
# each bank into packed row i / i+16. The output mreg is therefore a
# permuted layout — packed[i, :] is NOT probs[i, :] — so the second
# matmul does not compute probs@V in the PyTorch attention sense.
# Golden uses atlas's own VectorEngineModel + SARTLLinearFunction which
# have the same permuted layout, so RTL and model agree bit-exactly.
#
# DRAM layout:
#   [0x0000..0x03FF]  Q FP8  (32x32,  1024 B, 32 beats)
#   [0x0400..0x07FF]  K FP8  (32x32,  1024 B, 32 beats)
#   [0x0800..0x0BFF]  V FP8  (32x32,  1024 B, 32 beats)
#   [0x0C00..0x0FFF]  scale BF16 bank 0   (1024 B, 32 beats)
#   [0x1000..0x13FF]  scale BF16 bank 1   (1024 B, 32 beats)
#   [0x1400..0x17FF]  out   BF16 bank 0   (1024 B, 32 beats)
#   [0x1800..0x1BFF]  out   BF16 bank 1   (1024 B, 32 beats)
#
# Mreg allocation:
#   m0       = Q (FP8)              m2       = K (FP8)
#   m3       = K^T (XLU dst)        m4       = V (FP8)
#   m5       = V^T (XLU dst)
#   m6,m7    = scale (BF16 pair)
#   m10,m11  = scores (VMATPOP dst)
#   m12,m13  = scaled               m14,m15  = rowmax
#   m16,m17  = scaled-rowmax        m18,m19  = exp(...)
#   m20,m21  = rowsum               m22,m23  = 1/rowsum
#   m24,m25  = probs (BF16)         m26      = probs (FP8, permuted)
#   m28,m29  = out (VMATPOP dst)

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Q → VMEM[0] → m0 ──
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    # ── Load K → VMEM[256] → m2 ──
    LI    x6, 256
    LI    x1, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    VLOAD 2, x6, 0
    DELAY 40

    # ── Load V → VMEM[512] → m4 ──
    LI    x6, 512
    LI    x1, 2048
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 512
    VLOAD 4, x6, 0
    DELAY 40

    # ── Load scale bank 0 → VMEM[768] → m6 ──
    LI    x6, 768
    LI    x1, 3072
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    # ── Load scale bank 1 → VMEM[1024] → m7 ──
    LI    x6, 1024
    LI    x1, 4096
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    LI    x6, 768
    VLOAD 6, x6, 0
    DELAY 40
    VLOAD 7, x6, 8
    DELAY 40

    # ── Matmul 1: scores = Q @ K (push K^T) ──
    VTRPOSE.XLU 3, 2
    DELAY 80
    VMATPUSH.W.MXU0 0, 3
    DELAY 40
    VMATMUL.MXU0 0, 0, 0
    DELAY 200
    VMATPOP.BF16.MXU0 10, 0
    DELAY 40

    # ── Scale: (m12, m13) = scores * scale ──
    VMUL.BF16 12, 10, 6
    DELAY 150

    # ── Stable softmax ──
    VREDMAX.ROW.BF16 14, 12
    DELAY 150
    VSUB.BF16 16, 12, 14
    DELAY 150
    VEXP 18, 16
    DELAY 150
    VREDSUM.ROW.BF16 20, 18
    DELAY 150
    VRECIP.BF16 22, 20
    DELAY 150
    VMUL.BF16 24, 18, 22
    DELAY 150

    # ── Pack probs BF16 → FP8 (unit scale, permuted layout) ──
    SELI 0, 0x7F
    VFP8PACK 26, 24, 0
    DELAY 150

    # ── Matmul 2: out = packed_probs @ V (push V^T) ──
    VTRPOSE.XLU 5, 4
    DELAY 80
    VMATPUSH.W.MXU1 0, 5
    DELAY 40
    VMATMUL.MXU1 0, 26, 0
    DELAY 200
    VMATPOP.BF16.MXU1 28, 0
    DELAY 40

    # ── Store m28 (cols 0-15) → VMEM[1280] → DRAM[0x1400] ──
    LI    x8, 1280
    VSTORE 28, x8, 0
    DELAY 40

    # ── Store m29 (cols 16-31) → VMEM[1536] → DRAM[0x1800] ──
    VSTORE 29, x8, 8
    DELAY 40

    LI    x8, 1280
    LI    x3, 0x1400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    LI    x8, 1536
    LI    x3, 0x1800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

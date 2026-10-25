# smolvla_attention.S — scaled dot-product attention on MXU1
# Tests: single-tile scaled dot-product attention pipeline
#   scores = Q @ K             (MXU1, FP8 → BF16)
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
# Golden uses atlas's own VectorEngineModel + IPTLinearRTLFunction which
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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load Q → VMEM[0] → m0 ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

    # ── Load K → VMEM[256] → m2 ──
    LI    x6, 256
    LI    x1, 0x90000400
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 256
    vload m2, 0(x6)
    DELAY 33

    # ── Load V → VMEM[512] → m4 ──
    LI    x6, 512
    LI    x1, 0x90000800
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 512
    vload m4, 0(x6)
    DELAY 33

    # ── Load scale bank 0 → VMEM[768] → m6 ──
    LI    x6, 768
    LI    x1, 0x90000C00
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    # ── Load scale bank 1 → VMEM[1024] → m7 ──
    LI    x6, 1024
    LI    x1, 0x90001000
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    LI    x6, 768
    vload m6, 0(x6)
    DELAY 33
    vload m7, 8(x6)
    DELAY 33

    # ── Matmul 1: scores = Q @ K (push K^T) ──
    vtrpose.xlu m3, m2
    DELAY 64
    vmatpush.weight.mxu1 w0, m3
    DELAY 31
    vmatmul.mxu1 acc0, m0, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m10, acc0
    DELAY 31

    # ── Scale: (m12, m13) = scores * scale ──
    vmul.bf16 m12, m10, m6
    DELAY 64

    # ── Stable softmax ──
    vredmax.row.bf16 m14, m12
    DELAY 32
    vsub.bf16 m16, m12, m14
    DELAY 64
    vexp.bf16 m18, m16
    DELAY 64
    vredsum.row.bf16 m20, m18
    DELAY 37
    vrecip.bf16 m22, m20
    DELAY 64
    vmul.bf16 m24, m18, m22
    DELAY 64

    # ── Pack probs BF16 → FP8 (unit scale, permuted layout) ──
    seli e0, 0x7F
    vpack.bf16.fp8 m26, m24, e0
    DELAY 64

    # ── Matmul 2: out = packed_probs @ V (push V^T) ──
    vtrpose.xlu m5, m4
    DELAY 64
    vmatpush.weight.mxu1 w0, m5
    DELAY 31
    vmatmul.mxu1 acc0, m26, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m28, acc0
    DELAY 31

    # ── Store m28 (cols 0-15) → VMEM[1280] → DRAM[0x1400] ──
    LI    x8, 1280
    vstore m28, 0(x8)
    DELAY 33

    # ── Store m29 (cols 16-31) → VMEM[1536] → DRAM[0x1800] ──
    vstore m29, 8(x8)
    DELAY 33

    LI    x8, 1280
    LI    x3, 0x90001400
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    LI    x8, 1536
    LI    x3, 0x90001800
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

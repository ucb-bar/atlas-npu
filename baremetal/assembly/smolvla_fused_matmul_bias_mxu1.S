# smolvla_fused_matmul_bias_mxu1.S (MXU1 / IPT) - port of npu-model smolvla_fused_matmul_bias
# Tests: out = (A_bf16 @ B_fp8) + bias_bf16
#   A is BF16, on-chip quantized to FP8 via VMATPUSH.ACC.BF16/VMATPOP.FP8
#   B is pre-quantized FP8 in DRAM
#   bias is BF16 added with VADD.BF16
#
# Atlas MXU computes A @ W^T natively, so B is VTRPOSE.XLU'd to B^T
# before VMATPUSH.W.MXU1; the matmul then yields A @ B.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_fused_matmul_bias_mxu1.py
#
# DRAM layout (set by gen_smolvla_fused_matmul_bias.py):
#   [0x0000..0x07FF]  A bf16  (32x32, 2 banks = 2048 B)
#   [0x0800..0x0BFF]  B fp8   (32x32, 1024 B; 64-beat slot to 0x0FFF)
#   [0x1000..0x17FF]  bias bf16 (32x32, 2 banks = 2048 B)
#   [0x1800..0x1FFF]  out  bf16 (32x32, 2 banks = 2048 B)
#
# Mreg allocation:
#   m0,m1   = A bf16
#   m4      = B fp8
#   m5      = B^T fp8 (via VTRPOSE.XLU)
#   m6,m7   = bias bf16
#   m8      = A fp8 (via VMATPUSH.ACC.BF16 + SELI + VMATPOP.FP8)
#   m10,m11 = matmul result bf16
#   m12,m13 = result + bias = output
#
# VMEM word allocation:
#   [0..511]     = A bf16 pair (banks 0 & 1)
#   [512..767]   = B fp8 single
#   [768..1279]  = bias bf16 pair
#   [1280..1791] = out bf16 pair

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024              # bank size in bytes

    # ── Load A bf16 → VMEM[0..511] → (m0, m1) ──
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

    # ── Load B fp8 → VMEM[512..767] → m4 ──
    LI    x7, 512
    LI    x1, 0x90000800
    dma.load x7, x1, x2, ch0
    dma.wait ch0

    LI    x7, 512
    vload m4, 0(x7)
    DELAY 33

    # ── Load bias bf16 → VMEM[768..1279] → (m6, m7) ──
    LI    x9, 768
    LI    x1, 0x90001000
    dma.load x9, x1, x2, ch0
    dma.wait ch0
    LI    x9, 1024
    LI    x1, 0x90001400
    dma.load x9, x1, x2, ch0
    dma.wait ch0

    LI    x9, 768
    vload m6, 0(x9)
    DELAY 33
    vload m7, 8(x9)
    DELAY 33

    # ── Transpose B so MXU computes A @ B, not A @ B^T ──
    vtrpose.xlu m5, m4
    DELAY 64

    # ── Quantize A bf16 (m0,m1) → m8 fp8 via acc[1] roundtrip ──
    vmatpush.acc.bf16.mxu1 acc1, m0
    DELAY 31
    seli e0, 0x7F                # E8M0 unit scale (2^0 = 1.0)
    vmatpop.fp8.acc.mxu1 m8, acc1, e0
    DELAY 31

    # ── Matmul: (m10,m11) = m8 @ m5  (= A @ B^T^T = A @ B) ──
    vmatpush.weight.mxu1 w0, m5
    DELAY 31
    vmatmul.mxu1 acc0, m8, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m10, acc0
    DELAY 31

    # ── Bias add: (m12,m13) = (m10,m11) + (m6,m7) ──
    vadd.bf16 m12, m10, m6
    DELAY 64

    # ── Store (m12,m13) → VMEM[1280..1791] → DRAM[0x1800..0x1FFF] ──
    LI    x8, 1280
    vstore m12, 0(x8)
    DELAY 33
    vstore m13, 8(x8)
    DELAY 33

    LI    x8, 1280
    LI    x3, 0x90001800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1536
    LI    x3, 0x90001C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

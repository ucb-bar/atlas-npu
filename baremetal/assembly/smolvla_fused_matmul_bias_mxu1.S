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
# @DRAM_BASE 0x00000001
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

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024              # bank size in bytes

    # ── Load A bf16 → VMEM[0..511] → (m0, m1) ──
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

    # ── Load B fp8 → VMEM[512..767] → m4 ──
    LI    x7, 512
    LI    x1, 2048
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0

    LI    x7, 512
    VLOAD 4, x7, 0
    DELAY 40

    # ── Load bias bf16 → VMEM[768..1279] → (m6, m7) ──
    LI    x9, 768
    LI    x1, 4096
    DMA.LOAD  x9, x1, x2, 0
    DMA.WAIT  0
    LI    x9, 1024
    LI    x1, 5120
    DMA.LOAD  x9, x1, x2, 0
    DMA.WAIT  0

    LI    x9, 768
    VLOAD 6, x9, 0
    DELAY 40
    VLOAD 7, x9, 8
    DELAY 40

    # ── Transpose B so MXU computes A @ B, not A @ B^T ──
    VTRPOSE.XLU 5, 4
    DELAY 80

    # ── Quantize A bf16 (m0,m1) → m8 fp8 via acc[1] roundtrip ──
    VMATPUSH.ACC.BF16.MXU1 1, 0
    DELAY 80
    SELI 0, 0x7F                # E8M0 unit scale (2^0 = 1.0)
    VMATPOP.FP8.MXU1 8, 0, 1
    DELAY 40

    # ── Matmul: (m10,m11) = m8 @ m5  (= A @ B^T^T = A @ B) ──
    VMATPUSH.W.MXU1  0, 5
    DELAY 40
    VMATMUL.MXU1     0, 8, 0
    DELAY 120
    VMATPOP.BF16.MXU1 10, 0
    DELAY 40

    # ── Bias add: (m12,m13) = (m10,m11) + (m6,m7) ──
    VADD.BF16 12, 10, 6
    DELAY 150

    # ── Store (m12,m13) → VMEM[1280..1791] → DRAM[0x1800..0x1FFF] ──
    LI    x8, 1280
    VSTORE 12, x8, 0
    DELAY 40
    VSTORE 13, x8, 8
    DELAY 40

    LI    x8, 1280
    LI    x3, 0x1800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1536
    LI    x3, 0x1C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

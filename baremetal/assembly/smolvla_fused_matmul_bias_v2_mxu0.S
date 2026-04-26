# smolvla_fused_matmul_bias_v2_mxu0.S (MXU0 / SA) - alt port matching the npu-model reference file
# Tests: out = (A_fp8 @ B_fp8) + bias_bf16
#   Both A and B are PRE-QUANTIZED FP8 tiles in DRAM (no on-chip quant).
#   Matches the npu-model smolvla_fused_matmul_bias sketch and the
#   existing smolvla_matmul_mxu0.S activation path.
#   Simpler than V1 but does NOT exercise the BF16->FP8 hardware quant.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_fused_matmul_bias_v2_mxu0.py
#
# DRAM layout (set by gen_smolvla_fused_matmul_bias_v2.py):
#   [0x0000..0x03FF]  A fp8   (32x32, 1024 B)
#   [0x0400..0x07FF]  B fp8   (32x32, 1024 B)
#   [0x0800..0x0FFF]  bias bf16 (32x32, 2 banks = 2048 B)
#   [0x1000..0x17FF]  out  bf16 (32x32, 2 banks = 2048 B)
#
# Mreg allocation:
#   m0      = A fp8
#   m4      = B fp8
#   m5      = B^T fp8 (via VTRPOSE.XLU)
#   m6,m7   = bias bf16
#   m10,m11 = matmul result bf16
#   m12,m13 = result + bias = output
#
# VMEM word allocation:
#   [0..255]     = A fp8 single
#   [256..511]   = B fp8 single
#   [512..1023]  = bias bf16 pair
#   [1024..1535] = out bf16 pair

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024              # bank size in bytes

    # ── Load A fp8 → VMEM[0..255] → m0 ──
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    # ── Load B fp8 → VMEM[256..511] → m4 ──
    LI    x7, 256
    LI    x1, 1024
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0

    LI    x7, 256
    VLOAD 4, x7, 0
    DELAY 40

    # ── Load bias bf16 → VMEM[512..1023] → (m6, m7) ──
    LI    x9, 512
    LI    x1, 2048
    DMA.LOAD  x9, x1, x2, 0
    DMA.WAIT  0
    LI    x9, 768
    LI    x1, 3072
    DMA.LOAD  x9, x1, x2, 0
    DMA.WAIT  0

    LI    x9, 512
    VLOAD 6, x9, 0
    DELAY 40
    VLOAD 7, x9, 8
    DELAY 40

    # ── Transpose B so MXU computes A @ B, not A @ B^T ──
    VTRPOSE.XLU 5, 4
    DELAY 80

    # ── Matmul: (m10,m11) = m0 @ m5  (= A @ B^T^T = A @ B) ──
    VMATPUSH.W.MXU0  0, 5
    DELAY 40
    VMATMUL.MXU0     0, 0, 0
    DELAY 120
    VMATPOP.BF16.MXU0 10, 0
    DELAY 40

    # ── Bias add: (m12,m13) = (m10,m11) + (m6,m7) ──
    VADD.BF16 12, 10, 6
    DELAY 150

    # ── Store (m12,m13) → VMEM[1024..1535] → DRAM[0x1000..0x17FF] ──
    LI    x8, 1024
    VSTORE 12, x8, 0
    DELAY 40
    VSTORE 13, x8, 8
    DELAY 40

    LI    x8, 1024
    LI    x3, 0x1000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x1400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

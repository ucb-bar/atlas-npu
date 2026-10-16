# smolvla_fused_matmul_bias_v2_mxu1.S (MXU1 / IPT) - alt port matching the npu-model reference file
# Tests: out = (A_fp8 @ B_fp8) + bias_bf16
#   Both A and B are PRE-QUANTIZED FP8 tiles in DRAM (no on-chip quant).
#   Matches the npu-model smolvla_fused_matmul_bias sketch and the
#   existing smolvla_matmul_mxu0.S activation path.
#   Simpler than V1 but does NOT exercise the BF16->FP8 hardware quant.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_fused_matmul_bias_v2_mxu1.py
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

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024              # bank size in bytes

    # ── Load A fp8 → VMEM[0..255] → m0 ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

    # ── Load B fp8 → VMEM[256..511] → m4 ──
    LI    x7, 256
    LI    x1, 0x90000400
    dma.load x7, x1, x2, ch0
    dma.wait ch0

    LI    x7, 256
    vload m4, 0(x7)
    DELAY 33

    # ── Load bias bf16 → VMEM[512..1023] → (m6, m7) ──
    LI    x9, 512
    LI    x1, 0x90000800
    dma.load x9, x1, x2, ch0
    dma.wait ch0
    LI    x9, 768
    LI    x1, 0x90000C00
    dma.load x9, x1, x2, ch0
    dma.wait ch0

    LI    x9, 512
    vload m6, 0(x9)
    DELAY 33
    vload m7, 8(x9)
    DELAY 33

    # ── Transpose B so MXU computes A @ B, not A @ B^T ──
    vtrpose.xlu m5, m4
    DELAY 64

    # ── Matmul: (m10,m11) = m0 @ m5  (= A @ B^T^T = A @ B) ──
    vmatpush.weight.mxu1 w0, m5
    DELAY 31
    vmatmul.mxu1 acc0, m0, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m10, acc0
    DELAY 31

    # ── Bias add: (m12,m13) = (m10,m11) + (m6,m7) ──
    vadd.bf16 m12, m10, m6
    DELAY 64

    # ── Store (m12,m13) → VMEM[1024..1535] → DRAM[0x1000..0x17FF] ──
    LI    x8, 1024
    vstore m12, 0(x8)
    DELAY 33
    vstore m13, 8(x8)
    DELAY 33

    LI    x8, 1024
    LI    x3, 0x90001000
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90001400
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

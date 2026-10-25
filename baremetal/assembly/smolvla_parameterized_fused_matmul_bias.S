# smolvla_parameterized_fused_matmul_bias.S — tiled
#   (A_bf16 @ B_fp8) + bias_bf16 on MXU1
# The BF16→FP8 quantize roundtrip for A uses MXU1's acc[1]; the matmul
# uses acc[0] and reads the quantized FP8 mreg.
#
# Port of jrmills20 parameterized_fused_matmul_bias.py + the stock
# smolvla_fused_matmul_bias per-tile body.
#
# A is BF16; on-chip quantized to FP8 via VMATPUSH.ACC.BF16 + SELI + VMATPOP.FP8.
# B is pre-quantized FP8 in DRAM. Bias is BF16 added with VADD.BF16.
# NO VMATPUSH.ACC.FP8 preload — we mirror stock's quantize-roundtrip pattern.
#
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_fused_matmul_bias.py
#
# Shape: M=64, K=32, N=64 → M_tiles=2, N_tiles=2, K_tiles=1 (single K-tile
# per output — no accumulation in this first port).
#
# DRAM layout:
#   [0x0000..0x0FFF]  A BF16 (2 tiles x 2048B)
#   [0x1000..0x1FFF]  B FP8  (2 tiles x 2048B slot; 1024B data + 1024B slot
#                             padding, matches stock 64-beat slot convention)
#   [0x2000..0x3FFF]  bias BF16 (4 tiles x 2048B)
#   [0x4000..0x5FFF]  out  BF16 (4 tiles x 2048B)
#
# VMEM staging (reused every iteration):
#   [0..511]     = A  bf16 pair
#   [512..767]   = B  fp8 single
#   [768..1279]  = bias bf16 pair
#   [1280..1791] = out  bf16 pair
#
# Mreg allocation (per tile, matches stock):
#   m0,m1   = A bf16                m4  = B fp8
#   m5      = B^T fp8 (VTRPOSE)     m6,m7  = bias bf16
#   m8      = A fp8 (via quantize)  m10,m11 = matmul bf16
#   m12,m13 = output = matmul + bias
#
# Register map:
#   x2 = 1024 (tile size)          x5 = 0x00000001 (DRAM base)
#   x10 = output_tile_idx (0..3)   x11 = 4
#   x12 = m_tile                   x13 = n_tile
#   x20 = A[m,0] DRAM offset
#   x22 = B[0,n] DRAM offset       x25 = 0x1000 (B base)
#   x23 = bias[m,n] offset         x26 = 0x2000 (bias base)
#   x24 = out[m,n] offset          x27 = 0x4000 (out base)
#   x1,x6,x7,x8,x9 = VMEM/DMA scratch; x28 = fail code

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x25, 0x1000
    LI    x26, 0x2000
    LI    x27, 0x4000
    ADDI  x11, x0, 4
    LUI   x15, 0x90000          # FPGA DRAM offset 0x9000_0000

    ADDI  x10, x0, 0

output_tile_loop:
    SRLI  x12, x10, 1
    ANDI  x13, x10, 1

    # ── A[m, 0] DRAM offset = m * 2048 ──
    SLLI  x20, x12, 11
    ADD   x20, x20, x15
    # ── B[0, n] DRAM offset = n * 2048 + 0x1000 ──
    SLLI  x22, x13, 11
    ADD   x22, x22, x25
    ADD   x22, x22, x15
    # ── bias[m, n] DRAM offset = (m*2 + n) * 2048 + 0x2000 ──
    SLLI  x23, x12, 1
    ADD   x23, x23, x13
    SLLI  x23, x23, 11
    ADD   x23, x23, x26
    ADD   x23, x23, x15
    # ── out[m, n] DRAM offset = (m*2 + n) * 2048 + 0x4000 ──
    SLLI  x24, x12, 1
    ADD   x24, x24, x13
    SLLI  x24, x24, 11
    ADD   x24, x24, x27
    ADD   x24, x24, x15

    # ── Load A bf16 → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    dma.load x6, x20, x2, ch0
    dma.wait ch0
    LI    x6, 256
    ADDI  x1, x20, 1024
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ── Load B fp8 → VMEM[512] → m4 ──
    LI    x7, 512
    dma.load x7, x22, x2, ch0
    dma.wait ch0
    LI    x7, 512
    vload m4, 0(x7)
    DELAY 33

    # ── Load bias bf16 → VMEM[768..1279] → (m6, m7) ──
    LI    x9, 768
    dma.load x9, x23, x2, ch0
    dma.wait ch0
    LI    x9, 1024
    ADDI  x1, x23, 1024
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
    seli e0, 0x7F
    vmatpop.fp8.acc.mxu1 m8, acc1, e0
    DELAY 31

    # ── MXU1: (m10,m11) = m8 @ m5 (= A @ B) ──
    vmatpush.weight.mxu1 w0, m5
    DELAY 31
    vmatmul.mxu1 acc0, m8, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m10, acc0
    DELAY 31

    # ── Bias add: (m12, m13) = (m10, m11) + (m6, m7) ──
    vadd.bf16 m12, m10, m6
    DELAY 64

    # ── Store (m12, m13) → VMEM[1280..1791] ──
    LI    x8, 1280
    vstore m12, 0(x8)
    DELAY 33
    vstore m13, 8(x8)
    DELAY 33

    # ── DMA out: two halves → C[m,n] ──
    LI    x8, 1280
    dma.store x24, x8, x2, ch1
    dma.wait ch1
    ADDI  x1, x24, 1024
    LI    x8, 1536
    dma.store x1, x8, x2, ch1
    dma.wait ch1

    ADDI  x10, x10, 1
    BNE   x10, x11, output_tile_loop
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

# smolvla_parameterized_fused_matmul_bias.S — dual-MXU tiled
#   (A_bf16 @ B_fp8) + bias_bf16
# Column-split across engines: output tiles with n_tile=0 run on MXU0
# (SA); tiles with n_tile=1 run on MXU1 (IPT). The BF16→FP8 quantize
# roundtrip for A stays on MXU0's acc[1] (idle at that point); both
# matmul engines read the same quantized FP8 mreg.
#
# Port of jrmills20 parameterized_fused_matmul_bias.py + the stock
# smolvla_fused_matmul_bias per-tile body.
#
# A is BF16; on-chip quantized to FP8 via VMATPUSH.ACC.BF16 + SELI + VMATPOP.FP8.
# B is pre-quantized FP8 in DRAM. Bias is BF16 added with VADD.BF16.
# NO VMATPUSH.ACC.FP8 preload — we mirror stock's quantize-roundtrip pattern.
#
# @TIMEOUT 500000
# @DRAM_BASE 0x00000001
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

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x25, 0x1000
    LI    x26, 0x2000
    LI    x27, 0x4000
    ADDI  x11, x0, 4

    ADDI  x10, x0, 0

output_tile_loop:
    SRLI  x12, x10, 1
    ANDI  x13, x10, 1

    # ── A[m, 0] DRAM offset = m * 2048 ──
    SLLI  x20, x12, 11
    # ── B[0, n] DRAM offset = n * 2048 + 0x1000 ──
    SLLI  x22, x13, 11
    ADD   x22, x22, x25
    # ── bias[m, n] DRAM offset = (m*2 + n) * 2048 + 0x2000 ──
    SLLI  x23, x12, 1
    ADD   x23, x23, x13
    SLLI  x23, x23, 11
    ADD   x23, x23, x26
    # ── out[m, n] DRAM offset = (m*2 + n) * 2048 + 0x4000 ──
    SLLI  x24, x12, 1
    ADD   x24, x24, x13
    SLLI  x24, x24, 11
    ADD   x24, x24, x27

    # ── Load A bf16 → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x20, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    ADDI  x1, x20, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ── Load B fp8 → VMEM[512] → m4 ──
    LI    x7, 512
    DMA.LOAD  x7, x22, x2, 0
    DMA.WAIT  0
    LI    x7, 512
    VLOAD 4, x7, 0
    DELAY 40

    # ── Load bias bf16 → VMEM[768..1279] → (m6, m7) ──
    LI    x9, 768
    DMA.LOAD  x9, x23, x2, 0
    DMA.WAIT  0
    LI    x9, 1024
    ADDI  x1, x23, 1024
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
    VMATPUSH.ACC.BF16.MXU0 1, 0
    DELAY 80
    SELI 0, 0x7F
    VMATPOP.FP8.MXU0 8, 0, 1
    DELAY 40

    # ── Dispatch to MXU0 (n_tile=0) or MXU1 (n_tile=1) ──
    BNE   x13, x0, matmul_mxu1
    NOP

    # ── MXU0 path: (m10,m11) = m8 @ m5 (= A @ B) ──
    VMATPUSH.W.MXU0 0, 5
    DELAY 40
    VMATMUL.MXU0 0, 8, 0
    DELAY 120
    VMATPOP.BF16.MXU0 10, 0
    DELAY 40
    BEQ   x0, x0, matmul_done
    NOP

matmul_mxu1:
    # ── MXU1 path ──
    VMATPUSH.W.MXU1 0, 5
    DELAY 40
    VMATMUL.MXU1 0, 8, 0
    DELAY 120
    VMATPOP.BF16.MXU1 10, 0
    DELAY 40

matmul_done:

    # ── Bias add: (m12, m13) = (m10, m11) + (m6, m7) ──
    VADD.BF16 12, 10, 6
    DELAY 150

    # ── Store (m12, m13) → VMEM[1280..1791] ──
    LI    x8, 1280
    VSTORE 12, x8, 0
    DELAY 40
    VSTORE 13, x8, 8
    DELAY 40

    # ── DMA out: two halves → C[m,n] ──
    LI    x8, 1280
    DMA.STORE x24, x8, x2, 1
    DMA.WAIT  1
    ADDI  x1, x24, 1024
    LI    x8, 1536
    DMA.STORE x1, x8, x2, 1
    DMA.WAIT  1

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

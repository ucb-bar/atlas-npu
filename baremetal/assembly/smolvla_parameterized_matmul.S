# smolvla_parameterized_matmul.S — dual-MXU tiled C = A @ B for 64x32x64
# Column-split across engines: output tiles with n_tile=0 run on MXU0
# (SA); tiles with n_tile=1 run on MXU1 (IPT). Per-engine accumulators
# are independent, so the two paths don't alias. The shared tile loop
# dispatches via a BNE on the inner n_tile index.
#
# Port of jrmills20 parameterized_matmul.py. First parameterization step:
# scale M and N to 64 (4 output tiles); keep K=32 (single K-tile per
# output, no VMATMUL.ACC accumulation).
#
# Both MXU engines load weights in transposed orientation (y = x @ w^T),
# so each B tile is VTRPOSE.XLU'd once before VMATPUSH.W regardless of
# which engine consumes it. gen_smolvla_parameterized_matmul.py routes
# per-tile goldens through SARTLLinearFunction or IPTLinearRTLFunction
# depending on which engine ran that tile.
#
# @TIMEOUT 400000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_smolvla_parameterized_matmul.py
#
# Shape: M=64, K=32, N=64 → M_tiles=2, N_tiles=2, K_tiles=1.
# Tile-major layout (matches Jeremy's _tile_matrix).
#
# DRAM layout:
#   [0x0000..0x07FF]  A FP8  (M_tiles*K_tiles = 2 tiles x 1024B)
#   [0x1000..0x17FF]  B FP8  (K_tiles*N_tiles = 2 tiles x 1024B)
#   [0x2000..0x3FFF]  C BF16 (M_tiles*N_tiles = 4 tiles x 2048B,
#                            each tile: cols 0-15 low half (1024B),
#                                       cols 16-31 high half (1024B))
#
# VMEM staging:
#   [0..255]     A tile (FP8, m0)
#   [256..511]   B tile (FP8, m4; VTRPOSE → m5)
#   [512..767]   C cols  0-15 (BF16, m2)
#   [768..1023]  C cols 16-31 (BF16, m3)
#
# Register map:
#   x2   = 1024 (tile size bytes)     x5  = 0x00000001 (DRAM base)
#   x10  = output_tile_idx (0..3)     x11 = 4 (total output tiles)
#   x12  = m_tile (idx >> 1)          x13 = n_tile (idx & 1)
#   x16  = C DRAM offset scratch      x17 = 0x2000 (C DRAM base)
#   x20  = A[m,0] DRAM offset
#   x22  = B[0,n] DRAM offset         x25 = 0x1000 (B DRAM base)
#   x1,x6,x7,x8 = VMEM/DMA scratch    x28 = fail code

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x25, 0x1000
    LI    x17, 0x2000
    ADDI  x11, x0, 4

    ADDI  x10, x0, 0           # output_tile_idx = 0

output_tile_loop:
    SRLI  x12, x10, 1          # m_tile = idx >> 1
    ANDI  x13, x10, 1          # n_tile = idx & 1

    # ── A[m, 0] DRAM offset = m_tile * 1024 ──
    SLLI  x20, x12, 10
    # ── B[0, n] DRAM offset = n_tile * 1024 + 0x1000 ──
    SLLI  x22, x13, 10
    ADD   x22, x22, x25

    # ── DMA A tile → VMEM[0] ──
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x20, x2, 0
    DMA.WAIT  0

    # ── DMA B tile → VMEM[256] ──
    LI    x6, 256
    DMA.LOAD  x6, x22, x2, 0
    DMA.WAIT  0

    # ── VLOAD A → m0, B → m4 ──
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    LI    x6, 256
    VLOAD 4, x6, 0
    DELAY 33

    # ── Transpose B → m5 (so MXU computes A @ B, not A @ B^T) ──
    VTRPOSE.XLU 5, 4
    DELAY 64

    # ── Dispatch to MXU0 (n_tile=0) or MXU1 (n_tile=1) ──
    BNE   x13, x0, matmul_mxu1
    NOP

    # ── MXU0 path: push weight + matmul + pop ──
    VMATPUSH.W.MXU0 0, 5
    DELAY 31
    VMATMUL.MXU0 0, 0, 0
    DELAY 95
    VMATPOP.BF16.MXU0 2, 0
    DELAY 31
    BEQ   x0, x0, matmul_done
    NOP

matmul_mxu1:
    # ── MXU1 path ──
    VMATPUSH.W.MXU1 0, 5
    DELAY 31
    VMATMUL.MXU1 0, 0, 0
    DELAY 34
    VMATPOP.BF16.MXU1 2, 0
    DELAY 31

matmul_done:

    # ── Store m2 (cols 0-15) → VMEM[512], m3 (cols 16-31) → VMEM[768] ──
    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 33
    LI    x8, 768
    VSTORE 3, x8, 0
    DELAY 33

    # ── DMA out: C[m,n] two halves ──
    # C DRAM offset = (m*N_tiles + n) * TILE_BYTES_BF16 + 0x2000
    #               = (m*2 + n) * 2048 + 0x2000
    SLLI  x16, x12, 1          # m*2
    ADD   x16, x16, x13        # m*2 + n
    SLLI  x16, x16, 11         # *2048
    ADD   x16, x16, x17        # + 0x2000 (C base)

    LI    x8, 512
    DMA.STORE x16, x8, x2, 1
    DMA.WAIT  1

    ADDI  x16, x16, 1024
    LI    x8, 768
    DMA.STORE x16, x8, x2, 1
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

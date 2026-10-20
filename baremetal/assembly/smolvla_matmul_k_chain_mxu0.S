# smolvla_matmul_k_chain_mxu0.S (MXU0 / SA) — port of npu-model smolvla_matmul_k_chain
# Tests: two-K-tile FP8 matmul exercising the MXU0 accumulator chain
#   C[32,32] = A[:,0:32]  @ B[0:32,:]      (VMATMUL.MXU0     → reset acc)
#            + A[:,32:64] @ B[32:64,:]     (VMATMUL.ACC.MXU0 → accumulate)
# @TIMEOUT 40000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_matmul_k_chain_mxu0.py
#
# Each B tile is XLU-transposed before being pushed to the weight slot
# so the MXU computes A_k @ B_k rather than A_k @ B_k^T.
#
# DRAM layout:
#   [0x0000..0x03FF]  A_K0 FP8 (32×32)            (1024 B, 32 beats)
#   [0x0400..0x07FF]  A_K1 FP8 (32×32)            (1024 B, 32 beats)
#   [0x0800..0x0BFF]  B_K0 FP8 (32×32)            (1024 B, 32 beats)
#   [0x0C00..0x0FFF]  B_K1 FP8 (32×32)            (1024 B, 32 beats)
#   [0x1000..0x13FF]  Output cols  0-15 BF16       (1024 B, 32 beats)
#   [0x1400..0x17FF]  Output cols 16-31 BF16       (1024 B, 32 beats)
#
# Mreg allocation:
#   m0 = A_K0   m2 = A_K1
#   m4,m5 = B_K0 raw/transposed   m6,m7 = B_K1 raw/transposed
#   m8,m9 = pop output (cols 0-15, cols 16-31)

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load A_K0 → VMEM[0] → m0 ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

    # ── Load B_K0 → VMEM[256] → m4 → m5 ──
    LI    x7, 256
    LI    x1, 0x90000800
    dma.load x7, x1, x2, ch0
    dma.wait ch0
    LI    x7, 256
    vload m4, 0(x7)
    DELAY 33
    vtrpose.xlu m5, m4
    DELAY 64

    # ── First K-tile: acc = A_K0 @ B_K0 ──
    vmatpush.weight.mxu0 w0, m5
    DELAY 31
    vmatmul.mxu0 acc0, m0, w0
    DELAY 95

    # ── Load A_K1 → VMEM[512] → m2 ──
    LI    x6, 512
    LI    x1, 0x90000400
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 512
    vload m2, 0(x6)
    DELAY 33

    # ── Load B_K1 → VMEM[768] → m6 → m7 ──
    LI    x7, 768
    LI    x1, 0x90000C00
    dma.load x7, x1, x2, ch0
    dma.wait ch0
    LI    x7, 768
    vload m6, 0(x7)
    DELAY 33
    vtrpose.xlu m7, m6
    DELAY 64

    # ── Second K-tile: acc += A_K1 @ B_K1 ──
    vmatpush.weight.mxu0 w0, m7
    DELAY 31
    vmatmul.acc.mxu0 acc0, m2, w0
    DELAY 95

    vmatpop.bf16.acc.mxu0 m8, acc0
    DELAY 31

    # ── Store m8 (cols 0-15) → VMEM[1024] → DRAM[0x1000] ──
    LI    x8, 1024
    vstore m8, 0(x8)
    DELAY 33

    # ── Store m9 (cols 16-31) → VMEM[1280] → DRAM[0x1400] ──
    LI    x9, 1280
    vstore m9, 0(x9)
    DELAY 33

    LI    x8, 1024
    LI    x3, 0x90001000
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    LI    x9, 1280
    LI    x3, 0x90001400
    dma.store x3, x9, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

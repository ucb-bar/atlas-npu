# smolvla_fused_attention.S — dual-MXU flash attention (MXU0 + MXU1)
# Tests: flash attention (online softmax) over a 2-K-tile attention block
#   Q[32,64] BF16, K[32,64] BF16 (×2 tiles), V_mlir[64,32] BF16 (×2 tiles)
#   head_dim = 64, k_seq = 64
#
#   for each K tile:
#     scores  = Q @ K^T              (MXU0, FP8 → BF16, accumulator chain)
#     scaled  = scores * scale       (VPU pair op)
#     m_new   = max(m_prev, rowmax(scaled))
#     exp_diff= exp(m_prev - m_new)
#     O      *= exp_diff             (rescale running output)
#     exp_s   = exp(scaled - m_new)
#     exp_s_q = quantize(exp_s) BF16→FP8 via VMATPUSH.ACC.BF16/VMATPOP.FP8
#     O      += exp_s_q @ V          (MXU1 — V_left + V_right halves)
#     l       = exp_diff * l + rowsum(exp_s)
#     m_prev  = m_new
#   O /= l
#
# @TIMEOUT 200000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_fused_attention.py
#
# Atlas MXU computes A @ W^T, so K and V_mlir slices are pushed as
# weight directly (no XLU): Q @ K^T uses K as weight; exp_s @ V_left
# uses V_mlir[0:32,:] as weight (since V_left = V_mlir[0:32,:]^T).
#
# BF16 → FP8 quantization uses atlas's MXU0 acc-roundtrip path:
# VMATPUSH.ACC.BF16.MXU0 acc, vs writes (m_vs, m_vs+1) BF16 directly
# into accumulator buffer; VMATPOP.FP8.MXU0 vd, scaleReg, acc reads
# the same BF16 lanes and quantizes to E4M3 (unit scale via SELI 0x7F).
# This preserves row order (unlike VFP8PACK which permutes 2→1).
#
# DRAM layout (4096 B per BF16 tile = 128 beats):
#   [0x0000..0x0FFF]  Q       BF16 [32,64]    (4 × [32,16] sub-blocks)
#   [0x1000..0x1FFF]  K0      BF16 [32,64]
#   [0x2000..0x2FFF]  K1      BF16 [32,64]
#   [0x3000..0x3FFF]  V0_mlir BF16 [64,32]
#   [0x4000..0x4FFF]  V1_mlir BF16 [64,32]
#   [0x6000..0x6FFF]  OUT     BF16 [32,64]
#
# Mreg map:
#   Persistent (across both K tiles):
#     m0,m1   = Q_lo BF16 (cols 0:16, 16:32)
#     m2,m3   = Q_hi BF16 (cols 32:48, 48:64)
#     m4      = Q_lo FP8 (post-quantize)
#     m5      = Q_hi FP8
#     m6,m7   = scale BF16 pair (broadcast 0.125 = 0x3E00)
#     m8,m9   = m_prev BF16 (init -100.0 = 0xC2C8)
#     m10,m11 = l_prev BF16 (init 0)
#     m12,m13 = O_left  BF16 (cols 0:32 of final output)
#     m14,m15 = O_right BF16 (cols 32:64 of final output)
#
#   Reused per K tile:
#     m16,m17 = K_lo BF16 (cols 0:32)
#     m18,m19 = K_hi BF16 (cols 32:64)
#     m20     = K_lo FP8
#     m21     = K_hi FP8
#     m22,m23 = V_top BF16 (V_mlir[0:32, :])
#     m24,m25 = V_bot BF16 (V_mlir[32:64, :])
#     m26     = V_top FP8
#     m27     = V_bot FP8
#
#   Per-tile temporaries reuse low architectural registers once their
#   K/V staging payloads are dead.  This avoids m_i/m_{i+32} physical-bank
#   aliases while preserving the original dataflow.
#     m16,m17 = tile_max, later exp_diff * l, later 1/l
#     m18,m19 = m_new
#     m20,m21 = exp_diff (broadcast)
#     m22,m23 = exp_s BF16 [32,32]
#     m24     = exp_s FP8; m24,m25 later rowsum(exp_s)
#     m28,m29 = scores BF16, later vc_left BF16
#     m30,m31 = scaled BF16, later vc_right BF16

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 4096          # tile size in bytes (Q/K/V/OUT all 4096 B)

    # ────────────────────────────────────────────────────────────────────
    # Load Q: DRAM[0x0000..0x0FFF] → VMEM[0..1023] → m0..m3
    # ────────────────────────────────────────────────────────────────────
    ADDI  x6, x0, 0          # VMEM word addr 0
    LI    x1, 0x90000000  # DRAM byte addr 0
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33
    vload m2, 16(x6)
    DELAY 33
    vload m3, 24(x6)
    DELAY 33

    # ── Quantize Q_lo (m0,m1) → m4 FP8 via acc[1] roundtrip ──
    vmatpush.acc.bf16.mxu0 acc1, m0
    DELAY 31
    seli e0, 0x7F                # E8M0 unit scale (2^0 = 1.0)
    vmatpop.fp8.acc.mxu0 m4, acc1, e0
    DELAY 31

    # ── Quantize Q_hi (m2,m3) → m5 FP8 ──
    vmatpush.acc.bf16.mxu0 acc1, m2
    DELAY 31
    vmatpop.fp8.acc.mxu0 m5, acc1, e0
    DELAY 31

    # ────────────────────────────────────────────────────────────────────
    # Initialize persistent online-softmax state
    # ────────────────────────────────────────────────────────────────────
    vli.all m6, 0x3E00       # scale = 0.125 (1/sqrt(64)) — fills (m6, m7)
    DELAY 63
    vli.all m8, 0xC2C8       # m_prev = -100.0 BF16 (-inf init) — fills (m8, m9)
    DELAY 63
    vli.all m10, 0x0000      # l_prev = 0 — fills (m10, m11)
    DELAY 63
    vli.all m12, 0x0000      # O_left = 0 — fills (m12, m13)
    DELAY 63
    vli.all m14, 0x0000      # O_right = 0 — fills (m14, m15)
    DELAY 63

    # ════════════════════════════════════════════════════════════════════
    # K TILE 0  (k_seq 0:32)
    # ════════════════════════════════════════════════════════════════════

    # ── Load K0: DRAM[0x1000..0x1FFF] → VMEM[1024..2047] → m16..m19 ──
    LI    x6, 1024
    LI    x1, 0x90001000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 1024
    vload m16, 0(x6)
    DELAY 33
    vload m17, 8(x6)
    DELAY 33
    vload m18, 16(x6)
    DELAY 33
    vload m19, 24(x6)
    DELAY 33

    # ── Quantize K_lo, K_hi to FP8 ──
    vmatpush.acc.bf16.mxu0 acc1, m16
    DELAY 31
    vmatpop.fp8.acc.mxu0 m20, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu0 acc1, m18
    DELAY 31
    vmatpop.fp8.acc.mxu0 m21, acc1, e0
    DELAY 31

    # ── Load V0_mlir: DRAM[0x3000..0x3FFF] → VMEM[2048..3071] → m22..m25 ──
    LI    x6, 2048
    LI    x1, 0x90003000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 2048
    vload m22, 0(x6)         # V_top cols 0:16
    DELAY 33
    vload m23, 8(x6)         # V_top cols 16:32
    DELAY 33
    vload m24, 16(x6)        # V_bot cols 0:16
    DELAY 33
    vload m25, 24(x6)        # V_bot cols 16:32
    DELAY 33

    # ── Quantize V_top, V_bot to FP8 ──
    vmatpush.acc.bf16.mxu0 acc1, m22
    DELAY 31
    vmatpop.fp8.acc.mxu0 m26, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu0 acc1, m24
    DELAY 31
    vmatpop.fp8.acc.mxu0 m27, acc1, e0
    DELAY 31

    # ── Q @ K^T: scores = Q_lo @ K_lo^T + Q_hi @ K_hi^T ──
    vmatpush.weight.mxu0 w0, m20    # weight = K_lo FP8
    DELAY 31
    vmatmul.mxu0 acc0, m4, w0     # acc[0] = Q_lo @ K_lo^T
    DELAY 95
    vmatpush.weight.mxu0 w0, m21    # weight = K_hi FP8
    DELAY 31
    vmatmul.acc.mxu0 acc0, m5, w0 # acc[0] += Q_hi @ K_hi^T
    DELAY 95
    vmatpop.bf16.acc.mxu0 m28, acc0  # (m28, m29) = scores BF16
    DELAY 31

    # ── Scale scores ──
    vmul.bf16 m30, m28, m6
    DELAY 64

    # ── tile_max = rowmax(scaled), broadcast across 32 cols ──
    vredmax.row.bf16 m16, m30
    DELAY 32

    # ── m_new = max(m_prev, tile_max) ──
    vmaximum.bf16 m18, m8, m16
    DELAY 64

    # ── exp_diff = exp(m_prev - m_new) ──
    vsub.bf16 m20, m8, m18
    DELAY 64
    vexp.bf16 m20, m20
    DELAY 64

    # ── Rescale running output: O *= exp_diff ──
    vmul.bf16 m12, m12, m20
    DELAY 64
    vmul.bf16 m14, m14, m20
    DELAY 64

    # ── exp_s = exp(scaled - m_new) ──
    vsub.bf16 m22, m30, m18
    DELAY 64
    vexp.bf16 m22, m22
    DELAY 64

    # ── Quantize exp_s BF16 → FP8 via acc[1] roundtrip ──
    vmatpush.acc.bf16.mxu0 acc1, m22
    DELAY 31
    vmatpop.fp8.acc.mxu0 m24, acc1, e0
    DELAY 31

    # ── exp_s @ V_left:  weight = m26 (V_top FP8) ──
    vmatpush.weight.mxu1 w0, m26
    DELAY 31
    vmatmul.mxu1 acc0, m24, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m28, acc0   # (m28, m29) = vc_left BF16
    DELAY 31

    # ── exp_s @ V_right: weight = m27 (V_bot FP8) ──
    vmatpush.weight.mxu1 w0, m27
    DELAY 31
    vmatmul.mxu1 acc0, m24, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m30, acc0   # (m30, m31) = vc_right BF16
    DELAY 31

    # ── O += V contributions ──
    vadd.bf16 m12, m12, m28
    DELAY 64
    vadd.bf16 m14, m14, m30
    DELAY 64

    # ── l = exp_diff * l + rowsum(exp_s) ──
    vmul.bf16 m16, m20, m10
    DELAY 64
    vredsum.row.bf16 m24, m22
    DELAY 37
    vadd.bf16 m10, m16, m24
    DELAY 64

    # ── m_prev = m_new (pair-op VMOV copies both halves) ──
    vmov m8, m18
    DELAY 64

    # ════════════════════════════════════════════════════════════════════
    # K TILE 1  (k_seq 32:64) — same body, K1 / V1 inputs
    # ════════════════════════════════════════════════════════════════════

    # ── Load K1 ──
    LI    x6, 1024
    LI    x1, 0x90002000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 1024
    vload m16, 0(x6)
    DELAY 33
    vload m17, 8(x6)
    DELAY 33
    vload m18, 16(x6)
    DELAY 33
    vload m19, 24(x6)
    DELAY 33

    vmatpush.acc.bf16.mxu0 acc1, m16
    DELAY 31
    vmatpop.fp8.acc.mxu0 m20, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu0 acc1, m18
    DELAY 31
    vmatpop.fp8.acc.mxu0 m21, acc1, e0
    DELAY 31

    # ── Load V1_mlir ──
    LI    x6, 2048
    LI    x1, 0x90004000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 2048
    vload m22, 0(x6)
    DELAY 33
    vload m23, 8(x6)
    DELAY 33
    vload m24, 16(x6)
    DELAY 33
    vload m25, 24(x6)
    DELAY 33

    vmatpush.acc.bf16.mxu0 acc1, m22
    DELAY 31
    vmatpop.fp8.acc.mxu0 m26, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu0 acc1, m24
    DELAY 31
    vmatpop.fp8.acc.mxu0 m27, acc1, e0
    DELAY 31

    vmatpush.weight.mxu0 w0, m20
    DELAY 31
    vmatmul.mxu0 acc0, m4, w0
    DELAY 95
    vmatpush.weight.mxu0 w0, m21
    DELAY 31
    vmatmul.acc.mxu0 acc0, m5, w0
    DELAY 95
    vmatpop.bf16.acc.mxu0 m28, acc0
    DELAY 31

    vmul.bf16 m30, m28, m6
    DELAY 64
    vredmax.row.bf16 m16, m30
    DELAY 32
    vmaximum.bf16 m18, m8, m16
    DELAY 64
    vsub.bf16 m20, m8, m18
    DELAY 64
    vexp.bf16 m20, m20
    DELAY 64

    vmul.bf16 m12, m12, m20
    DELAY 64
    vmul.bf16 m14, m14, m20
    DELAY 64

    vsub.bf16 m22, m30, m18
    DELAY 64
    vexp.bf16 m22, m22
    DELAY 64

    vmatpush.acc.bf16.mxu0 acc1, m22
    DELAY 31
    vmatpop.fp8.acc.mxu0 m24, acc1, e0
    DELAY 31

    vmatpush.weight.mxu1 w0, m26
    DELAY 31
    vmatmul.mxu1 acc0, m24, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m28, acc0
    DELAY 31

    vmatpush.weight.mxu1 w0, m27
    DELAY 31
    vmatmul.mxu1 acc0, m24, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m30, acc0
    DELAY 31

    vadd.bf16 m12, m12, m28
    DELAY 64
    vadd.bf16 m14, m14, m30
    DELAY 64

    vmul.bf16 m16, m20, m10
    DELAY 64
    vredsum.row.bf16 m24, m22
    DELAY 37
    vadd.bf16 m10, m16, m24
    DELAY 64

    vmov m8, m18
    DELAY 64

    # ════════════════════════════════════════════════════════════════════
    # Normalize: O /= l
    # ════════════════════════════════════════════════════════════════════
    vrecip.bf16 m16, m10
    DELAY 64
    vmul.bf16 m12, m12, m16
    DELAY 64
    vmul.bf16 m14, m14, m16
    DELAY 64

    # ════════════════════════════════════════════════════════════════════
    # Store O → VMEM[3072..4095] → DRAM[0x6000]
    # ════════════════════════════════════════════════════════════════════
    LI    x8, 3072
    vstore m12, 0(x8)
    DELAY 33
    vstore m13, 8(x8)
    DELAY 33
    vstore m14, 16(x8)
    DELAY 33
    vstore m15, 24(x8)
    DELAY 33

    LI    x8, 3072
    LI    x3, 0x90006000
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

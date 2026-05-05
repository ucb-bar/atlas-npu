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
# @DRAM_BASE 0x00000001
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

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 4096          # tile size in bytes (Q/K/V/OUT all 4096 B)

    # ────────────────────────────────────────────────────────────────────
    # Load Q: DRAM[0x0000..0x0FFF] → VMEM[0..1023] → m0..m3
    # ────────────────────────────────────────────────────────────────────
    ADDI  x6, x0, 0          # VMEM word addr 0
    ADDI  x1, x0, 0          # DRAM byte addr 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40
    VLOAD 2, x6, 16
    DELAY 40
    VLOAD 3, x6, 24
    DELAY 40

    # ── Quantize Q_lo (m0,m1) → m4 FP8 via acc[1] roundtrip ──
    VMATPUSH.ACC.BF16.MXU0 1, 0
    DELAY 80
    SELI 0, 0x7F                # E8M0 unit scale (2^0 = 1.0)
    VMATPOP.FP8.MXU0 4, 0, 1
    DELAY 40

    # ── Quantize Q_hi (m2,m3) → m5 FP8 ──
    VMATPUSH.ACC.BF16.MXU0 1, 2
    DELAY 80
    VMATPOP.FP8.MXU0 5, 0, 1
    DELAY 40

    # ────────────────────────────────────────────────────────────────────
    # Initialize persistent online-softmax state
    # ────────────────────────────────────────────────────────────────────
    VLI.ALL 6, 0x3E00       # scale = 0.125 (1/sqrt(64)) — fills (m6, m7)
    DELAY 150
    VLI.ALL 8, 0xC2C8       # m_prev = -100.0 BF16 (-inf init) — fills (m8, m9)
    DELAY 150
    VLI.ALL 10, 0x0000      # l_prev = 0 — fills (m10, m11)
    DELAY 150
    VLI.ALL 12, 0x0000      # O_left = 0 — fills (m12, m13)
    DELAY 150
    VLI.ALL 14, 0x0000      # O_right = 0 — fills (m14, m15)
    DELAY 150

    # ════════════════════════════════════════════════════════════════════
    # K TILE 0  (k_seq 0:32)
    # ════════════════════════════════════════════════════════════════════

    # ── Load K0: DRAM[0x1000..0x1FFF] → VMEM[1024..2047] → m16..m19 ──
    LI    x6, 1024
    LI    x1, 0x1000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1024
    VLOAD 16, x6, 0
    DELAY 40
    VLOAD 17, x6, 8
    DELAY 40
    VLOAD 18, x6, 16
    DELAY 40
    VLOAD 19, x6, 24
    DELAY 40

    # ── Quantize K_lo, K_hi to FP8 ──
    VMATPUSH.ACC.BF16.MXU0 1, 16
    DELAY 80
    VMATPOP.FP8.MXU0 20, 0, 1
    DELAY 40
    VMATPUSH.ACC.BF16.MXU0 1, 18
    DELAY 80
    VMATPOP.FP8.MXU0 21, 0, 1
    DELAY 40

    # ── Load V0_mlir: DRAM[0x3000..0x3FFF] → VMEM[2048..3071] → m22..m25 ──
    LI    x6, 2048
    LI    x1, 0x3000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 2048
    VLOAD 22, x6, 0         # V_top cols 0:16
    DELAY 40
    VLOAD 23, x6, 8         # V_top cols 16:32
    DELAY 40
    VLOAD 24, x6, 16        # V_bot cols 0:16
    DELAY 40
    VLOAD 25, x6, 24        # V_bot cols 16:32
    DELAY 40

    # ── Quantize V_top, V_bot to FP8 ──
    VMATPUSH.ACC.BF16.MXU0 1, 22
    DELAY 80
    VMATPOP.FP8.MXU0 26, 0, 1
    DELAY 40
    VMATPUSH.ACC.BF16.MXU0 1, 24
    DELAY 80
    VMATPOP.FP8.MXU0 27, 0, 1
    DELAY 40

    # ── Q @ K^T: scores = Q_lo @ K_lo^T + Q_hi @ K_hi^T ──
    VMATPUSH.W.MXU0 0, 20    # weight = K_lo FP8
    DELAY 40
    VMATMUL.MXU0 0, 4, 0     # acc[0] = Q_lo @ K_lo^T
    DELAY 200
    VMATPUSH.W.MXU0 0, 21    # weight = K_hi FP8
    DELAY 40
    VMATMUL.ACC.MXU0 0, 5, 0 # acc[0] += Q_hi @ K_hi^T
    DELAY 200
    VMATPOP.BF16.MXU0 28, 0  # (m28, m29) = scores BF16
    DELAY 40

    # ── Scale scores ──
    VMUL.BF16 30, 28, 6
    DELAY 150

    # ── tile_max = rowmax(scaled), broadcast across 32 cols ──
    VREDMAX.ROW.BF16 16, 30
    DELAY 150

    # ── m_new = max(m_prev, tile_max) ──
    VMAX.BF16 18, 8, 16
    DELAY 150

    # ── exp_diff = exp(m_prev - m_new) ──
    VSUB.BF16 20, 8, 18
    DELAY 150
    VEXP 20, 20
    DELAY 150

    # ── Rescale running output: O *= exp_diff ──
    VMUL.BF16 12, 12, 20
    DELAY 150
    VMUL.BF16 14, 14, 20
    DELAY 150

    # ── exp_s = exp(scaled - m_new) ──
    VSUB.BF16 22, 30, 18
    DELAY 150
    VEXP 22, 22
    DELAY 150

    # ── Quantize exp_s BF16 → FP8 via acc[1] roundtrip ──
    VMATPUSH.ACC.BF16.MXU0 1, 22
    DELAY 80
    VMATPOP.FP8.MXU0 24, 0, 1
    DELAY 40

    # ── exp_s @ V_left:  weight = m26 (V_top FP8) ──
    VMATPUSH.W.MXU1 0, 26
    DELAY 40
    VMATMUL.MXU1 0, 24, 0
    DELAY 200
    VMATPOP.BF16.MXU1 28, 0   # (m28, m29) = vc_left BF16
    DELAY 40

    # ── exp_s @ V_right: weight = m27 (V_bot FP8) ──
    VMATPUSH.W.MXU1 0, 27
    DELAY 40
    VMATMUL.MXU1 0, 24, 0
    DELAY 200
    VMATPOP.BF16.MXU1 30, 0   # (m30, m31) = vc_right BF16
    DELAY 40

    # ── O += V contributions ──
    VADD.BF16 12, 12, 28
    DELAY 150
    VADD.BF16 14, 14, 30
    DELAY 150

    # ── l = exp_diff * l + rowsum(exp_s) ──
    VMUL.BF16 16, 20, 10
    DELAY 150
    VREDSUM.ROW.BF16 24, 22
    DELAY 150
    VADD.BF16 10, 16, 24
    DELAY 150

    # ── m_prev = m_new (pair-op VMOV copies both halves) ──
    VMOV 8, 18
    DELAY 150

    # ════════════════════════════════════════════════════════════════════
    # K TILE 1  (k_seq 32:64) — same body, K1 / V1 inputs
    # ════════════════════════════════════════════════════════════════════

    # ── Load K1 ──
    LI    x6, 1024
    LI    x1, 0x2000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1024
    VLOAD 16, x6, 0
    DELAY 40
    VLOAD 17, x6, 8
    DELAY 40
    VLOAD 18, x6, 16
    DELAY 40
    VLOAD 19, x6, 24
    DELAY 40

    VMATPUSH.ACC.BF16.MXU0 1, 16
    DELAY 80
    VMATPOP.FP8.MXU0 20, 0, 1
    DELAY 40
    VMATPUSH.ACC.BF16.MXU0 1, 18
    DELAY 80
    VMATPOP.FP8.MXU0 21, 0, 1
    DELAY 40

    # ── Load V1_mlir ──
    LI    x6, 2048
    LI    x1, 0x4000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 2048
    VLOAD 22, x6, 0
    DELAY 40
    VLOAD 23, x6, 8
    DELAY 40
    VLOAD 24, x6, 16
    DELAY 40
    VLOAD 25, x6, 24
    DELAY 40

    VMATPUSH.ACC.BF16.MXU0 1, 22
    DELAY 80
    VMATPOP.FP8.MXU0 26, 0, 1
    DELAY 40
    VMATPUSH.ACC.BF16.MXU0 1, 24
    DELAY 80
    VMATPOP.FP8.MXU0 27, 0, 1
    DELAY 40

    VMATPUSH.W.MXU0 0, 20
    DELAY 40
    VMATMUL.MXU0 0, 4, 0
    DELAY 200
    VMATPUSH.W.MXU0 0, 21
    DELAY 40
    VMATMUL.ACC.MXU0 0, 5, 0
    DELAY 200
    VMATPOP.BF16.MXU0 28, 0
    DELAY 40

    VMUL.BF16 30, 28, 6
    DELAY 150
    VREDMAX.ROW.BF16 16, 30
    DELAY 150
    VMAX.BF16 18, 8, 16
    DELAY 150
    VSUB.BF16 20, 8, 18
    DELAY 150
    VEXP 20, 20
    DELAY 150

    VMUL.BF16 12, 12, 20
    DELAY 150
    VMUL.BF16 14, 14, 20
    DELAY 150

    VSUB.BF16 22, 30, 18
    DELAY 150
    VEXP 22, 22
    DELAY 150

    VMATPUSH.ACC.BF16.MXU0 1, 22
    DELAY 80
    VMATPOP.FP8.MXU0 24, 0, 1
    DELAY 40

    VMATPUSH.W.MXU1 0, 26
    DELAY 40
    VMATMUL.MXU1 0, 24, 0
    DELAY 200
    VMATPOP.BF16.MXU1 28, 0
    DELAY 40

    VMATPUSH.W.MXU1 0, 27
    DELAY 40
    VMATMUL.MXU1 0, 24, 0
    DELAY 200
    VMATPOP.BF16.MXU1 30, 0
    DELAY 40

    VADD.BF16 12, 12, 28
    DELAY 150
    VADD.BF16 14, 14, 30
    DELAY 150

    VMUL.BF16 16, 20, 10
    DELAY 150
    VREDSUM.ROW.BF16 24, 22
    DELAY 150
    VADD.BF16 10, 16, 24
    DELAY 150

    VMOV 8, 18
    DELAY 150

    # ════════════════════════════════════════════════════════════════════
    # Normalize: O /= l
    # ════════════════════════════════════════════════════════════════════
    VRECIP.BF16 16, 10
    DELAY 150
    VMUL.BF16 12, 12, 16
    DELAY 150
    VMUL.BF16 14, 14, 16
    DELAY 150

    # ════════════════════════════════════════════════════════════════════
    # Store O → VMEM[3072..4095] → DRAM[0x6000]
    # ════════════════════════════════════════════════════════════════════
    LI    x8, 3072
    VSTORE 12, x8, 0
    DELAY 40
    VSTORE 13, x8, 8
    DELAY 40
    VSTORE 14, x8, 16
    DELAY 40
    VSTORE 15, x8, 24
    DELAY 40

    LI    x8, 3072
    LI    x3, 0x6000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

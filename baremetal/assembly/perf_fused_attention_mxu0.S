# gen_perf_fused_attention_mxu0.S
# Tests: flash attention (online softmax) over a 2-K-tile attention block
#
# This variant uses the VMATPUSH.ACC.BF16 / VMATPOP.FP8.MXU0 acc-roundtrip
# for BF16 -> FP8 quantization (instead of VFP8PACK). The acc-roundtrip
# uses MXU0's accumulator buffer 1 as a staging area:
#
#   VMATPUSH.ACC.BF16.MXU0 1, vs   # acc[1] = {m[vs], m[vs+1]}  (32x32 BF16)
#   VMATPOP.FP8.MXU0 vd, 0, 1      # m[vd] = quantize_fp8(acc[1], e[0])
#
# Buffer 0 stays reserved for the actual Q@K^T and exp_s@V matmuls.
# SELI 0, 0x7F sets the unit-scale (E8M0 = 0x7F) for all FP8 pops.
#
# @TIMEOUT 200000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_perf_fused_attention_mxu0.py
#
# Mreg map (Updated for all-upfront loading):
#   Loaded FP8 Tiles (Persistent across math):
#     m0,m1   = K0 FP8            m2,m3   = V0 FP8
#     m4,m5   = Q FP8             m20,m21 = K1 FP8
#     m22,m23 = V1 FP8
#
#   Math State (Persistent):
#     m6,m7   = scale BF16        m8,m9   = m_prev BF16
#     m10,m11 = l_prev BF16       m12,m13 = O_left BF16
#     m14,m15 = O_right BF16
#
#   Scratch (Reused per K tile):
#     m16..m19, m24..m31 = Used for intermediate math operations
# theoretical = (16 * 64 + 32*4)*2 + 3*64 = 2496 
# actual = 3430
# utilization = 72.7%

    ADDI  x28, x0, 1

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 4096

# ═══════════════════════════════════════════════════════════════
# PRE-LOAD: all DMA loads issued and waited
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0       # Q
    LI    x6, 0x1000
    LI    x1, 0x1000
    DMA.LOAD  x6, x1, x2, 1       # K0
    LI    x6, 0x2000
    LI    x1, 0x2000
    DMA.LOAD  x6, x1, x2, 2       # K1
    LI    x6, 0x3000
    LI    x1, 0x3000
    DMA.LOAD  x6, x1, x2, 3       # V0
    LI    x6, 0x4000
    LI    x1, 0x4000
    DMA.LOAD  x6, x1, x2, 4       # V1

    DMA.WAIT  0
    DMA.WAIT  1
    DMA.WAIT  2
    DMA.WAIT  3
    DMA.WAIT  4

    # ── Set unit scale (E8M0 2^0 = 1.0) for every VMATPOP.FP8 below ──
    SELI 0, 0x7F

# ═══════════════════════════════════════════════════════════════
# VLOAD & QUANTIZE: Process all tiles into FP8 upfront
# Each VFP8PACK <vd>, <vs>, 0 has been replaced by:
#   VMATPUSH.ACC.BF16.MXU0 1, <vs>   ; acc[1] = {m[vs], m[vs+1]}
#   VMATPOP.FP8.MXU0 <vd>, 0, 1      ; m[vd] = quantize(acc[1])
# ═══════════════════════════════════════════════════════════════

    # ── Load Q → m0..m3, pack to m4..m5 ──
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33
    VLOAD 2, x6, 16
    DELAY 33
    VLOAD 3, x6, 24
    DELAY 33
    VMATPUSH.ACC.BF16.MXU0 1, 0
    DELAY 31
    VMATPOP.FP8.MXU0 4, 0, 1
    DELAY 31
    VMATPUSH.ACC.BF16.MXU0 1, 2
    DELAY 31
    VMATPOP.FP8.MXU0 5, 0, 1
    DELAY 31

    # ── Load K0 → m16..m19, pack to m0..m1 ──
    LI    x6, 0x1000
    VLOAD 16, x6, 0
    DELAY 33
    VLOAD 17, x6, 8
    DELAY 33
    VLOAD 18, x6, 16
    DELAY 33
    VLOAD 19, x6, 24
    DELAY 33
    VMATPUSH.ACC.BF16.MXU0 1, 16
    DELAY 31
    VMATPOP.FP8.MXU0 0, 0, 1
    DELAY 31
    VMATPUSH.ACC.BF16.MXU0 1, 18
    DELAY 31
    VMATPOP.FP8.MXU0 1, 0, 1
    DELAY 31

    # ── Load V0 → m16..m19, pack to m2..m3 ──
    LI    x6, 0x3000
    VLOAD 16, x6, 0
    DELAY 33
    VLOAD 17, x6, 8
    DELAY 33
    VLOAD 18, x6, 16
    DELAY 33
    VLOAD 19, x6, 24
    DELAY 33
    VMATPUSH.ACC.BF16.MXU0 1, 16
    DELAY 31
    VMATPOP.FP8.MXU0 2, 0, 1
    DELAY 31
    VMATPUSH.ACC.BF16.MXU0 1, 18
    DELAY 31
    VMATPOP.FP8.MXU0 3, 0, 1
    DELAY 31

    # ── Load K1 → m16..m19, pack to m20..m21 ──
    LI    x6, 0x2000
    VLOAD 16, x6, 0
    DELAY 33
    VLOAD 17, x6, 8
    DELAY 33
    VLOAD 18, x6, 16
    DELAY 33
    VLOAD 19, x6, 24
    DELAY 33
    VMATPUSH.ACC.BF16.MXU0 1, 16
    DELAY 31
    VMATPOP.FP8.MXU0 20, 0, 1
    DELAY 31
    VMATPUSH.ACC.BF16.MXU0 1, 18
    DELAY 31
    VMATPOP.FP8.MXU0 21, 0, 1
    DELAY 31

    # ── Load V1 → m16..m19, pack to m22..m23 ──
    LI    x6, 0x4000
    VLOAD 16, x6, 0
    DELAY 33
    VLOAD 17, x6, 8
    DELAY 33
    VLOAD 18, x6, 16
    DELAY 33
    VLOAD 19, x6, 24
    DELAY 33
    VMATPUSH.ACC.BF16.MXU0 1, 16
    DELAY 31
    VMATPOP.FP8.MXU0 22, 0, 1
    DELAY 31
    VMATPUSH.ACC.BF16.MXU0 1, 18
    DELAY 31
    VMATPOP.FP8.MXU0 23, 0, 1
    DELAY 31

    # ── Initialize persistent state ──────────────────────────────────────
    VLI.ALL 6,  0x3E00          # scale  = 0.125
    DELAY 63
    VLI.ALL 8,  0xC2C8          # m_prev = -100.0
    DELAY 63
    VLI.ALL 10, 0x0000          # l_prev = 0
    DELAY 63
    VLI.ALL 12, 0x0000          # O_left = 0
    DELAY 63
    VLI.ALL 14, 0x0000          # O_right = 0
    DELAY 63

# ═══════════════════════════════════════════════════════════════
# CYCLE COUNTER — start
# ═══════════════════════════════════════════════════════════════
    CSRRS x20, 0xC00, x0

    # ════════════════════════════════════════════════════════════════════
    # K TILE 0
    # ════════════════════════════════════════════════════════════════════

    # ── Q @ K0^T ────────────────────────────────────────────────────────
    VMATPUSH.W.MXU0 0, 0
    DELAY 31
    VMATMUL.MXU0 0, 4, 0
    DELAY 95
    VMATPUSH.W.MXU0 0, 1
    DELAY 31
    VMATMUL.ACC.MXU0 0, 5, 0
    DELAY 95
    VMATPOP.BF16.MXU0 28, 0      # m28,29 = scores
    DELAY 31

    # ── Scale scores ─────────────────────────────────────────────────────
    VMUL.BF16 26, 28, 6          # m26,27 = scaled
    DELAY 64

    # ── tile_max = rowmax(scaled) ────────────────────────────────────────
    VREDMAX.ROW.BF16 16, 26      # m16,17 = tile_max
    DELAY 32

    # ── m_new = max(m_prev, tile_max) ───────────────────────────────────
    VMAX.BF16 18, 8, 16          # m18,19 = m_new
    DELAY 64

    # ── exp_diff = exp(m_prev - m_new) ──────────────────────────────────
    VSUB.BF16 24, 8, 18          # m24,25 = exp_diff
    DELAY 64
    VEXP 24, 24
    DELAY 64

    # ── O *= exp_diff ────────────────────────────────────────────────────
    VMUL.BF16 12, 12, 24
    DELAY 64
    VMUL.BF16 14, 14, 24
    DELAY 64

    # ── exp_s = exp(scaled - m_new) ─────────────────────────────────────
    VSUB.BF16 26, 26, 18         # m26,27 = exp_s (overwrites scaled)
    DELAY 64
    VEXP 26, 26
    DELAY 64

    # ── Quantize exp_s → FP8 (acc-roundtrip via buffer 1) ───────────────
    VMATPUSH.ACC.BF16.MXU0 1, 26
    DELAY 31
    VMATPOP.FP8.MXU0 30, 0, 1    # m30 = exp_s_q
    DELAY 31

    # ── l = exp_diff * l + rowsum(exp_s) ────────────────────────────────
    VMUL.BF16 16, 24, 10         # m16,17 = exp_diff * l (overwrites tile_max)
    DELAY 64
    VREDSUM.ROW.BF16 24, 26      # m24,25 = rowsum (overwrites exp_diff)
    DELAY 37
    VADD.BF16 10, 16, 24         # l = new_l
    DELAY 64

    # ── m_prev = m_new ───────────────────────────────────────────────────
    VMOV 8, 18
    DELAY 64

    # ── exp_s @ V0_left ──────────────────────────────────────────────────
    VMATPUSH.W.MXU0 0, 2
    DELAY 31
    VMATMUL.MXU0 0, 30, 0
    DELAY 95
    VMATPOP.BF16.MXU0 26, 0      # m26,27 = V_left contrib (overwrites exp_s)
    DELAY 31

    # ── exp_s @ V0_right ─────────────────────────────────────────────────
    VMATPUSH.W.MXU0 0, 3
    DELAY 31
    VMATMUL.MXU0 0, 30, 0
    DELAY 95
    VMATPOP.BF16.MXU0 28, 0      # m28,29 = V_right contrib (overwrites scores)
    DELAY 31

    # ── O += V contributions ─────────────────────────────────────────────
    VADD.BF16 12, 12, 26
    DELAY 64
    VADD.BF16 14, 14, 28
    DELAY 64

    # ════════════════════════════════════════════════════════════════════
    # K TILE 1
    # ════════════════════════════════════════════════════════════════════

    # ── Q @ K1^T ────────────────────────────────────────────────────────
    VMATPUSH.W.MXU0 0, 20
    DELAY 31
    VMATMUL.MXU0 0, 4, 0
    DELAY 95
    VMATPUSH.W.MXU0 0, 21
    DELAY 31
    VMATMUL.ACC.MXU0 0, 5, 0
    DELAY 95
    VMATPOP.BF16.MXU0 28, 0
    DELAY 31

    # ── Scale scores ─────────────────────────────────────────────────────
    VMUL.BF16 26, 28, 6
    DELAY 64

    # ── tile_max = rowmax(scaled) ────────────────────────────────────────
    VREDMAX.ROW.BF16 16, 26
    DELAY 32

    # ── m_new = max(m_prev, tile_max) ───────────────────────────────────
    VMAX.BF16 18, 8, 16
    DELAY 64

    # ── exp_diff = exp(m_prev - m_new) ──────────────────────────────────
    VSUB.BF16 24, 8, 18
    DELAY 64
    VEXP 24, 24
    DELAY 64

    # ── O *= exp_diff ────────────────────────────────────────────────────
    VMUL.BF16 12, 12, 24
    DELAY 64
    VMUL.BF16 14, 14, 24
    DELAY 64

    # ── exp_s = exp(scaled - m_new) ─────────────────────────────────────
    VSUB.BF16 26, 26, 18
    DELAY 64
    VEXP 26, 26
    DELAY 64

    # ── Quantize exp_s → FP8 (acc-roundtrip via buffer 1) ───────────────
    VMATPUSH.ACC.BF16.MXU0 1, 26
    DELAY 31
    VMATPOP.FP8.MXU0 30, 0, 1
    DELAY 31

    # ── l = exp_diff * l + rowsum(exp_s) ────────────────────────────────
    VMUL.BF16 16, 24, 10
    DELAY 64
    VREDSUM.ROW.BF16 24, 26
    DELAY 37
    VADD.BF16 10, 16, 24
    DELAY 64

    # ── m_prev = m_new ───────────────────────────────────────────────────
    VMOV 8, 18
    DELAY 64

    # ── exp_s @ V1_left ──────────────────────────────────────────────────
    VMATPUSH.W.MXU0 0, 22
    DELAY 31
    VMATMUL.MXU0 0, 30, 0
    DELAY 95
    VMATPOP.BF16.MXU0 26, 0
    DELAY 31

    # ── exp_s @ V1_right ─────────────────────────────────────────────────
    VMATPUSH.W.MXU0 0, 23
    DELAY 31
    VMATMUL.MXU0 0, 30, 0
    DELAY 95
    VMATPOP.BF16.MXU0 28, 0
    DELAY 31

    # ── O += V contributions ─────────────────────────────────────────────
    VADD.BF16 12, 12, 26
    DELAY 64
    VADD.BF16 14, 14, 28
    DELAY 64

    # ════════════════════════════════════════════════════════════════════
    # Normalize: O /= l
    # ════════════════════════════════════════════════════════════════════
    VRECIP.BF16 16, 10
    DELAY 64
    VMUL.BF16 12, 12, 16
    DELAY 64
    VMUL.BF16 14, 14, 16
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# CYCLE COUNTER — end
# ═══════════════════════════════════════════════════════════════
    CSRRS x21, 0xC00, x0
    SUB   x22, x21, x20         # Final active duration without pauses
    CSRRW x0,  0xC11, x22       # Write out total cycles

# ═══════════════════════════════════════════════════════════════
# POST-STORE: Dump O to VMEM, then DMA to DRAM
# ═══════════════════════════════════════════════════════════════
    LI    x8, 0x1400
    VSTORE 12, x8, 0
    DELAY 33
    VSTORE 13, x8, 8
    DELAY 33
    VSTORE 14, x8, 16
    DELAY 33
    VSTORE 15, x8, 24
    DELAY 33

    LI    x8, 0x1400
    LI    x3, 0x6000
    DMA.STORE x3, x8, x2, 0
    DMA.WAIT  0

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
# gen_perf_fused_attention_MXU1.S
# Tests: flash attention (online softmax) over a 2-K-tile attention block
#
# This variant uses the VMATPUSH.ACC.BF16 / VMATPOP.FP8.ACC.MXU1 acc-roundtrip
# for BF16 -> FP8 quantization (instead of VFP8PACK). The acc-roundtrip
# uses MXU1's accumulator buffer 1 as a staging area:
#
#   VMATPUSH.ACC.BF16.MXU1 acc1, m<vs>   # acc[1] = {m[vs], m[vs+1]}  (32x32 BF16)
#   VMATPOP.FP8.ACC.MXU1 m<vd>, acc1, e0      # m[vd] = quantize_fp8(acc[1], e[0])
#
# Buffer 0 stays reserved for the actual Q@K^T and exp_s@V matmuls.
# SELI e0, 0x7F sets the unit-scale (E8M0 = 0x7F) for all FP8 pops.
#
# @TIMEOUT 200000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_perf_fused_attention_MXU1.py
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
# actual = 2942
# utilization = 84.8%

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 4096

# ═══════════════════════════════════════════════════════════════
# PRE-LOAD: all DMA loads issued and waited
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0       # Q
    LI    x6, 0x1000
    LI    x1, 0x90001000
    dma.load x6, x1, x2, ch1       # K0
    LI    x6, 0x2000
    LI    x1, 0x90002000
    dma.load x6, x1, x2, ch2       # K1
    LI    x6, 0x3000
    LI    x1, 0x90003000
    dma.load x6, x1, x2, ch3       # V0
    LI    x6, 0x4000
    LI    x1, 0x90004000
    dma.load x6, x1, x2, ch4       # V1

    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4

    # ── Set unit scale (E8M0 2^0 = 1.0) for every VMATPOP.FP8 below ──
    seli e0, 0x7F

# ═══════════════════════════════════════════════════════════════
# VLOAD & QUANTIZE: Process all tiles into FP8 upfront
# Each VPACK.BF16.FP8 m<vd>, m<vs>, e0 has been replaced by:
#   VMATPUSH.ACC.BF16.MXU1 acc1, m<vs>   ; acc[1] = {m[vs], m[vs+1]}
#   VMATPOP.FP8.ACC.MXU1 m<vd>, acc1, e0      ; m[vd] = quantize(acc[1])
# ═══════════════════════════════════════════════════════════════

    # ── Load Q → m0..m3, pack to m4..m5 ──
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33
    vload m2, 16(x6)
    DELAY 33
    vload m3, 24(x6)
    DELAY 33
    vmatpush.acc.bf16.mxu1 acc1, m0
    DELAY 31
    vmatpop.fp8.acc.mxu1 m4, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu1 acc1, m2
    DELAY 31
    vmatpop.fp8.acc.mxu1 m5, acc1, e0
    DELAY 31

    # ── Load K0 → m16..m19, pack to m0..m1 ──
    LI    x6, 0x1000
    vload m16, 0(x6)
    DELAY 33
    vload m17, 8(x6)
    DELAY 33
    vload m18, 16(x6)
    DELAY 33
    vload m19, 24(x6)
    DELAY 33
    vmatpush.acc.bf16.mxu1 acc1, m16
    DELAY 31
    vmatpop.fp8.acc.mxu1 m0, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu1 acc1, m18
    DELAY 31
    vmatpop.fp8.acc.mxu1 m1, acc1, e0
    DELAY 31

    # ── Load V0 → m16..m19, pack to m2..m3 ──
    LI    x6, 0x3000
    vload m16, 0(x6)
    DELAY 33
    vload m17, 8(x6)
    DELAY 33
    vload m18, 16(x6)
    DELAY 33
    vload m19, 24(x6)
    DELAY 33
    vmatpush.acc.bf16.mxu1 acc1, m16
    DELAY 31
    vmatpop.fp8.acc.mxu1 m2, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu1 acc1, m18
    DELAY 31
    vmatpop.fp8.acc.mxu1 m3, acc1, e0
    DELAY 31

    # ── Load K1 → m16..m19, pack to m20..m21 ──
    LI    x6, 0x2000
    vload m16, 0(x6)
    DELAY 33
    vload m17, 8(x6)
    DELAY 33
    vload m18, 16(x6)
    DELAY 33
    vload m19, 24(x6)
    DELAY 33
    vmatpush.acc.bf16.mxu1 acc1, m16
    DELAY 31
    vmatpop.fp8.acc.mxu1 m20, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu1 acc1, m18
    DELAY 31
    vmatpop.fp8.acc.mxu1 m21, acc1, e0
    DELAY 31

    # ── Load V1 → m16..m19, pack to m22..m23 ──
    LI    x6, 0x4000
    vload m16, 0(x6)
    DELAY 33
    vload m17, 8(x6)
    DELAY 33
    vload m18, 16(x6)
    DELAY 33
    vload m19, 24(x6)
    DELAY 33
    vmatpush.acc.bf16.mxu1 acc1, m16
    DELAY 31
    vmatpop.fp8.acc.mxu1 m22, acc1, e0
    DELAY 31
    vmatpush.acc.bf16.mxu1 acc1, m18
    DELAY 31
    vmatpop.fp8.acc.mxu1 m23, acc1, e0
    DELAY 31

    # ── Initialize persistent state ──────────────────────────────────────
    vli.all m6, 0x3E00          # scale  = 0.125
    DELAY 63
    vli.all m8, 0xC2C8          # m_prev = -100.0
    DELAY 63
    vli.all m10, 0x0000          # l_prev = 0
    DELAY 63
    vli.all m12, 0x0000          # O_left = 0
    DELAY 63
    vli.all m14, 0x0000          # O_right = 0
    DELAY 63

# ═══════════════════════════════════════════════════════════════
# CYCLE COUNTER — start
# ═══════════════════════════════════════════════════════════════
    CSRRS x20, 0xC00, x0

    # ════════════════════════════════════════════════════════════════════
    # K TILE 0
    # ════════════════════════════════════════════════════════════════════

    # ── Q @ K0^T ────────────────────────────────────────────────────────
    vmatpush.weight.mxu1 w0, m0
    DELAY 31
    vmatmul.mxu1 acc0, m4, w0
    DELAY 34
    vmatpush.weight.mxu1 w0, m1
    DELAY 31
    vmatmul.acc.mxu1 acc0, m5, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m28, acc0      # m28,29 = scores
    DELAY 31

    # ── Scale scores ─────────────────────────────────────────────────────
    vmul.bf16 m26, m28, m6          # m26,27 = scaled
    DELAY 64

    # ── tile_max = rowmax(scaled) ────────────────────────────────────────
    vredmax.row.bf16 m16, m26      # m16,17 = tile_max
    DELAY 32

    # ── m_new = max(m_prev, tile_max) ───────────────────────────────────
    vmaximum.bf16 m18, m8, m16          # m18,19 = m_new
    DELAY 64

    # ── exp_diff = exp(m_prev - m_new) ──────────────────────────────────
    vsub.bf16 m24, m8, m18          # m24,25 = exp_diff
    DELAY 64
    vexp.bf16 m24, m24
    DELAY 64

    # ── O *= exp_diff ────────────────────────────────────────────────────
    vmul.bf16 m12, m12, m24
    DELAY 64
    vmul.bf16 m14, m14, m24
    DELAY 64

    # ── exp_s = exp(scaled - m_new) ─────────────────────────────────────
    vsub.bf16 m26, m26, m18         # m26,27 = exp_s (overwrites scaled)
    DELAY 64
    vexp.bf16 m26, m26
    DELAY 64

    # ── Quantize exp_s → FP8 (acc-roundtrip via buffer 1) ───────────────
    vmatpush.acc.bf16.mxu1 acc1, m26
    DELAY 31
    vmatpop.fp8.acc.mxu1 m30, acc1, e0    # m30 = exp_s_q
    DELAY 31

    # ── l = exp_diff * l + rowsum(exp_s) ────────────────────────────────
    vmul.bf16 m16, m24, m10         # m16,17 = exp_diff * l (overwrites tile_max)
    DELAY 64
    vredsum.row.bf16 m24, m26      # m24,25 = rowsum (overwrites exp_diff)
    DELAY 37
    vadd.bf16 m10, m16, m24         # l = new_l
    DELAY 64

    # ── m_prev = m_new ───────────────────────────────────────────────────
    vmov m8, m18
    DELAY 64

    # ── exp_s @ V0_left ──────────────────────────────────────────────────
    vmatpush.weight.mxu1 w0, m2
    DELAY 31
    vmatmul.mxu1 acc0, m30, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m26, acc0      # m26,27 = V_left contrib (overwrites exp_s)
    DELAY 31

    # ── exp_s @ V0_right ─────────────────────────────────────────────────
    vmatpush.weight.mxu1 w0, m3
    DELAY 31
    vmatmul.mxu1 acc0, m30, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m28, acc0      # m28,29 = V_right contrib (overwrites scores)
    DELAY 31

    # ── O += V contributions ─────────────────────────────────────────────
    vadd.bf16 m12, m12, m26
    DELAY 64
    vadd.bf16 m14, m14, m28
    DELAY 64

    # ════════════════════════════════════════════════════════════════════
    # K TILE 1
    # ════════════════════════════════════════════════════════════════════

    # ── Q @ K1^T ────────────────────────────────────────────────────────
    vmatpush.weight.mxu1 w0, m20
    DELAY 31
    vmatmul.mxu1 acc0, m4, w0
    DELAY 34
    vmatpush.weight.mxu1 w0, m21
    DELAY 31
    vmatmul.acc.mxu1 acc0, m5, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m28, acc0
    DELAY 31

    # ── Scale scores ─────────────────────────────────────────────────────
    vmul.bf16 m26, m28, m6
    DELAY 64

    # ── tile_max = rowmax(scaled) ────────────────────────────────────────
    vredmax.row.bf16 m16, m26
    DELAY 32

    # ── m_new = max(m_prev, tile_max) ───────────────────────────────────
    vmaximum.bf16 m18, m8, m16
    DELAY 64

    # ── exp_diff = exp(m_prev - m_new) ──────────────────────────────────
    vsub.bf16 m24, m8, m18
    DELAY 64
    vexp.bf16 m24, m24
    DELAY 64

    # ── O *= exp_diff ────────────────────────────────────────────────────
    vmul.bf16 m12, m12, m24
    DELAY 64
    vmul.bf16 m14, m14, m24
    DELAY 64

    # ── exp_s = exp(scaled - m_new) ─────────────────────────────────────
    vsub.bf16 m26, m26, m18
    DELAY 64
    vexp.bf16 m26, m26
    DELAY 64

    # ── Quantize exp_s → FP8 (acc-roundtrip via buffer 1) ───────────────
    vmatpush.acc.bf16.mxu1 acc1, m26
    DELAY 31
    vmatpop.fp8.acc.mxu1 m30, acc1, e0
    DELAY 31

    # ── l = exp_diff * l + rowsum(exp_s) ────────────────────────────────
    vmul.bf16 m16, m24, m10
    DELAY 64
    vredsum.row.bf16 m24, m26
    DELAY 37
    vadd.bf16 m10, m16, m24
    DELAY 64

    # ── m_prev = m_new ───────────────────────────────────────────────────
    vmov m8, m18
    DELAY 64

    # ── exp_s @ V1_left ──────────────────────────────────────────────────
    vmatpush.weight.mxu1 w0, m22
    DELAY 31
    vmatmul.mxu1 acc0, m30, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m26, acc0
    DELAY 31

    # ── exp_s @ V1_right ─────────────────────────────────────────────────
    vmatpush.weight.mxu1 w0, m23
    DELAY 31
    vmatmul.mxu1 acc0, m30, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m28, acc0
    DELAY 31

    # ── O += V contributions ─────────────────────────────────────────────
    vadd.bf16 m12, m12, m26
    DELAY 64
    vadd.bf16 m14, m14, m28
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

# ═══════════════════════════════════════════════════════════════
# CYCLE COUNTER — end
# ═══════════════════════════════════════════════════════════════
    CSRRS x21, 0xC00, x0
    SUB   x22, x21, x20         # Final active duration without pauses
    CSRRW x0,  0x801, x22       # Write out total cycles

# ═══════════════════════════════════════════════════════════════
# POST-STORE: Dump O to VMEM, then DMA to DRAM
# ═══════════════════════════════════════════════════════════════
    LI    x8, 0x1400
    vstore m12, 0(x8)
    DELAY 33
    vstore m13, 8(x8)
    DELAY 33
    vstore m14, 16(x8)
    DELAY 33
    vstore m15, 24(x8)
    DELAY 33

    LI    x8, 0x1400
    LI    x3, 0x90006000
    dma.store x3, x8, x2, ch0
    dma.wait ch0

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL
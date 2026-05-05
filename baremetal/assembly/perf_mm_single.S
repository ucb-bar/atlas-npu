# perf_mm_single.S — single-matmul per-MXU utilization characterization
# @TIMEOUT 200000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_perf_mm_single.py 500
# @PERF_UTIL_THRESHOLD 5
# @PERF_MACS_MXU0 32768
# @PERF_MACS_MXU1 32768
#
# Issues exactly one VMATMUL (overwrite mode — no acc-buffer dependency)
# to each of MXU0 and MXU1, back-to-back, and banks the mcycles delta
# across the timed window into CSR_DBG1 as the utilization denominator.
# Util is reported as MACs * 100 / (1024 * dbg1_cycles) per MXU, where
# 1024 = peak MAC/cycle (32x32 PEs) and each 32x32x32 VMATMUL = 32768 MACs.
#
# Expected utilization is inherently low here — a single VMATMUL covers
# only ~95 cycles of compute against a window that also includes the
# other MXU's matmul plus setup/halt overhead. The 5% threshold is a
# regression gate, not a performance target: util collapsing toward 0
# means a single-matmul command never streamed.
#
# Under the 1R1W MREG contract MXU0 and MXU1 cannot read the same bank
# in the same cycle, so the X activation tile is VLOAD'd into two MREGs
# — one per MXU — even though the data is identical.
#
# MREG bank map:
#   m0   X (MXU0 activation)
#   m1   X duplicate (MXU1 activation, distinct bank)
#   m2   W0 raw FP8
#   m3   W0 transposed (MXU0 weight slot 0)
#   m4   W1 raw FP8
#   m5   W1 transposed (MXU1 weight slot 0)
#
# DRAM layout (must match gen_perf_mm_single.py):
#   0x0000  X      (1024B FP8, 32x32)
#   0x0400  W0     (1024B FP8, 32x32)
#   0x0800  W1     (1024B FP8, 32x32)

    LI    x5,  0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127
    SELI  1, 127

# ═══════════════════════════════════════════════════════════════
# SETUP: Load activations X → m0, duplicate → m1
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x0000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# SETUP: MXU0 weights — load W0 → m2, transpose → m3, push wslot 0
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x0400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 2, x6, 0
    DELAY 40
    VTRPOSE.XLU 3, 2
    DELAY 80
    VMATPUSH.W.MXU0 0, 3
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# SETUP: MXU1 weights — load W1 → m4, transpose → m5, push wslot 0
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x0800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40
    VTRPOSE.XLU 5, 4
    DELAY 80
    VMATPUSH.W.MXU1 0, 5
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS — reset perf counters, snapshot mcycles
# ═══════════════════════════════════════════════════════════════
    CSRW  x0,  0xC20                # zero mxu0/mxu1 busy counters
    CSRR  x20, 0xC00                # x20 = cycle count at start

# ── One VMATMUL per MXU, overwrite mode. Issues on consecutive scalar
#    cycles (single-issue core). Each MXU's computeBusy stays high for
#    its full sequencer latency: aCompSetup (1) + aCompActive
#    (tileRows-1) + aCompDrain (pipeline tail). No overlap, no
#    accumulation — the psum read from the acc buffer is ignored. ──

    VMATMUL.MXU0  0, 0, 0     # accSel=0, vs1=m0, wSlot=0
    VMATMUL.MXU1  0, 1, 0     # accSel=0, vs1=m1, wSlot=0

    # Drain padding: SA 32x32 computeBusy ≈ 94 cycles, IPT
    # pipelineCuts={1} ≈ 33 cycles. 128 covers whichever topology is
    # wired to mxu0/mxu1 in AtlasCore, with margin.
    DELAY 128

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS — snapshot end cycle, bank delta in DBG1
# ═══════════════════════════════════════════════════════════════
    CSRR  x21, 0xC00                # x21 = cycle count at end
    SUB   x22, x21, x20             # x22 = end - start
    CSRW  x22, 0xC11                # stash delta in CSR_DBG1 (host reads @ 0x14)

# ═══════════════════════════════════════════════════════════════
# DONE — host reads busy0/busy1 from CSR_BASE + 0x1C/0x20 and the
#        mcycles delta from CSR_DBG1 @ 0x14
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

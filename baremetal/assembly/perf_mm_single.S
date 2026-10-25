# perf_mm_single.S — single-matmul MXU1 utilization characterization
# @TIMEOUT 200000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_perf_mm_single.py 500
# @PERF_UTIL_THRESHOLD 5
# @PERF_MACS_MXU1 32768
#
# Issues exactly one VMATMUL (overwrite mode — no acc-buffer dependency)
# to MXU1 and banks the mcycles delta across the timed window into
# CSR_DBG1 as the utilization denominator. Util is reported as
# MACs * 100 / (1024 * dbg1_cycles), where 1024 = peak MAC/cycle and each
# 32x32x32 VMATMUL = 32768 MACs.
#
# Expected utilization is inherently low here — a single VMATMUL covers
# only ~33 cycles of compute against a window that also includes the
# drain padding. The 5% threshold is a regression gate, not a performance
# target: util collapsing toward 0 means a single-matmul command never
# streamed.
#
# MREG bank map:
#   m0   X (MXU1 activation)
#   m2   W raw FP8
#   m3   W transposed (MXU1 weight slot 0)
#
# DRAM layout (must match gen_perf_mm_single.py):
#   0x0000  X      (1024B FP8, 32x32)
#   0x0400  W      (1024B FP8, 32x32)

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024
    seli e0, 127
    seli e1, 127

# ═══════════════════════════════════════════════════════════════
# SETUP: Load activations X → m0
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# SETUP: MXU1 weights — load W → m2, transpose → m3, push wslot 0
# ═══════════════════════════════════════════════════════════════

    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m2, 0(x6)
    DELAY 33
    vtrpose.xlu m3, m2
    DELAY 64
    vmatpush.weight.mxu1 w0, m3
    DELAY 31

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS — snapshot mcycles
# ═══════════════════════════════════════════════════════════════
    csrrs x20, 0xC00, x0                # x20 = cycle count at start

# ── One VMATMUL, overwrite mode. computeBusy stays high for the full
#    sequencer latency: aCompSetup (1) + aCompActive (tileRows-1) +
#    aCompDrain (pipeline tail). No accumulation — the psum read from
#    the acc buffer is ignored. ──

    vmatmul.mxu1 acc0, m0, w0     # accSel=0, vs1=m0, wSlot=0

    # Drain padding: IPT pipelineCuts={1} computeBusy ≈ 33 cycles.
    DELAY 34

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS — snapshot end cycle, bank delta in DBG1
# ═══════════════════════════════════════════════════════════════
    csrrs x21, 0xC00, x0                # x21 = cycle count at end
    SUB   x22, x21, x20             # x22 = end - start
    csrrw x0, 0xC11, x22                # stash delta in CSR_DBG1 (host reads @ 0x14)

# ═══════════════════════════════════════════════════════════════
# DONE — host reads the mcycles delta from CSR_DBG1 @ 0x14 and the
#        mxu1_busy counter (docs/perf-counters.md)
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    csrrw x0, 0xC10, x30
    ECALL

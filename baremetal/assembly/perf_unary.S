# perf_unary.S — maximum FU utilization: 100% paired slots
# @TIMEOUT 25000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_perf_unary.py
# @PERF_REPORT
#   Slot  Chain A (m0→m2)    Chain B (m4→m6)    Chain C (m10→m8)
#   ────  ─────────────────  ─────────────────  ──────────────────
#     1   VSQUARE            VTANH              —
#     2   VTANH              VEXP               —
#     3   VEXP               VSQRT              —
#     4   VSQRT              VLOG2              —
#     5   VLOG2              VRECIP             —
#     6   VRECIP             —(done)            VSQUARE
#     7   VSQUARE            —                  VTANH
#     8   VTANH              —                  VEXP
#
#       throughput per op = 32 (square/tanh) , 32 (tanh/exp) , 32 (exp/sqrt) , 32 (sqrt/log) , 32 (log/recip) , 32 (recip/squre) , 32 (square/tanh), 32 (tanh/exp)
#       computing for 2 32x32 so need 64 cycles
#     optimal cycle count = 64 + 64 + 64 + 64 + 64 + 64 + 64 + 64 = 512
# actual = 536 (without vload/vstore)
# utilization = 95.5%
# actual = 960 (with vload/vstore)
# utilization = 53.3%

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x12, 2048

# ─── Load A_raw → m0,m1 ────────────────────────────────────────
    LI    x6, 0x20000000
    LI    x9, 0x20000400 
    LI    x1, 0x90000000
    DMA.LOAD  x6, x1, x12, 0
    LI    x1, 0x90000800
    DMA.LOAD  x9, x1, x12, 1
    DMA.WAIT  0
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS
# ═══════════════════════════════════════════════════════════════
    CSRRS x20, 0xC00, x0        # x20 = cycle count at start

    VLOAD    0, x6, 0           # m0 ← A low  bank
    DELAY 33
    VLOAD    1, x6, 8           # m1 ← A high bank
    DELAY 33

# ─── Load B_raw → m4,m5 (also used as C_raw) ───────────────────
    

    VLOAD    4, x9, 0           # m4 ← B low  bank
    DELAY 33
    VLOAD    5, x9, 8           # m5 ← B high bank
    DELAY 33

    VLOAD    10, x9, 0          # m10 ← B low  bank (reload for chain C)
    DELAY 33
    VLOAD    11, x9, 8          # m11 ← B high bank
    DELAY 33

# ── Slot 1: VSQUARE(A,m0→m2)  ∥  VTANH(B,m4→m6) ────────────────────────
#   SquareCubeVec ∥ TanhRec — different FU groups ✓
#   A: square(A_raw) → m2,m3     reads m0,m1 (setup) ✓
#   B: tanh(tanh(B_raw)) → m6,m7 reads m4,m5 (primed in setup) ✓
#   Writing to disjoint register pairs (m2 vs m6) — no WAW ✓
    VSQUARE.BF16  2,  0         # m2,m3  = A²                     (65 cy)
    VTANH    6,  4         # m6,m7  = tanh(tanh(B_raw))      (65 cy)
    DELAY 64

# ── Slot 2: VTANH(A,m2)  ∥  VEXP(B,m6) ─────────────────────────────────
#   TanhRec ∥ Exp — different FU groups ✓
#   A: tanh(A²) → m2,m3          reads m2,m3 from slot 1 ✓
#   B: exp(tanh²(B)) → m6,m7     reads m6,m7 from slot 1 ✓
    VTANH    2,  2         # m2,m3  = tanh(A²)               (65 cy)
    VEXP     6,  6         # m6,m7  = exp(tanh(tanh(B)))     (65 cy)
    DELAY 64

# ── Slot 3: VEXP(A,m2)  ∥  VSQRT(B,m6) ─────────────────────────────────
#   Exp ∥ Sqrt — different FU groups ✓
#   A: exp(tanh(A²)) → m2,m3     reads m2,m3 from slot 2 ✓
#   B: sqrt(exp(...B)) → m6,m7   reads m6,m7 from slot 2 ✓
    VEXP     2,  2         # m2,m3  = exp(tanh(A²))          (65 cy)
    VSQRT    6,  6         # m6,m7  = sqrt(exp(...B))        (65 cy)
    DELAY 64

# ── Slot 4: VSQRT(A,m2)  ∥  VLOG2(B,m6) ────────────────────────────────
#   Sqrt ∥ Log — different FU groups ✓
#   A: sqrt(exp(tanh(A²))) → m2,m3   reads m2,m3 from slot 3 ✓
#   B: log2(sqrt(...B)) → m6,m7      reads m6,m7 from slot 3 ✓
    VSQRT    2,  2         # m2,m3  = sqrt(exp(tanh(A²)))    (65 cy)
    VLOG2    6,  6         # m6,m7  = log2(sqrt(...B))       (65 cy)
    DELAY 64

# ── Slot 5: VLOG2(A,m2)  ∥  VRECIP(B,m6) ───────────────────────────────
#   Log ∥ Rcp — different FU groups ✓
#   A: log2(sqrt(...A)) → m2,m3  reads m2,m3 from slot 4 ✓
#   B: recip(log2(...B)) → m6,m7 reads m6,m7 from slot 4 ✓
#   After this slot: chain B is COMPLETE in m6,m7 (Y_b output)
    VLOG2    2,  2         # m2,m3  = log2(sqrt(...A))       (65 cy)
    VRECIP.BF16   6,  6         # m6,m7  = recip(log2(...B))=Y_b  (65 cy)
    DELAY 64

# ── Slot 6: VRECIP(A,m2)  ∥  VSQUARE(C,m10→m8) ─────────────────────────
#   Rcp ∥ SquareCubeVec — different FU groups ✓
#   A: recip(log2(...A)) → m2,m3  reads m2,m3 from slot 5 ✓
#   C: square(tanh(B²)) → m8,m9  reads m10,m11 (primed in setup) ✓
#   Chain B done; chain C starts its timed-region run here.
#   Writing to m2 (A) vs m8 (C) — no WAW ✓
    VRECIP.BF16   2,  2         # m2,m3  = recip(log2(...A))      (65 cy)
    VSQUARE.BF16  8,  10        # m8,m9  = square(tanh(B²))       (65 cy)
    DELAY 64

# ── Slot 7: VSQUARE(A,m2)  ∥  VTANH(C,m8) ──────────────────────────────
#   SquareCubeVec ∥ TanhRec — different FU groups ✓
#   A: square(recip(...A)) → m2,m3   reads m2,m3 from slot 6 ✓
#   C: tanh(square(...C)) → m8,m9    reads m8,m9 from slot 6 ✓
    VSQUARE.BF16  2,  2         # m2,m3  = square(recip(...A))    (65 cy)
    VTANH    8,  8         # m8,m9  = tanh(square(...C))     (65 cy)
    DELAY 64

# ── Slot 8: VTANH(A,m2)  ∥  VEXP(C,m8) ─────────────────────────────────
#   TanhRec ∥ Exp — different FU groups ✓
#   A: tanh(square(recip(...A))) → m2,m3  reads m2,m3 from slot 7 ✓
#   C: exp(tanh(square(...C))) → m8,m9    reads m8,m9 from slot 7 ✓
#   After this slot: chain A complete in m2,m3 (Y_a); chain C complete in m8,m9 (Y_c)
    VTANH    2,  2         # m2,m3  = tanh(square(...A))=Y_a (65 cy)
    VEXP     8,  8         # m8,m9  = exp(tanh(...C))=Y_c    (65 cy)
    DELAY 64

# ═══════════════════════════════════════════════════════════════
# WRITEBACK (not timed)
# ═══════════════════════════════════════════════════════════════

    # Y_a (m2,m3) → DRAM 0x1000
    LI    x6, 0x20000000
    LI    x7, 0x20001000        
    LI    x8, 0x20001800 

    VSTORE 2, x6, 0
    DELAY 33
    VSTORE 3, x6, 8
    DELAY 33

    VSTORE 6, x7, 0
    DELAY 33
    VSTORE 7, x7, 8
    DELAY 33

    VSTORE 8, x8, 0
    DELAY 33
    VSTORE 9, x8, 8
    DELAY 33

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS — 520 cycles
# ═══════════════════════════════════════════════════════════════
    CSRRS x21, 0xC00, x0        # x21 = cycle count at end
    SUB   x22, x21, x20         # x22 = elapsed cycles
    CSRRW x0,  0xC11, x22       # stash in CSR_DBG1

    LI    x3, 0x90001000
    DMA.STORE x3, x6, x12, 1
    DMA.WAIT  1

    # Y_b (m6,m7) → DRAM 0x1800
    LI    x3, 0x90001800
    DMA.STORE x3, x7, x12, 1
    DMA.WAIT  1

    # Y_c (m8,m9) → DRAM 0x20000
    LI    x3, 0x90002000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL
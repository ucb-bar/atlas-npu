# perf_mm_dual_128x128x128.S — dual-MXU (M=N=K=128) FP8 matmul perf test
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_perf_mm_dual_128x128x128.py 500
# @PERF_REPORT
# @PERF_MACS_MXU0 1048576
# @PERF_MACS_MXU1 1048576
#
# C = A·B, 128x128 = 4x4 grid of 32x32 FP8 tiles. C[i,j] = sum_k A[i,k]·B[k,j].
# MXU0 pinned to top 2 rows; MXU1 to bottom. Sub-phases rotate B-col pair.
# No K-skew needed → vanilla goldens, stride-32 cadence.
# Measured: 1099 cyc, 93% util_mxu0, 93% util_mxu1.
#
# ── Spatial layout — 4×4 tile grid ──────────────────────────────────────
#
#                   ┌── B-col j=0 ──┬── B-col j=1 ──┬── B-col j=2 ──┬── B-col j=3 ──┐
#                   │   (rows 0-31  │  rows 32-63   │  rows 64-95   │  rows 96-127) │
#         ┌─────────┼───────────────┼───────────────┼───────────────┼───────────────┤
#         │ A-row   │               │               │               │               │
#         │ i=0     │    C[0,0]     │    C[0,1]     │    C[0,2]     │    C[0,3]     │
#         │ rows    │               │               │               │               │
#         │ 0-31    │               │               │               │               │
#         ├─────────┼───────────────┼───────────────┼───────────────┼───────────────┤
#         │ A-row   │               │               │               │               │
#         │ i=1     │    C[1,0]     │    C[1,1]     │    C[1,2]     │    C[1,3]     │
#         │ rows    │               │               │               │               │
#         │ 32-63   │               │               │               │               │
#         ├─────────┼───────────────┼───────────────┼───────────────┼───────────────┤
#         │ A-row   │               │               │               │               │
#         │ i=2     │    C[2,0]     │    C[2,1]     │    C[2,2]     │    C[2,3]     │
#         │ rows    │               │               │               │               │
#         │ 64-95   │               │               │               │               │
#         ├─────────┼───────────────┼───────────────┼───────────────┼───────────────┤
#         │ A-row   │               │               │               │               │
#         │ i=3     │    C[3,0]     │    C[3,1]     │    C[3,2]     │    C[3,3]     │
#         │ rows    │               │               │               │               │
#         │ 96-127  │               │               │               │               │
#         └─────────┴───────────────┴───────────────┴───────────────┴───────────────┘
#
# ── Ownership — row-pinned, column-rotated ─────────────────────────────
#
#             B-cols 0..1                   B-cols 2..3
#            (sub-α MXU0/                  (sub-β MXU0/
#             sub-β MXU1)                   sub-α MXU1)
#        ┌────────────────────┬────────────────────┐
# A rows │                    │                    │
# 0..1   │       MXU0 - α     │       MXU0 - β     │
# (MXU0) │   C[0,0]   C[0,1]  │   C[0,2]   C[0,3]  │
#        │   C[1,0]   C[1,1]  │   C[1,2]   C[1,3]  │
#        │   (TL)             │   (TR)             │
#        ├────────────────────┼────────────────────┤
# A rows │                    │                    │
# 2..3   │       MXU1 - β     │       MXU1 - α     │
# (MXU1) │   C[2,0]   C[2,1]  │   C[2,2]   C[2,3]  │
#        │   C[3,0]   C[3,1]  │   C[3,2]   C[3,3]  │
#        │   (BL)             │   (BR)             │
#        └────────────────────┴────────────────────┘
#
# ── Wall-clock timeline ─────────────────────────────────────────────────
#
#    cyc   0 ─────────────── 547 ────────────── ~1095
#          │                  │                   │
#    MXU0  │═══ sub-α: TL ════│═══ sub-β: TR ═════│
#          │  C[0..1, 0..1]   │  C[0..1, 2..3]    │
#          │  reads B[*,0..1] │  reads B[*,2..3]  │
#          │                  │                   │
#    MXU1  │═══ sub-α: BR ════│═══ sub-β: BL ═════│
#          │  C[2..3, 2..3]   │  C[2..3, 0..1]    │
#          │  reads B[*,2..3] │  reads B[*,0..1]  │
#          │                  │                   │
#         CSRR x20         seam              CSRR x21
#                        (≤5 cyc)
#
# ── Per-tile K-accumulation (inside one sub-phase) ──────────────────────
#
#    MXU0 sub-α — 16 matmuls + 4 pops, ~547 cyc:
#
#    ── K=0 ──   ── K=1 ──   ── K=2 ──   ── K=3 ──     pop
#    ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
#    │ A·B00 │ + │ A·B10 │ + │ A·B20 │ + │ A·B30 │ → C[0,0] (acc0) → m32 (b0 hi)
#    │ A·B01 │ + │ A·B11 │ + │ A·B21 │ + │ A·B31 │ → C[0,1] (acc1) → m33 (b1 hi)
#    └───────┘   └───────┘   └───────┘   └───────┘
#         epoch 1: A row = A[0,*]    (8 matmuls)
#
#    ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
#    │ A·B00 │ + │ A·B10 │ + │ A·B20 │ + │ A·B30 │ → C[1,0] (acc0) → m36 (b4 hi)
#    │ A·B01 │ + │ A·B11 │ + │ A·B21 │ + │ A·B31 │ → C[1,1] (acc1) → m37 (b5 hi)
#    └───────┘   └───────┘   └───────┘   └───────┘
#         epoch 2: A row = A[1,*]    (8 matmuls)
#
# ── Bank disjointness (no K-skew needed) ────────────────────────────────
#   sub-α: MXU0 = {b0..b7, B[*,0..1]};  MXU1 = {b8..b15, B[*,2..3]}
#   sub-β: MXU0 = {b0..b7, B[*,2..3]};  MXU1 = {b8..b15, B[*,0..1]}
# A-bank ownership pinned for whole test; B-col ownership swaps once at α→β.
#
# ── MREG allocation ─────────────────────────────────────────────────────
#   A tiles (banks 0..15, low half):       Pops m_{32+4i+j} (banks 0..15, high):
#     m0  A[0,0]   m4  A[1,0]                m32 C[0,0]   m36 C[1,0]
#     m1  A[0,1]   m5  A[1,1]                m33 C[0,1]   m37 C[1,1]
#     m2  A[0,2]   m6  A[1,2]                m34 C[0,2]   m38 C[1,2]
#     m3  A[0,3]   m7  A[1,3]                m35 C[0,3]   m39 C[1,3]
#     m8  A[2,0]   m12 A[3,0]                m40 C[2,0]   m44 C[3,0]
#     m9  A[2,1]   m13 A[3,1]                m41 C[2,1]   m45 C[3,1]
#     m10 A[2,2]   m14 A[3,2]                m42 C[2,2]   m46 C[3,2]
#     m11 A[2,3]   m15 A[3,3]                m43 C[2,3]   m47 C[3,3]
#
#   B^T tiles m_{16+4k+j} (banks 16..31, low half):
#     m16 B[0,0]^T  m20 B[1,0]^T  m24 B[2,0]^T  m28 B[3,0]^T
#     m17 B[0,1]^T  m21 B[1,1]^T  m25 B[2,1]^T  m29 B[3,1]^T
#     m18 B[0,2]^T  m22 B[1,2]^T  m26 B[2,2]^T  m30 B[3,2]^T
#     m19 B[0,3]^T  m23 B[1,3]^T  m27 B[2,3]^T  m31 B[3,3]^T
#
# Pop W (high half, phys row 32+r) coexists with A R (low half, phys row r)
# on the same bank: 1R+1W ports, independent rows (MregFile.scala:1-22).
#
# ── DRAM layout ─────────────────────────────────────────────────────────
#   A   @ 0x0000 + (4i + k) * 0x400   →   0x0000..0x3C00
#   B^T @ 0x4000 + (4k + j) * 0x400   →   0x4000..0x7C00
#   C   @ 0x8000 + (4i + j) * 0x400   →   0x8000..0xBC00

    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    SELI  0, 127                    # scale reg 0 = 0x7F -> unit scale

# ═══════════════════════════════════════════════════════════════
# SETUP (untimed): load 16 A tiles → m0..m15, 16 B^T tiles → m16..m31
# ═══════════════════════════════════════════════════════════════

    # A[0,0] -> m0
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    # A[0,1] -> m1
    ADDI  x6, x0, 0
    LI    x10, 0x90000400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 1, x6, 0
    DELAY 40

    # A[0,2] -> m2
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 2, x6, 0
    DELAY 40

    # A[0,3] -> m3
    ADDI  x6, x0, 0
    LI    x10, 0x90000C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 3, x6, 0
    DELAY 40

    # A[1,0] -> m4
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    DELAY 40

    # A[1,1] -> m5
    ADDI  x6, x0, 0
    LI    x10, 0x90001400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 5, x6, 0
    DELAY 40

    # A[1,2] -> m6
    ADDI  x6, x0, 0
    LI    x10, 0x90001800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    DELAY 40

    # A[1,3] -> m7
    ADDI  x6, x0, 0
    LI    x10, 0x90001C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 7, x6, 0
    DELAY 40

    # A[2,0] -> m8
    ADDI  x6, x0, 0
    LI    x10, 0x90002000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 8, x6, 0
    DELAY 40

    # A[2,1] -> m9
    ADDI  x6, x0, 0
    LI    x10, 0x90002400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 9, x6, 0
    DELAY 40

    # A[2,2] -> m10
    ADDI  x6, x0, 0
    LI    x10, 0x90002800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 10, x6, 0
    DELAY 40

    # A[2,3] -> m11
    ADDI  x6, x0, 0
    LI    x10, 0x90002C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 11, x6, 0
    DELAY 40

    # A[3,0] -> m12
    ADDI  x6, x0, 0
    LI    x10, 0x90003000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 12, x6, 0
    DELAY 40

    # A[3,1] -> m13
    ADDI  x6, x0, 0
    LI    x10, 0x90003400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 13, x6, 0
    DELAY 40

    # A[3,2] -> m14
    ADDI  x6, x0, 0
    LI    x10, 0x90003800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 14, x6, 0
    DELAY 40

    # A[3,3] -> m15
    ADDI  x6, x0, 0
    LI    x10, 0x90003C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 15, x6, 0
    DELAY 40

    # B[0,0]^T -> m16
    ADDI  x6, x0, 0
    LI    x10, 0x90004000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 16, x6, 0
    DELAY 40

    # B[0,1]^T -> m17
    ADDI  x6, x0, 0
    LI    x10, 0x90004400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 17, x6, 0
    DELAY 40

    # B[0,2]^T -> m18
    ADDI  x6, x0, 0
    LI    x10, 0x90004800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 18, x6, 0
    DELAY 40

    # B[0,3]^T -> m19
    ADDI  x6, x0, 0
    LI    x10, 0x90004C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 19, x6, 0
    DELAY 40

    # B[1,0]^T -> m20
    ADDI  x6, x0, 0
    LI    x10, 0x90005000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 20, x6, 0
    DELAY 40

    # B[1,1]^T -> m21
    ADDI  x6, x0, 0
    LI    x10, 0x90005400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 21, x6, 0
    DELAY 40

    # B[1,2]^T -> m22
    ADDI  x6, x0, 0
    LI    x10, 0x90005800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 22, x6, 0
    DELAY 40

    # B[1,3]^T -> m23
    ADDI  x6, x0, 0
    LI    x10, 0x90005C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 23, x6, 0
    DELAY 40

    # B[2,0]^T -> m24
    ADDI  x6, x0, 0
    LI    x10, 0x90006000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 24, x6, 0
    DELAY 40

    # B[2,1]^T -> m25
    ADDI  x6, x0, 0
    LI    x10, 0x90006400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 25, x6, 0
    DELAY 40

    # B[2,2]^T -> m26
    ADDI  x6, x0, 0
    LI    x10, 0x90006800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 26, x6, 0
    DELAY 40

    # B[2,3]^T -> m27
    ADDI  x6, x0, 0
    LI    x10, 0x90006C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 27, x6, 0
    DELAY 40

    # B[3,0]^T -> m28
    ADDI  x6, x0, 0
    LI    x10, 0x90007000
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 28, x6, 0
    DELAY 40

    # B[3,1]^T -> m29
    ADDI  x6, x0, 0
    LI    x10, 0x90007400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 29, x6, 0
    DELAY 40

    # B[3,2]^T -> m30
    ADDI  x6, x0, 0
    LI    x10, 0x90007800
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 30, x6, 0
    DELAY 40

    # B[3,3]^T -> m31
    ADDI  x6, x0, 0
    LI    x10, 0x90007C00
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 31, x6, 0
    DELAY 40

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION BEGINS — sub-α  →  ~2-cyc seam  →  sub-β
# MXU0 instructions track perf_mm_mxu0_64x64x128.S cyc;
# MXU1 instructions track perf_mm_mxu1_64x64x128.S cyc + 3.
# ═══════════════════════════════════════════════════════════════
    CSRR  x20, 0xC00                # cyc 0

# ─── Sub-α: MXU0 = TL (C[0..1, 0..1]) ‖ MXU1 = BR (C[2..3, 2..3]) ───────
# MXU0 pops → m32,m33,m36,m37. MXU1 pops → m42,m43,m46,m47.

    VMATPUSH.W.MXU0   0, 16         # cyc 1:  MXU0 wslot0 <- B[0,0]^T
    VMATMUL.MXU0      0, 0, 0       # cyc 2:  MXU0 M0_a   acc0 = A[0,0]*B[0,0]   (ow)
    DELAY 0                         # cyc 3   (1 cyc spacer)
    VMATPUSH.W.MXU1   0, 18         # cyc 4:  MXU1 wslot0 <- B[0,2]^T
    DELAY 27                        # cyc 5..32

    VMATPUSH.W.MXU0   1, 17         # cyc 33: MXU0 wslot1 <- B[0,1]^T
    VMATMUL.MXU0      1, 0, 1       # cyc 34: MXU0 M1_a   acc1 = A[0,0]*B[0,1]   (ow)
    DELAY 0                         # cyc 35
    VMATMUL.MXU1      0, 8, 0       # cyc 36: MXU1 M0_a   acc0 = A[2,0]*B[0,2]   (ow) [push@4 -> matmul@36, gap=32]
    VMATPUSH.W.MXU1   1, 22         # cyc 37: MXU1 wslot1 <- B[1,2]^T
    DELAY 26                        # cyc 38..64

    VMATPUSH.W.MXU0   0, 20         # cyc 65: MXU0 wslot0 <- B[1,0]^T
    VMATMUL.ACC.MXU0  0, 1, 0       # cyc 66: MXU0 M2_a   acc0 += A[0,1]*B[1,0]
    DELAY 0                         # cyc 67
    VMATMUL.MXU1      1, 12, 0      # cyc 68: MXU1 M1_a   acc1 = A[3,0]*B[0,2]   (ow, same wslot)
    DELAY 27                        # cyc 69..96

    VMATPUSH.W.MXU0   1, 21         # cyc 97: MXU0 wslot1 <- B[1,1]^T
    VMATMUL.ACC.MXU0  1, 1, 1       # cyc 98: MXU0 M3_a   acc1 += A[0,1]*B[1,1]
    DELAY 0                         # cyc 99
    VMATMUL.ACC.MXU1  0, 9, 1       # cyc 100: MXU1 M2_a   acc0 += A[2,1]*B[1,2]
    VMATPUSH.W.MXU1   0, 26         # cyc 101: MXU1 wslot0 <- B[2,2]^T
    DELAY 26                        # cyc 102..128

    VMATPUSH.W.MXU0   0, 24         # cyc 129: MXU0 wslot0 <- B[2,0]^T
    VMATMUL.ACC.MXU0  0, 2, 0       # cyc 130: MXU0 M4_a   acc0 += A[0,2]*B[2,0]
    DELAY 0                         # cyc 131
    VMATMUL.ACC.MXU1  1, 13, 1      # cyc 132: MXU1 M3_a   acc1 += A[3,1]*B[1,2]
    DELAY 27                        # cyc 133..160

    VMATPUSH.W.MXU0   1, 25         # cyc 161: MXU0 wslot1 <- B[2,1]^T
    VMATMUL.ACC.MXU0  1, 2, 1       # cyc 162: MXU0 M5_a   acc1 += A[0,2]*B[2,1]
    DELAY 0                         # cyc 163
    VMATMUL.ACC.MXU1  0, 10, 0      # cyc 164: MXU1 M4_a   acc0 += A[2,2]*B[2,2]
    VMATPUSH.W.MXU1   1, 30         # cyc 165: MXU1 wslot1 <- B[3,2]^T
    DELAY 26                        # cyc 166..192

    VMATPUSH.W.MXU0   0, 28         # cyc 193: MXU0 wslot0 <- B[3,0]^T
    VMATMUL.ACC.MXU0  0, 3, 0       # cyc 194: MXU0 M6_a   acc0 += A[0,3]*B[3,0]
    DELAY 0                         # cyc 195
    VMATMUL.ACC.MXU1  1, 14, 0      # cyc 196: MXU1 M5_a   acc1 += A[3,2]*B[2,2]
    DELAY 27                        # cyc 197..224

    VMATPUSH.W.MXU0   1, 29         # cyc 225: MXU0 wslot1 <- B[3,1]^T
    VMATMUL.ACC.MXU0  1, 3, 1       # cyc 226: MXU0 M7_a   acc1 += A[0,3]*B[3,1]
    DELAY 0                         # cyc 227
    VMATMUL.ACC.MXU1  0, 11, 1      # cyc 228: MXU1 M6_a   acc0 += A[2,3]*B[3,2]
    VMATPUSH.W.MXU1   0, 19         # cyc 229: MXU1 wslot0 <- B[0,3]^T (Phase B start for MXU1)
    DELAY 26                        # cyc 230..256

    # ── Phase A->B internal seam (within sub-alpha) — anchored verbatim ──
    VMATPUSH.W.MXU0   0, 16         # cyc 257: MXU0 wslot0 <- B[0,0]^T (Phase B start for MXU0)
    VMATPOP.FP8.MXU0  32, 0, 0      # cyc 258: m32 <- C[0,0] (acc0)
    VMATMUL.MXU0      0, 4, 0       # cyc 259: MXU0 M8_a   acc0 = A[1,0]*B[0,0]   (ow)
    VMATMUL.ACC.MXU1  1, 15, 1      # cyc 260: MXU1 M7_a   acc1 += A[3,3]*B[3,2]
    VMATPOP.FP8.MXU1  42, 0, 0      # cyc 261: m42 <- C[2,2] (acc0)
    DELAY 26                        # cyc 262..288

    VMATPUSH.W.MXU0   1, 17         # cyc 289: MXU0 wslot1 <- B[0,1]^T
    VMATPOP.FP8.MXU0  33, 0, 1      # cyc 290: m33 <- C[0,1] (acc1)
    VMATMUL.MXU0      1, 4, 1       # cyc 291: MXU0 M9_a   acc1 = A[1,0]*B[0,1]   (ow)
    VMATMUL.MXU1      0, 8, 0       # cyc 292: MXU1 M8_a   acc0 = A[2,0]*B[0,3]   (ow)
    VMATPUSH.W.MXU1   1, 23         # cyc 293: MXU1 wslot1 <- B[1,3]^T
    VMATPOP.FP8.MXU1  46, 0, 1      # cyc 294: m46 <- C[3,2] (acc1)
    DELAY 26                        # cyc 295..321

    VMATPUSH.W.MXU0   0, 20         # cyc 322: MXU0 wslot0 <- B[1,0]^T
    VMATMUL.ACC.MXU0  0, 5, 0       # cyc 323: MXU0 M10_a  acc0 += A[1,1]*B[1,0]
    VMATMUL.MXU1      1, 12, 0      # cyc 324: MXU1 M9_a   acc1 = A[3,0]*B[0,3]   (ow)
    DELAY 28                        # cyc 325..353

    VMATPUSH.W.MXU0   1, 21         # cyc 354: MXU0 wslot1 <- B[1,1]^T
    VMATMUL.ACC.MXU0  1, 5, 1       # cyc 355: MXU0 M11_a  acc1 += A[1,1]*B[1,1]
    VMATMUL.ACC.MXU1  0, 9, 1       # cyc 356: MXU1 M10_a  acc0 += A[2,1]*B[1,3]
    VMATPUSH.W.MXU1   0, 27         # cyc 357: MXU1 wslot0 <- B[2,3]^T
    DELAY 27                        # cyc 358..385

    VMATPUSH.W.MXU0   0, 24         # cyc 386: MXU0 wslot0 <- B[2,0]^T
    VMATMUL.ACC.MXU0  0, 6, 0       # cyc 387: MXU0 M12_a  acc0 += A[1,2]*B[2,0]
    VMATMUL.ACC.MXU1  1, 13, 1      # cyc 388: MXU1 M11_a  acc1 += A[3,1]*B[1,3]
    DELAY 28                        # cyc 389..417

    VMATPUSH.W.MXU0   1, 25         # cyc 418: MXU0 wslot1 <- B[2,1]^T
    VMATMUL.ACC.MXU0  1, 6, 1       # cyc 419: MXU0 M13_a  acc1 += A[1,2]*B[2,1]
    VMATMUL.ACC.MXU1  0, 10, 0      # cyc 420: MXU1 M12_a  acc0 += A[2,2]*B[2,3]
    VMATPUSH.W.MXU1   1, 31         # cyc 421: MXU1 wslot1 <- B[3,3]^T
    DELAY 27                        # cyc 422..449

    VMATPUSH.W.MXU0   0, 28         # cyc 450: MXU0 wslot0 <- B[3,0]^T
    VMATMUL.ACC.MXU0  0, 7, 0       # cyc 451: MXU0 M14_a  acc0 += A[1,3]*B[3,0]
    VMATMUL.ACC.MXU1  1, 14, 0      # cyc 452: MXU1 M13_a  acc1 += A[3,2]*B[2,3]
    DELAY 28                        # cyc 453..481

    VMATPUSH.W.MXU0   1, 29         # cyc 482: MXU0 wslot1 <- B[3,1]^T
    VMATMUL.ACC.MXU0  1, 7, 1       # cyc 483: MXU0 M15_a  acc1 += A[1,3]*B[3,1]
    VMATMUL.ACC.MXU1  0, 11, 1      # cyc 484: MXU1 M14_a  acc0 += A[2,3]*B[3,3]
    DELAY 29                        # cyc 485..514

    VMATPOP.FP8.MXU0  36, 0, 0      # cyc 515: m36 <- C[1,0] (acc0)
    VMATMUL.ACC.MXU1  1, 15, 1      # cyc 516: MXU1 M15_a  acc1 += A[3,3]*B[3,3]
    VMATPOP.FP8.MXU1  43, 0, 0      # cyc 517: m43 <- C[2,3] (acc0)
    DELAY 28                        # cyc 518..546

    VMATPOP.FP8.MXU0  37, 0, 1      # cyc 547: m37 <- C[1,1] (acc1)
    DELAY 0                         # cyc 548 (seam spacer)
    VMATPOP.FP8.MXU1  47, 0, 1      # cyc 549: m47 <- C[3,3] (acc1)

# ─── Sub-alpha -> sub-beta seam (cyc 550) ───
    DELAY 0                         # cyc 550 (seam spacer; first sub-beta PUSH at 551)

# ─── Sub-β: MXU0 = TR (C[0..1, 2..3]) ‖ MXU1 = BL (C[2..3, 0..1]) ───────
# Cycle offsets: MXU0 at anchor_cyc + 550; MXU1 at anchor_cyc + 553.
# Seam timing: MXU0 wbuf-drain (cyc 483 + 63 = 546 ≤ 551 ✓);
#              MXU1 wslots idle by cyc 548 < 554 ✓.

    VMATPUSH.W.MXU0   0, 18         # cyc 551: MXU0 wslot0 <- B[0,2]^T
    VMATMUL.MXU0      0, 0, 0       # cyc 552: MXU0 M0_b   acc0 = A[0,0]*B[0,2]   (ow)
    DELAY 0                         # cyc 553
    VMATPUSH.W.MXU1   0, 16         # cyc 554: MXU1 wslot0 <- B[0,0]^T
    DELAY 27                        # cyc 555..582

    VMATPUSH.W.MXU0   1, 19         # cyc 583: MXU0 wslot1 <- B[0,3]^T
    VMATMUL.MXU0      1, 0, 1       # cyc 584: MXU0 M1_b   acc1 = A[0,0]*B[0,3]   (ow)
    DELAY 0                         # cyc 585
    VMATMUL.MXU1      0, 8, 0       # cyc 586: MXU1 M0_b   acc0 = A[2,0]*B[0,0]   (ow) [push@554 -> matmul@586, gap=32]
    VMATPUSH.W.MXU1   1, 20         # cyc 587: MXU1 wslot1 <- B[1,0]^T
    DELAY 26                        # cyc 588..614

    VMATPUSH.W.MXU0   0, 22         # cyc 615: MXU0 wslot0 <- B[1,2]^T
    VMATMUL.ACC.MXU0  0, 1, 0       # cyc 616: MXU0 M2_b   acc0 += A[0,1]*B[1,2]
    DELAY 0                         # cyc 617
    VMATMUL.MXU1      1, 12, 0      # cyc 618: MXU1 M1_b   acc1 = A[3,0]*B[0,0]   (ow, same wslot)
    DELAY 27                        # cyc 619..646

    VMATPUSH.W.MXU0   1, 23         # cyc 647: MXU0 wslot1 <- B[1,3]^T
    VMATMUL.ACC.MXU0  1, 1, 1       # cyc 648: MXU0 M3_b   acc1 += A[0,1]*B[1,3]
    DELAY 0                         # cyc 649
    VMATMUL.ACC.MXU1  0, 9, 1       # cyc 650: MXU1 M2_b   acc0 += A[2,1]*B[1,0]
    VMATPUSH.W.MXU1   0, 24         # cyc 651: MXU1 wslot0 <- B[2,0]^T
    DELAY 26                        # cyc 652..678

    VMATPUSH.W.MXU0   0, 26         # cyc 679: MXU0 wslot0 <- B[2,2]^T
    VMATMUL.ACC.MXU0  0, 2, 0       # cyc 680: MXU0 M4_b   acc0 += A[0,2]*B[2,2]
    DELAY 0                         # cyc 681
    VMATMUL.ACC.MXU1  1, 13, 1      # cyc 682: MXU1 M3_b   acc1 += A[3,1]*B[1,0]
    DELAY 27                        # cyc 683..710

    VMATPUSH.W.MXU0   1, 27         # cyc 711: MXU0 wslot1 <- B[2,3]^T
    VMATMUL.ACC.MXU0  1, 2, 1       # cyc 712: MXU0 M5_b   acc1 += A[0,2]*B[2,3]
    DELAY 0                         # cyc 713
    VMATMUL.ACC.MXU1  0, 10, 0      # cyc 714: MXU1 M4_b   acc0 += A[2,2]*B[2,0]
    VMATPUSH.W.MXU1   1, 28         # cyc 715: MXU1 wslot1 <- B[3,0]^T
    DELAY 26                        # cyc 716..742

    VMATPUSH.W.MXU0   0, 30         # cyc 743: MXU0 wslot0 <- B[3,2]^T
    VMATMUL.ACC.MXU0  0, 3, 0       # cyc 744: MXU0 M6_b   acc0 += A[0,3]*B[3,2]
    DELAY 0                         # cyc 745
    VMATMUL.ACC.MXU1  1, 14, 0      # cyc 746: MXU1 M5_b   acc1 += A[3,2]*B[2,0]
    DELAY 27                        # cyc 747..774

    VMATPUSH.W.MXU0   1, 31         # cyc 775: MXU0 wslot1 <- B[3,3]^T
    VMATMUL.ACC.MXU0  1, 3, 1       # cyc 776: MXU0 M7_b   acc1 += A[0,3]*B[3,3]
    DELAY 0                         # cyc 777
    VMATMUL.ACC.MXU1  0, 11, 1      # cyc 778: MXU1 M6_b   acc0 += A[2,3]*B[3,0]
    VMATPUSH.W.MXU1   0, 17         # cyc 779: MXU1 wslot0 <- B[0,1]^T (Phase B start MXU1)
    DELAY 26                        # cyc 780..806

    # ── Sub-beta Phase A->B internal seam (anchored verbatim) ──
    VMATPUSH.W.MXU0   0, 18         # cyc 807: MXU0 wslot0 <- B[0,2]^T (Phase B start MXU0)
    VMATPOP.FP8.MXU0  34, 0, 0      # cyc 808: m34 <- C[0,2] (acc0)
    VMATMUL.MXU0      0, 4, 0       # cyc 809: MXU0 M8_b   acc0 = A[1,0]*B[0,2]   (ow)
    VMATMUL.ACC.MXU1  1, 15, 1      # cyc 810: MXU1 M7_b   acc1 += A[3,3]*B[3,0]
    VMATPOP.FP8.MXU1  40, 0, 0      # cyc 811: m40 <- C[2,0] (acc0)
    DELAY 26                        # cyc 812..838

    VMATPUSH.W.MXU0   1, 19         # cyc 839: MXU0 wslot1 <- B[0,3]^T
    VMATPOP.FP8.MXU0  35, 0, 1      # cyc 840: m35 <- C[0,3] (acc1)
    VMATMUL.MXU0      1, 4, 1       # cyc 841: MXU0 M9_b   acc1 = A[1,0]*B[0,3]   (ow)
    VMATMUL.MXU1      0, 8, 0       # cyc 842: MXU1 M8_b   acc0 = A[2,0]*B[0,1]   (ow)
    VMATPUSH.W.MXU1   1, 21         # cyc 843: MXU1 wslot1 <- B[1,1]^T
    VMATPOP.FP8.MXU1  44, 0, 1      # cyc 844: m44 <- C[3,0] (acc1)
    DELAY 26                        # cyc 845..871

    VMATPUSH.W.MXU0   0, 22         # cyc 872: MXU0 wslot0 <- B[1,2]^T
    VMATMUL.ACC.MXU0  0, 5, 0       # cyc 873: MXU0 M10_b  acc0 += A[1,1]*B[1,2]
    VMATMUL.MXU1      1, 12, 0      # cyc 874: MXU1 M9_b   acc1 = A[3,0]*B[0,1]   (ow)
    DELAY 28                        # cyc 875..903

    VMATPUSH.W.MXU0   1, 23         # cyc 904: MXU0 wslot1 <- B[1,3]^T
    VMATMUL.ACC.MXU0  1, 5, 1       # cyc 905: MXU0 M11_b  acc1 += A[1,1]*B[1,3]
    VMATMUL.ACC.MXU1  0, 9, 1       # cyc 906: MXU1 M10_b  acc0 += A[2,1]*B[1,1]
    VMATPUSH.W.MXU1   0, 25         # cyc 907: MXU1 wslot0 <- B[2,1]^T
    DELAY 27                        # cyc 908..935

    VMATPUSH.W.MXU0   0, 26         # cyc 936: MXU0 wslot0 <- B[2,2]^T
    VMATMUL.ACC.MXU0  0, 6, 0       # cyc 937: MXU0 M12_b  acc0 += A[1,2]*B[2,2]
    VMATMUL.ACC.MXU1  1, 13, 1      # cyc 938: MXU1 M11_b  acc1 += A[3,1]*B[1,1]
    DELAY 28                        # cyc 939..967

    VMATPUSH.W.MXU0   1, 27         # cyc 968: MXU0 wslot1 <- B[2,3]^T
    VMATMUL.ACC.MXU0  1, 6, 1       # cyc 969: MXU0 M13_b  acc1 += A[1,2]*B[2,3]
    VMATMUL.ACC.MXU1  0, 10, 0      # cyc 970: MXU1 M12_b  acc0 += A[2,2]*B[2,1]
    VMATPUSH.W.MXU1   1, 29         # cyc 971: MXU1 wslot1 <- B[3,1]^T
    DELAY 27                        # cyc 972..999

    VMATPUSH.W.MXU0   0, 30         # cyc 1000: MXU0 wslot0 <- B[3,2]^T
    VMATMUL.ACC.MXU0  0, 7, 0       # cyc 1001: MXU0 M14_b  acc0 += A[1,3]*B[3,2]
    VMATMUL.ACC.MXU1  1, 14, 0      # cyc 1002: MXU1 M13_b  acc1 += A[3,2]*B[2,1]
    DELAY 28                        # cyc 1003..1031

    VMATPUSH.W.MXU0   1, 31         # cyc 1032: MXU0 wslot1 <- B[3,3]^T
    VMATMUL.ACC.MXU0  1, 7, 1       # cyc 1033: MXU0 M15_b  acc1 += A[1,3]*B[3,3]
    VMATMUL.ACC.MXU1  0, 11, 1      # cyc 1034: MXU1 M14_b  acc0 += A[2,3]*B[3,1]
    DELAY 29                        # cyc 1035..1064

    VMATPOP.FP8.MXU0  38, 0, 0      # cyc 1065: m38 <- C[1,2] (acc0)
    VMATMUL.ACC.MXU1  1, 15, 1      # cyc 1066: MXU1 M15_b  acc1 += A[3,3]*B[3,1]
    VMATPOP.FP8.MXU1  41, 0, 0      # cyc 1067: m41 <- C[2,1] (acc0)
    DELAY 28                        # cyc 1068..1096

    VMATPOP.FP8.MXU0  39, 0, 1      # cyc 1097: m39 <- C[1,3] (acc1)
    DELAY 0                         # cyc 1098 (final spacer)
    VMATPOP.FP8.MXU1  45, 0, 1      # cyc 1099: m45 <- C[3,1] (acc1)

# ═══════════════════════════════════════════════════════════════
# MEASURED REGION ENDS (CSRR right after last POP issue, mirrors anchors)
# ═══════════════════════════════════════════════════════════════
    CSRR  x21, 0xC00                # cyc ~1100
    SUB   x22, x21, x20             # delta cycles
    CSRW  x22, 0xC11                # stash in CSR_DBG1 (host reads @ 0x14)

# ═══════════════════════════════════════════════════════════════
# WRITEBACK (untimed): 16 C tiles m32..m47 → DRAM 0x8000..0xBFFF
# ═══════════════════════════════════════════════════════════════

    # C[0,0] (m32) -> 0x8000
    LI    x8, 0
    VSTORE 32, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90008000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[0,1] (m33) -> 0x8400
    LI    x8, 0
    VSTORE 33, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90008400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[0,2] (m34) -> 0x8800
    LI    x8, 0
    VSTORE 34, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90008800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[0,3] (m35) -> 0x8C00
    LI    x8, 0
    VSTORE 35, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90008C00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[1,0] (m36) -> 0x9000
    LI    x8, 0
    VSTORE 36, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90009000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[1,1] (m37) -> 0x9400
    LI    x8, 0
    VSTORE 37, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90009400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[1,2] (m38) -> 0x9800
    LI    x8, 0
    VSTORE 38, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90009800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[1,3] (m39) -> 0x9C00
    LI    x8, 0
    VSTORE 39, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x90009C00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[2,0] (m40) -> 0xA000
    LI    x8, 0
    VSTORE 40, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x9000A000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[2,1] (m41) -> 0xA400
    LI    x8, 0
    VSTORE 41, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x9000A400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[2,2] (m42) -> 0xA800
    LI    x8, 0
    VSTORE 42, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x9000A800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[2,3] (m43) -> 0xAC00
    LI    x8, 0
    VSTORE 43, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x9000AC00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[3,0] (m44) -> 0xB000
    LI    x8, 0
    VSTORE 44, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x9000B000
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[3,1] (m45) -> 0xB400
    LI    x8, 0
    VSTORE 45, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x9000B400
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[3,2] (m46) -> 0xB800
    LI    x8, 0
    VSTORE 46, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x9000B800
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

    # C[3,3] (m47) -> 0xBC00
    LI    x8, 0
    VSTORE 47, x8, 0
    DELAY 40
    LI    x8, 0
    LI    x3, 0x9000BC00
    DMA.STORE x3, x8, x12, 1
    DMA.WAIT  1

# ═══════════════════════════════════════════════════════════════
# DONE — host verifies DRAM 0x8000-0xBFFF and prints per-MXU util.
# ═══════════════════════════════════════════════════════════════
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

# mreg_tracker_full_sweep.S — sweep every MREG bank as both a read and write
# @TIMEOUT 1500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mreg_tracker_full_sweep.py
#
# Coverage intent:
#   - atlasTile.core.mregTracker  toggle ~30% baseline
#       The mregTracker output vectors readBusy[63:0] and writeBusy[63:0]
#       have bits that only toggle when the corresponding MREG bank is
#       actively being read or written. Existing tests touch a tight
#       subset of banks (mostly 0..15, 48..63 via the "upper bank"
#       test), leaving many mid-range banks never active.
#   - atlasTile.core.mreg wrapper line 45.82 / branch 36.55 — sweeping
#     every bank in turn exercises wrapper-level decoders for bank
#     indices that never normally fire.
#   - Individual bank shells with known gaps: banks_32 (77.95%),
#     banks_34 (77.47%), banks_35 (78.23%), banks_49 (65.02%),
#     banks_51 (75.86%). This test also touches these banks through
#     cross-engine paths (VPU pair op, XLU, MXU push/pop) rather than
#     VLOAD/VSTORE alone.
#
# Strategy:
#   1. DMA-load a single 1 KiB FP8 reference tile into VMEM.
#   2. VLOAD that tile into EVERY bank 0..63 in sequence. Each VLOAD
#      drives writeBusy[b]=1 for its bank b for 33 cycles; the sequence
#      therefore walks all 64 write-tracker bits.
#   3. VSTORE EVERY bank 0..63 into a dedicated 64 KiB VMEM staging
#      region. Each VSTORE drives readBusy[b]=1, walking all 64
#      read-tracker bits.
#   4. DMA-store the 64 KiB staging region to DRAM for oracle checking.
#   5. Cross-engine passes on the known-weak banks {32, 34, 35, 49, 51}:
#        - VPU VMOV pair ops read/write these banks
#        - XLU transpose uses them as source/dest
#        - MXU0/MXU1 push/pop uses them as weight / accumulator sources
#      These touches don't extend the oracle (they re-stage into banks
#      we never check), but they do exercise the tracker from multiple
#      non-LSU engines on those specific banks.
#
# VMEM word layout:
#   [0..255]          source tile (1024 B)
#   [1024..17407]     64x 256-word slots for VSTORE drains of bank b
#                       slot_b_word_base = 1024 + b * 256
#                       (b = 0..63)
#
# DRAM layout:
#   0x00000..0x003FF   input tile                (1024 B)
#   0x01000..0x10FFF   64 contiguous copies of
#                      the input tile — one per
#                      bank, in bank-index order (64 KiB = 64 * 1 KiB)

    ADDI  x28, x0, 1
    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 1024             # 1 KiB tile size

# ----------------------------------------------------------------------
# PHASE 0: Load source tile into VMEM[word 0 .. 255]
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x12, 0
    DMA.WAIT  0

# ----------------------------------------------------------------------
# PHASE 1: VLOAD every MREG bank 0..63 from the same source tile.
#          This drives writeBusy[b] active for each bank in turn.
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0             # VMEM word base = 0 for every VLOAD
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 0
    DELAY 40
    VLOAD 2, x6, 0
    DELAY 40
    VLOAD 3, x6, 0
    DELAY 40
    VLOAD 4, x6, 0
    DELAY 40
    VLOAD 5, x6, 0
    DELAY 40
    VLOAD 6, x6, 0
    DELAY 40
    VLOAD 7, x6, 0
    DELAY 40
    VLOAD 8, x6, 0
    DELAY 40
    VLOAD 9, x6, 0
    DELAY 40
    VLOAD 10, x6, 0
    DELAY 40
    VLOAD 11, x6, 0
    DELAY 40
    VLOAD 12, x6, 0
    DELAY 40
    VLOAD 13, x6, 0
    DELAY 40
    VLOAD 14, x6, 0
    DELAY 40
    VLOAD 15, x6, 0
    DELAY 40
    VLOAD 16, x6, 0
    DELAY 40
    VLOAD 17, x6, 0
    DELAY 40
    VLOAD 18, x6, 0
    DELAY 40
    VLOAD 19, x6, 0
    DELAY 40
    VLOAD 20, x6, 0
    DELAY 40
    VLOAD 21, x6, 0
    DELAY 40
    VLOAD 22, x6, 0
    DELAY 40
    VLOAD 23, x6, 0
    DELAY 40
    VLOAD 24, x6, 0
    DELAY 40
    VLOAD 25, x6, 0
    DELAY 40
    VLOAD 26, x6, 0
    DELAY 40
    VLOAD 27, x6, 0
    DELAY 40
    VLOAD 28, x6, 0
    DELAY 40
    VLOAD 29, x6, 0
    DELAY 40
    VLOAD 30, x6, 0
    DELAY 40
    VLOAD 31, x6, 0
    DELAY 40
    VLOAD 32, x6, 0
    DELAY 40
    VLOAD 33, x6, 0
    DELAY 40
    VLOAD 34, x6, 0
    DELAY 40
    VLOAD 35, x6, 0
    DELAY 40
    VLOAD 36, x6, 0
    DELAY 40
    VLOAD 37, x6, 0
    DELAY 40
    VLOAD 38, x6, 0
    DELAY 40
    VLOAD 39, x6, 0
    DELAY 40
    VLOAD 40, x6, 0
    DELAY 40
    VLOAD 41, x6, 0
    DELAY 40
    VLOAD 42, x6, 0
    DELAY 40
    VLOAD 43, x6, 0
    DELAY 40
    VLOAD 44, x6, 0
    DELAY 40
    VLOAD 45, x6, 0
    DELAY 40
    VLOAD 46, x6, 0
    DELAY 40
    VLOAD 47, x6, 0
    DELAY 40
    VLOAD 48, x6, 0
    DELAY 40
    VLOAD 49, x6, 0
    DELAY 40
    VLOAD 50, x6, 0
    DELAY 40
    VLOAD 51, x6, 0
    DELAY 40
    VLOAD 52, x6, 0
    DELAY 40
    VLOAD 53, x6, 0
    DELAY 40
    VLOAD 54, x6, 0
    DELAY 40
    VLOAD 55, x6, 0
    DELAY 40
    VLOAD 56, x6, 0
    DELAY 40
    VLOAD 57, x6, 0
    DELAY 40
    VLOAD 58, x6, 0
    DELAY 40
    VLOAD 59, x6, 0
    DELAY 40
    VLOAD 60, x6, 0
    DELAY 40
    VLOAD 61, x6, 0
    DELAY 40
    VLOAD 62, x6, 0
    DELAY 40
    VLOAD 63, x6, 0
    DELAY 40

# ----------------------------------------------------------------------
# PHASE 2: Cross-engine stimulation on weakly-covered banks.
#          Uses fresh inputs already loaded above. Each of these
#          pair ops reads/writes banks that are otherwise never
#          touched from non-LSU engines.
#
#   VPU pair ops must have even vd, vs1, vs2. We pick pairs:
#     (32,33) -> (34,35)       via VMOV
#     (48,49) -> (50,51)       via VMOV
#     (52,53) -> (32,33)       via VMOV       [rewrites 32 from VPU]
#
#   XLU:
#     m34 <- xlu(m35)   (single bank)
#     m49 <- xlu(m51)
#
#   MXU0 weight push from m34, acc seed from m3, matmul on m32,
#   pop back into m36. This exercises the tracker seeing MXU-sourced
#   reads/writes on mid-range banks.
# ----------------------------------------------------------------------

    # VPU VMOV (32,33) -> (34,35)
    VMOV  34, 32
    DELAY 160

    # VPU VMOV (48,49) -> (50,51)
    VMOV  50, 48
    DELAY 160

    # VPU VMOV (52,53) -> (32,33) — rewrites 32,33 from the VPU
    VMOV  32, 52
    DELAY 160

    # XLU transpose: single-bank source -> single-bank dest
    VTRPOSE.XLU 34, 35
    DELAY 80

    VTRPOSE.XLU 49, 51
    DELAY 80

    # MXU0 touch on weak banks: push weight from m34, zero-seed acc from m3,
    # compute using m32 as activation, pop back into m36.
    VMATPUSH.W.MXU0       0, 34
    DELAY 40
    VMATPUSH.ACC.FP8.MXU0 0, 3
    DELAY 40
    VMATMUL.MXU0          0, 32, 0
    DELAY 140
    VMATPOP.FP8.MXU0      36, 0, 0
    DELAY 40

    # MXU1 touch: push weight from m49, acc seed from m3, matmul with m51, pop to m53.
    VMATPUSH.W.MXU1       0, 49
    DELAY 40
    VMATPUSH.ACC.FP8.MXU1 0, 3
    DELAY 40
    VMATMUL.MXU1          0, 51, 0
    DELAY 140
    VMATPOP.FP8.MXU1      53, 0, 0
    DELAY 40

# ----------------------------------------------------------------------
# PHASE 3: Reload EVERY bank 0..63 with the source tile again so the
#          VSTORE sweep below sees a known pristine copy in every bank.
#          This is what the generator oracle assumes.
# ----------------------------------------------------------------------
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 0
    DELAY 40
    VLOAD 2, x6, 0
    DELAY 40
    VLOAD 3, x6, 0
    DELAY 40
    VLOAD 4, x6, 0
    DELAY 40
    VLOAD 5, x6, 0
    DELAY 40
    VLOAD 6, x6, 0
    DELAY 40
    VLOAD 7, x6, 0
    DELAY 40
    VLOAD 8, x6, 0
    DELAY 40
    VLOAD 9, x6, 0
    DELAY 40
    VLOAD 10, x6, 0
    DELAY 40
    VLOAD 11, x6, 0
    DELAY 40
    VLOAD 12, x6, 0
    DELAY 40
    VLOAD 13, x6, 0
    DELAY 40
    VLOAD 14, x6, 0
    DELAY 40
    VLOAD 15, x6, 0
    DELAY 40
    VLOAD 16, x6, 0
    DELAY 40
    VLOAD 17, x6, 0
    DELAY 40
    VLOAD 18, x6, 0
    DELAY 40
    VLOAD 19, x6, 0
    DELAY 40
    VLOAD 20, x6, 0
    DELAY 40
    VLOAD 21, x6, 0
    DELAY 40
    VLOAD 22, x6, 0
    DELAY 40
    VLOAD 23, x6, 0
    DELAY 40
    VLOAD 24, x6, 0
    DELAY 40
    VLOAD 25, x6, 0
    DELAY 40
    VLOAD 26, x6, 0
    DELAY 40
    VLOAD 27, x6, 0
    DELAY 40
    VLOAD 28, x6, 0
    DELAY 40
    VLOAD 29, x6, 0
    DELAY 40
    VLOAD 30, x6, 0
    DELAY 40
    VLOAD 31, x6, 0
    DELAY 40
    VLOAD 32, x6, 0
    DELAY 40
    VLOAD 33, x6, 0
    DELAY 40
    VLOAD 34, x6, 0
    DELAY 40
    VLOAD 35, x6, 0
    DELAY 40
    VLOAD 36, x6, 0
    DELAY 40
    VLOAD 37, x6, 0
    DELAY 40
    VLOAD 38, x6, 0
    DELAY 40
    VLOAD 39, x6, 0
    DELAY 40
    VLOAD 40, x6, 0
    DELAY 40
    VLOAD 41, x6, 0
    DELAY 40
    VLOAD 42, x6, 0
    DELAY 40
    VLOAD 43, x6, 0
    DELAY 40
    VLOAD 44, x6, 0
    DELAY 40
    VLOAD 45, x6, 0
    DELAY 40
    VLOAD 46, x6, 0
    DELAY 40
    VLOAD 47, x6, 0
    DELAY 40
    VLOAD 48, x6, 0
    DELAY 40
    VLOAD 49, x6, 0
    DELAY 40
    VLOAD 50, x6, 0
    DELAY 40
    VLOAD 51, x6, 0
    DELAY 40
    VLOAD 52, x6, 0
    DELAY 40
    VLOAD 53, x6, 0
    DELAY 40
    VLOAD 54, x6, 0
    DELAY 40
    VLOAD 55, x6, 0
    DELAY 40
    VLOAD 56, x6, 0
    DELAY 40
    VLOAD 57, x6, 0
    DELAY 40
    VLOAD 58, x6, 0
    DELAY 40
    VLOAD 59, x6, 0
    DELAY 40
    VLOAD 60, x6, 0
    DELAY 40
    VLOAD 61, x6, 0
    DELAY 40
    VLOAD 62, x6, 0
    DELAY 40
    VLOAD 63, x6, 0
    DELAY 40

# ----------------------------------------------------------------------
# PHASE 4: VSTORE each bank 0..63 into its own 1 KiB VMEM slot starting
#          at VMEM word 1024. Each VSTORE drives readBusy[b] active.
#
#   slot_b_word_base = 1024 + b * 256
#   immediate step = 32 words, so imm12 = slot_b_word_base / 32
#                                       = 32 + b * 8
# ----------------------------------------------------------------------
    LI    x7, 1024
    VSTORE 0,  x7, 0
    DELAY 40
    VSTORE 1,  x7, 8
    DELAY 40
    VSTORE 2,  x7, 16
    DELAY 40
    VSTORE 3,  x7, 24
    DELAY 40
    VSTORE 4,  x7, 32
    DELAY 40
    VSTORE 5,  x7, 40
    DELAY 40
    VSTORE 6,  x7, 48
    DELAY 40
    VSTORE 7,  x7, 56
    DELAY 40
    VSTORE 8,  x7, 64
    DELAY 40
    VSTORE 9,  x7, 72
    DELAY 40
    VSTORE 10, x7, 80
    DELAY 40
    VSTORE 11, x7, 88
    DELAY 40
    VSTORE 12, x7, 96
    DELAY 40
    VSTORE 13, x7, 104
    DELAY 40
    VSTORE 14, x7, 112
    DELAY 40
    VSTORE 15, x7, 120
    DELAY 40
    VSTORE 16, x7, 128
    DELAY 40
    VSTORE 17, x7, 136
    DELAY 40
    VSTORE 18, x7, 144
    DELAY 40
    VSTORE 19, x7, 152
    DELAY 40
    VSTORE 20, x7, 160
    DELAY 40
    VSTORE 21, x7, 168
    DELAY 40
    VSTORE 22, x7, 176
    DELAY 40
    VSTORE 23, x7, 184
    DELAY 40
    VSTORE 24, x7, 192
    DELAY 40
    VSTORE 25, x7, 200
    DELAY 40
    VSTORE 26, x7, 208
    DELAY 40
    VSTORE 27, x7, 216
    DELAY 40
    VSTORE 28, x7, 224
    DELAY 40
    VSTORE 29, x7, 232
    DELAY 40
    VSTORE 30, x7, 240
    DELAY 40
    VSTORE 31, x7, 248
    DELAY 40
    VSTORE 32, x7, 256
    DELAY 40
    VSTORE 33, x7, 264
    DELAY 40
    VSTORE 34, x7, 272
    DELAY 40
    VSTORE 35, x7, 280
    DELAY 40
    VSTORE 36, x7, 288
    DELAY 40
    VSTORE 37, x7, 296
    DELAY 40
    VSTORE 38, x7, 304
    DELAY 40
    VSTORE 39, x7, 312
    DELAY 40
    VSTORE 40, x7, 320
    DELAY 40
    VSTORE 41, x7, 328
    DELAY 40
    VSTORE 42, x7, 336
    DELAY 40
    VSTORE 43, x7, 344
    DELAY 40
    VSTORE 44, x7, 352
    DELAY 40
    VSTORE 45, x7, 360
    DELAY 40
    VSTORE 46, x7, 368
    DELAY 40
    VSTORE 47, x7, 376
    DELAY 40
    VSTORE 48, x7, 384
    DELAY 40
    VSTORE 49, x7, 392
    DELAY 40
    VSTORE 50, x7, 400
    DELAY 40
    VSTORE 51, x7, 408
    DELAY 40
    VSTORE 52, x7, 416
    DELAY 40
    VSTORE 53, x7, 424
    DELAY 40
    VSTORE 54, x7, 432
    DELAY 40
    VSTORE 55, x7, 440
    DELAY 40
    VSTORE 56, x7, 448
    DELAY 40
    VSTORE 57, x7, 456
    DELAY 40
    VSTORE 58, x7, 464
    DELAY 40
    VSTORE 59, x7, 472
    DELAY 40
    VSTORE 60, x7, 480
    DELAY 40
    VSTORE 61, x7, 488
    DELAY 40
    VSTORE 62, x7, 496
    DELAY 40
    VSTORE 63, x7, 504
    DELAY 40

# ----------------------------------------------------------------------
# PHASE 5: Drain the 64 KiB staging region to DRAM for oracle check.
#          DMA single-transfer max is 4096 B = 4 KiB, so do 16 x 4 KiB
#          back-to-back DMA.STOREs on channel 1.
# ----------------------------------------------------------------------
    LI    x11, 4096             # DMA size

    # chunk i: VMEM word base = 1024 + i * 1024, DRAM byte = 0x1000 + i*0x1000
    LI    x8, 1024
    LI    x10, 0x1000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 2048
    LI    x10, 0x2000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 3072
    LI    x10, 0x3000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 4096
    LI    x10, 0x4000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 5120
    LI    x10, 0x5000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 6144
    LI    x10, 0x6000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 7168
    LI    x10, 0x7000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 8192
    LI    x10, 0x8000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 9216
    LI    x10, 0x9000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 10240
    LI    x10, 0xA000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 11264
    LI    x10, 0xB000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 12288
    LI    x10, 0xC000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 13312
    LI    x10, 0xD000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 14336
    LI    x10, 0xE000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 15360
    LI    x10, 0xF000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

    LI    x8, 16384
    LI    x10, 0x10000
    DMA.STORE x10, x8, x11, 1
    DMA.WAIT 1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

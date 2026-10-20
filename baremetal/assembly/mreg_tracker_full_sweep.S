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
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024             # 1 KiB tile size

# ----------------------------------------------------------------------
# PHASE 0: Load source tile into VMEM[word 0 .. 255]
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x12, ch0
    dma.wait ch0

# ----------------------------------------------------------------------
# PHASE 1: VLOAD every MREG bank 0..63 from the same source tile.
#          This drives writeBusy[b] active for each bank in turn.
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0             # VMEM word base = 0 for every VLOAD
    vload m0, 0(x6)
    DELAY 33
    vload m1, 0(x6)
    DELAY 33
    vload m2, 0(x6)
    DELAY 33
    vload m3, 0(x6)
    DELAY 33
    vload m4, 0(x6)
    DELAY 33
    vload m5, 0(x6)
    DELAY 33
    vload m6, 0(x6)
    DELAY 33
    vload m7, 0(x6)
    DELAY 33
    vload m8, 0(x6)
    DELAY 33
    vload m9, 0(x6)
    DELAY 33
    vload m10, 0(x6)
    DELAY 33
    vload m11, 0(x6)
    DELAY 33
    vload m12, 0(x6)
    DELAY 33
    vload m13, 0(x6)
    DELAY 33
    vload m14, 0(x6)
    DELAY 33
    vload m15, 0(x6)
    DELAY 33
    vload m16, 0(x6)
    DELAY 33
    vload m17, 0(x6)
    DELAY 33
    vload m18, 0(x6)
    DELAY 33
    vload m19, 0(x6)
    DELAY 33
    vload m20, 0(x6)
    DELAY 33
    vload m21, 0(x6)
    DELAY 33
    vload m22, 0(x6)
    DELAY 33
    vload m23, 0(x6)
    DELAY 33
    vload m24, 0(x6)
    DELAY 33
    vload m25, 0(x6)
    DELAY 33
    vload m26, 0(x6)
    DELAY 33
    vload m27, 0(x6)
    DELAY 33
    vload m28, 0(x6)
    DELAY 33
    vload m29, 0(x6)
    DELAY 33
    vload m30, 0(x6)
    DELAY 33
    vload m31, 0(x6)
    DELAY 33
    vload m32, 0(x6)
    DELAY 33
    vload m33, 0(x6)
    DELAY 33
    vload m34, 0(x6)
    DELAY 33
    vload m35, 0(x6)
    DELAY 33
    vload m36, 0(x6)
    DELAY 33
    vload m37, 0(x6)
    DELAY 33
    vload m38, 0(x6)
    DELAY 33
    vload m39, 0(x6)
    DELAY 33
    vload m40, 0(x6)
    DELAY 33
    vload m41, 0(x6)
    DELAY 33
    vload m42, 0(x6)
    DELAY 33
    vload m43, 0(x6)
    DELAY 33
    vload m44, 0(x6)
    DELAY 33
    vload m45, 0(x6)
    DELAY 33
    vload m46, 0(x6)
    DELAY 33
    vload m47, 0(x6)
    DELAY 33
    vload m48, 0(x6)
    DELAY 33
    vload m49, 0(x6)
    DELAY 33
    vload m50, 0(x6)
    DELAY 33
    vload m51, 0(x6)
    DELAY 33
    vload m52, 0(x6)
    DELAY 33
    vload m53, 0(x6)
    DELAY 33
    vload m54, 0(x6)
    DELAY 33
    vload m55, 0(x6)
    DELAY 33
    vload m56, 0(x6)
    DELAY 33
    vload m57, 0(x6)
    DELAY 33
    vload m58, 0(x6)
    DELAY 33
    vload m59, 0(x6)
    DELAY 33
    vload m60, 0(x6)
    DELAY 33
    vload m61, 0(x6)
    DELAY 33
    vload m62, 0(x6)
    DELAY 33
    vload m63, 0(x6)
    DELAY 33

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
    vmov m34, m32
    DELAY 64

    # VPU VMOV (48,49) -> (50,51)
    vmov m50, m48
    DELAY 64

    # VPU VMOV (52,53) -> (32,33) — rewrites 32,33 from the VPU
    vmov m32, m52
    DELAY 64

    # XLU transpose: single-bank source -> single-bank dest
    vtrpose.xlu m34, m35
    DELAY 64

    vtrpose.xlu m49, m51
    DELAY 64

    # MXU0 touch on weak banks: push weight from m34, zero-seed acc from m3,
    # compute using m32 as activation, pop back into m36.
    vmatpush.weight.mxu0 w0, m34
    DELAY 31
    vmatpush.acc.fp8.mxu0 acc0, m3
    DELAY 31
    vmatmul.mxu0 acc0, m32, w0
    DELAY 95
    vmatpop.fp8.acc.mxu0 m36, acc0, e0
    DELAY 31

    # MXU1 touch: push weight from m49, acc seed from m3, matmul with m51, pop to m53.
    vmatpush.weight.mxu1 w0, m49
    DELAY 31
    vmatpush.acc.fp8.mxu1 acc0, m3
    DELAY 31
    vmatmul.mxu1 acc0, m51, w0
    DELAY 34
    vmatpop.fp8.acc.mxu1 m53, acc0, e0
    DELAY 31

# ----------------------------------------------------------------------
# PHASE 3: Reload EVERY bank 0..63 with the source tile again so the
#          VSTORE sweep below sees a known pristine copy in every bank.
#          This is what the generator oracle assumes.
# ----------------------------------------------------------------------
    vload m0, 0(x6)
    DELAY 33
    vload m1, 0(x6)
    DELAY 33
    vload m2, 0(x6)
    DELAY 33
    vload m3, 0(x6)
    DELAY 33
    vload m4, 0(x6)
    DELAY 33
    vload m5, 0(x6)
    DELAY 33
    vload m6, 0(x6)
    DELAY 33
    vload m7, 0(x6)
    DELAY 33
    vload m8, 0(x6)
    DELAY 33
    vload m9, 0(x6)
    DELAY 33
    vload m10, 0(x6)
    DELAY 33
    vload m11, 0(x6)
    DELAY 33
    vload m12, 0(x6)
    DELAY 33
    vload m13, 0(x6)
    DELAY 33
    vload m14, 0(x6)
    DELAY 33
    vload m15, 0(x6)
    DELAY 33
    vload m16, 0(x6)
    DELAY 33
    vload m17, 0(x6)
    DELAY 33
    vload m18, 0(x6)
    DELAY 33
    vload m19, 0(x6)
    DELAY 33
    vload m20, 0(x6)
    DELAY 33
    vload m21, 0(x6)
    DELAY 33
    vload m22, 0(x6)
    DELAY 33
    vload m23, 0(x6)
    DELAY 33
    vload m24, 0(x6)
    DELAY 33
    vload m25, 0(x6)
    DELAY 33
    vload m26, 0(x6)
    DELAY 33
    vload m27, 0(x6)
    DELAY 33
    vload m28, 0(x6)
    DELAY 33
    vload m29, 0(x6)
    DELAY 33
    vload m30, 0(x6)
    DELAY 33
    vload m31, 0(x6)
    DELAY 33
    vload m32, 0(x6)
    DELAY 33
    vload m33, 0(x6)
    DELAY 33
    vload m34, 0(x6)
    DELAY 33
    vload m35, 0(x6)
    DELAY 33
    vload m36, 0(x6)
    DELAY 33
    vload m37, 0(x6)
    DELAY 33
    vload m38, 0(x6)
    DELAY 33
    vload m39, 0(x6)
    DELAY 33
    vload m40, 0(x6)
    DELAY 33
    vload m41, 0(x6)
    DELAY 33
    vload m42, 0(x6)
    DELAY 33
    vload m43, 0(x6)
    DELAY 33
    vload m44, 0(x6)
    DELAY 33
    vload m45, 0(x6)
    DELAY 33
    vload m46, 0(x6)
    DELAY 33
    vload m47, 0(x6)
    DELAY 33
    vload m48, 0(x6)
    DELAY 33
    vload m49, 0(x6)
    DELAY 33
    vload m50, 0(x6)
    DELAY 33
    vload m51, 0(x6)
    DELAY 33
    vload m52, 0(x6)
    DELAY 33
    vload m53, 0(x6)
    DELAY 33
    vload m54, 0(x6)
    DELAY 33
    vload m55, 0(x6)
    DELAY 33
    vload m56, 0(x6)
    DELAY 33
    vload m57, 0(x6)
    DELAY 33
    vload m58, 0(x6)
    DELAY 33
    vload m59, 0(x6)
    DELAY 33
    vload m60, 0(x6)
    DELAY 33
    vload m61, 0(x6)
    DELAY 33
    vload m62, 0(x6)
    DELAY 33
    vload m63, 0(x6)
    DELAY 33

# ----------------------------------------------------------------------
# PHASE 4: VSTORE each bank 0..63 into its own 1 KiB VMEM slot starting
#          at VMEM word 1024. Each VSTORE drives readBusy[b] active.
#
#   slot_b_word_base = 1024 + b * 256
#   immediate step = 32 words, so imm12 = slot_b_word_base / 32
#                                       = 32 + b * 8
# ----------------------------------------------------------------------
    LI    x7, 1024
    vstore m0, 0(x7)
    DELAY 33
    vstore m1, 8(x7)
    DELAY 33
    vstore m2, 16(x7)
    DELAY 33
    vstore m3, 24(x7)
    DELAY 33
    vstore m4, 32(x7)
    DELAY 33
    vstore m5, 40(x7)
    DELAY 33
    vstore m6, 48(x7)
    DELAY 33
    vstore m7, 56(x7)
    DELAY 33
    vstore m8, 64(x7)
    DELAY 33
    vstore m9, 72(x7)
    DELAY 33
    vstore m10, 80(x7)
    DELAY 33
    vstore m11, 88(x7)
    DELAY 33
    vstore m12, 96(x7)
    DELAY 33
    vstore m13, 104(x7)
    DELAY 33
    vstore m14, 112(x7)
    DELAY 33
    vstore m15, 120(x7)
    DELAY 33
    vstore m16, 128(x7)
    DELAY 33
    vstore m17, 136(x7)
    DELAY 33
    vstore m18, 144(x7)
    DELAY 33
    vstore m19, 152(x7)
    DELAY 33
    vstore m20, 160(x7)
    DELAY 33
    vstore m21, 168(x7)
    DELAY 33
    vstore m22, 176(x7)
    DELAY 33
    vstore m23, 184(x7)
    DELAY 33
    vstore m24, 192(x7)
    DELAY 33
    vstore m25, 200(x7)
    DELAY 33
    vstore m26, 208(x7)
    DELAY 33
    vstore m27, 216(x7)
    DELAY 33
    vstore m28, 224(x7)
    DELAY 33
    vstore m29, 232(x7)
    DELAY 33
    vstore m30, 240(x7)
    DELAY 33
    vstore m31, 248(x7)
    DELAY 33
    vstore m32, 256(x7)
    DELAY 33
    vstore m33, 264(x7)
    DELAY 33
    vstore m34, 272(x7)
    DELAY 33
    vstore m35, 280(x7)
    DELAY 33
    vstore m36, 288(x7)
    DELAY 33
    vstore m37, 296(x7)
    DELAY 33
    vstore m38, 304(x7)
    DELAY 33
    vstore m39, 312(x7)
    DELAY 33
    vstore m40, 320(x7)
    DELAY 33
    vstore m41, 328(x7)
    DELAY 33
    vstore m42, 336(x7)
    DELAY 33
    vstore m43, 344(x7)
    DELAY 33
    vstore m44, 352(x7)
    DELAY 33
    vstore m45, 360(x7)
    DELAY 33
    vstore m46, 368(x7)
    DELAY 33
    vstore m47, 376(x7)
    DELAY 33
    vstore m48, 384(x7)
    DELAY 33
    vstore m49, 392(x7)
    DELAY 33
    vstore m50, 400(x7)
    DELAY 33
    vstore m51, 408(x7)
    DELAY 33
    vstore m52, 416(x7)
    DELAY 33
    vstore m53, 424(x7)
    DELAY 33
    vstore m54, 432(x7)
    DELAY 33
    vstore m55, 440(x7)
    DELAY 33
    vstore m56, 448(x7)
    DELAY 33
    vstore m57, 456(x7)
    DELAY 33
    vstore m58, 464(x7)
    DELAY 33
    vstore m59, 472(x7)
    DELAY 33
    vstore m60, 480(x7)
    DELAY 33
    vstore m61, 488(x7)
    DELAY 33
    vstore m62, 496(x7)
    DELAY 33
    vstore m63, 504(x7)
    DELAY 33

# ----------------------------------------------------------------------
# PHASE 5: Drain the 64 KiB staging region to DRAM for oracle check.
#          DMA single-transfer max is 4096 B = 4 KiB, so do 16 x 4 KiB
#          back-to-back DMA.STOREs on channel 1.
# ----------------------------------------------------------------------
    LI    x11, 4096             # DMA size

    # chunk i: VMEM word base = 1024 + i * 1024, DRAM byte = 0x1000 + i*0x1000
    LI    x8, 1024
    LI    x10, 0x90001000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 2048
    LI    x10, 0x90002000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 3072
    LI    x10, 0x90003000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 4096
    LI    x10, 0x90004000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 5120
    LI    x10, 0x90005000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 6144
    LI    x10, 0x90006000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 7168
    LI    x10, 0x90007000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 8192
    LI    x10, 0x90008000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 9216
    LI    x10, 0x90009000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 10240
    LI    x10, 0x9000A000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 11264
    LI    x10, 0x9000B000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 12288
    LI    x10, 0x9000C000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 13312
    LI    x10, 0x9000D000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 14336
    LI    x10, 0x9000E000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 15360
    LI    x10, 0x9000F000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

    LI    x8, 16384
    LI    x10, 0x90010000
    dma.store x10, x8, x11, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

# mreg_cross_engine_full_rotor.S - drive MREG tracker with XLU + VPU traffic
#                                   on every bank index in 0..63
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mreg_cross_engine_full_rotor.py
#
# Coverage intent (cov_overall.modinfo.txt):
#   - core.mregTracker  toggle 33.71%   <-- target
#       The tracker maintains per-bank read/write busy state and a
#       per-bank "source engine" code. Existing tests touch every
#       bank from VLOAD/VSTORE only; non-LSU engines (XLU, VPU, MXU)
#       fire on a small handful of mid-range banks. This test walks
#       VTRPOSE.XLU through every bank index 0..63 as both reader
#       and writer, then walks VMOV pair ops across the same range,
#       so the tracker's source-engine code field flips on every
#       bank and its readBusy/writeBusy bits toggle from non-LSU
#       paths repeatedly.
#   - core.xlu wrapper line/branch coverage benefits from a long
#     in-Idle -> in-Busy -> in-Idle chain on consecutive cycles.
#   - core.mreg.banks_<N>  shells for upper banks see fresh data
#     buses through the VMOV chain.
#
# Design (golden-by-construction):
#
#   The XLU chain m1 = T(m0), m2 = T(m1), ... = T applied k times to m0
#   gives:
#       even banks = m0          (T applied an even number of times)
#       odd  banks = T(m0)       (T applied an odd  number of times)
#
#   We avoid having to model XLU element width by feeding a SYMMETRIC
#   1024-byte tile (byte[r][c] == byte[c][r] for r, c in 0..31). For a
#   symmetric 32x32 byte matrix, T(m0) == m0 byte-for-byte regardless
#   of how XLU groups elements.  The oracle for any chained bank is
#   therefore the original tile.
#
#   After the XLU chain, every bank 0..63 holds the symmetric tile.
#   We then issue a VPU VMOV rotor that rewrites pair (b, b+1) from
#   pair (b-2, b-1) for b in {2, 4, 6, ..., 62}, touching every even
#   pair via VPU (writeBusy on (b, b+1), readBusy on (b-2, b-1)).
#
#   Finally we VSTORE bank 63 back to VMEM, DMA.STORE to DRAM byte
#   offset 0x1000, and the generator checks that the bytes match the
#   original symmetric tile.
#
# VMEM word layout:
#     0..255     input tile staging (1 KiB)
#     256..511   output staging     (1 KiB)
#
# DRAM byte layout:
#     0x0000..0x03FF   input symmetric tile
#     0x1000..0x13FF   final VSTORE drain (oracle: equals input tile)

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024              # 1 KiB tile size

# ----------------------------------------------------------------------
# PHASE 0: load the symmetric tile from DRAM into VMEM[0..255], then
#          VLOAD m0 from VMEM bank 0.
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x12, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

# ----------------------------------------------------------------------
# PHASE 1: XLU rotor m_k = T(m_(k-1)) for k = 1..63. Each VTRPOSE.XLU
#          touches a single source bank (read) and a single dest bank
#          (write), exercising the tracker's XLU-source bit pattern on
#          EVERY bank index in turn. Consecutive transposes also stress
#          the XLU FSM Idle -> Busy -> Idle handoff 63 times in a row.
# ----------------------------------------------------------------------
    vtrpose.xlu m1, m0
    DELAY 64
    vtrpose.xlu m2, m1
    DELAY 64
    vtrpose.xlu m3, m2
    DELAY 64
    vtrpose.xlu m4, m3
    DELAY 64
    vtrpose.xlu m5, m4
    DELAY 64
    vtrpose.xlu m6, m5
    DELAY 64
    vtrpose.xlu m7, m6
    DELAY 64
    vtrpose.xlu m8, m7
    DELAY 64
    vtrpose.xlu m9, m8
    DELAY 64
    vtrpose.xlu m10, m9
    DELAY 64
    vtrpose.xlu m11, m10
    DELAY 64
    vtrpose.xlu m12, m11
    DELAY 64
    vtrpose.xlu m13, m12
    DELAY 64
    vtrpose.xlu m14, m13
    DELAY 64
    vtrpose.xlu m15, m14
    DELAY 64
    vtrpose.xlu m16, m15
    DELAY 64
    vtrpose.xlu m17, m16
    DELAY 64
    vtrpose.xlu m18, m17
    DELAY 64
    vtrpose.xlu m19, m18
    DELAY 64
    vtrpose.xlu m20, m19
    DELAY 64
    vtrpose.xlu m21, m20
    DELAY 64
    vtrpose.xlu m22, m21
    DELAY 64
    vtrpose.xlu m23, m22
    DELAY 64
    vtrpose.xlu m24, m23
    DELAY 64
    vtrpose.xlu m25, m24
    DELAY 64
    vtrpose.xlu m26, m25
    DELAY 64
    vtrpose.xlu m27, m26
    DELAY 64
    vtrpose.xlu m28, m27
    DELAY 64
    vtrpose.xlu m29, m28
    DELAY 64
    vtrpose.xlu m30, m29
    DELAY 64
    vtrpose.xlu m31, m30
    DELAY 64
    vtrpose.xlu m32, m31
    DELAY 64
    vtrpose.xlu m33, m32
    DELAY 64
    vtrpose.xlu m34, m33
    DELAY 64
    vtrpose.xlu m35, m34
    DELAY 64
    vtrpose.xlu m36, m35
    DELAY 64
    vtrpose.xlu m37, m36
    DELAY 64
    vtrpose.xlu m38, m37
    DELAY 64
    vtrpose.xlu m39, m38
    DELAY 64
    vtrpose.xlu m40, m39
    DELAY 64
    vtrpose.xlu m41, m40
    DELAY 64
    vtrpose.xlu m42, m41
    DELAY 64
    vtrpose.xlu m43, m42
    DELAY 64
    vtrpose.xlu m44, m43
    DELAY 64
    vtrpose.xlu m45, m44
    DELAY 64
    vtrpose.xlu m46, m45
    DELAY 64
    vtrpose.xlu m47, m46
    DELAY 64
    vtrpose.xlu m48, m47
    DELAY 64
    vtrpose.xlu m49, m48
    DELAY 64
    vtrpose.xlu m50, m49
    DELAY 64
    vtrpose.xlu m51, m50
    DELAY 64
    vtrpose.xlu m52, m51
    DELAY 64
    vtrpose.xlu m53, m52
    DELAY 64
    vtrpose.xlu m54, m53
    DELAY 64
    vtrpose.xlu m55, m54
    DELAY 64
    vtrpose.xlu m56, m55
    DELAY 64
    vtrpose.xlu m57, m56
    DELAY 64
    vtrpose.xlu m58, m57
    DELAY 64
    vtrpose.xlu m59, m58
    DELAY 64
    vtrpose.xlu m60, m59
    DELAY 64
    vtrpose.xlu m61, m60
    DELAY 64
    vtrpose.xlu m62, m61
    DELAY 64
    vtrpose.xlu m63, m62
    DELAY 64

# ----------------------------------------------------------------------
# PHASE 2: VPU pair-op rotor. Each VMOV reads pair (b-2, b-1) and
#          writes pair (b, b+1). Going b = 2, 4, ..., 62 walks every
#          even pair across the bank space, so the tracker sees VPU-
#          source reads on (0,1)..(60,61) and VPU-source writes on
#          (2,3)..(62,63).
# ----------------------------------------------------------------------
    vmov m2, m0
    DELAY 64
    vmov m4, m2
    DELAY 64
    vmov m6, m4
    DELAY 64
    vmov m8, m6
    DELAY 64
    vmov m10, m8
    DELAY 64
    vmov m12, m10
    DELAY 64
    vmov m14, m12
    DELAY 64
    vmov m16, m14
    DELAY 64
    vmov m18, m16
    DELAY 64
    vmov m20, m18
    DELAY 64
    vmov m22, m20
    DELAY 64
    vmov m24, m22
    DELAY 64
    vmov m26, m24
    DELAY 64
    vmov m28, m26
    DELAY 64
    vmov m30, m28
    DELAY 64
    vmov m32, m30
    DELAY 64
    vmov m34, m32
    DELAY 64
    vmov m36, m34
    DELAY 64
    vmov m38, m36
    DELAY 64
    vmov m40, m38
    DELAY 64
    vmov m42, m40
    DELAY 64
    vmov m44, m42
    DELAY 64
    vmov m46, m44
    DELAY 64
    vmov m48, m46
    DELAY 64
    vmov m50, m48
    DELAY 64
    vmov m52, m50
    DELAY 64
    vmov m54, m52
    DELAY 64
    vmov m56, m54
    DELAY 64
    vmov m58, m56
    DELAY 64
    vmov m60, m58
    DELAY 64
    vmov m62, m60
    DELAY 64

# ----------------------------------------------------------------------
# PHASE 3: VSTORE bank 63 back to VMEM and DMA-store for oracle check.
#          After the XLU chain, m63 == T(m0); after the VPU chain that
#          read pair (60, 61) into (62, 63) but did NOT touch bank 63
#          BEFORE the chain wrote (62, 63)... wait, the last VMOV is
#          `VMOV 62, 60` which writes (62, 63). Bank 63 therefore now
#          holds the post-VMOV pair-write copy of m61 (which itself is
#          the symmetric tile after an odd-count XLU chain).
#
#          Because the input tile is symmetric, every bank's bytes are
#          identical to the original tile, so the VSTORE drain bytes
#          equal the original tile bytes.
# ----------------------------------------------------------------------
    LI    x7, 256
    vstore m63, 0(x7)
    DELAY 33

    LI    x8, 256                # VMEM word base of output region
    LI    x10, 0x90001000  # DRAM byte offset
    dma.store x10, x8, x12, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

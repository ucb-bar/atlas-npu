# mreg_cross_engine_full_rotor.S - drive MREG tracker with XLU + VPU traffic
#                                   on every bank index in 0..63
# @TIMEOUT 800000
# @DRAM_BASE 0x00000001
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
    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024              # 1 KiB tile size

# ----------------------------------------------------------------------
# PHASE 0: load the symmetric tile from DRAM into VMEM[0..255], then
#          VLOAD m0 from VMEM bank 0.
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD x6, x1, x12, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33

# ----------------------------------------------------------------------
# PHASE 1: XLU rotor m_k = T(m_(k-1)) for k = 1..63. Each VTRPOSE.XLU
#          touches a single source bank (read) and a single dest bank
#          (write), exercising the tracker's XLU-source bit pattern on
#          EVERY bank index in turn. Consecutive transposes also stress
#          the XLU FSM Idle -> Busy -> Idle handoff 63 times in a row.
# ----------------------------------------------------------------------
    VTRPOSE.XLU  1,  0
    DELAY 64
    VTRPOSE.XLU  2,  1
    DELAY 64
    VTRPOSE.XLU  3,  2
    DELAY 64
    VTRPOSE.XLU  4,  3
    DELAY 64
    VTRPOSE.XLU  5,  4
    DELAY 64
    VTRPOSE.XLU  6,  5
    DELAY 64
    VTRPOSE.XLU  7,  6
    DELAY 64
    VTRPOSE.XLU  8,  7
    DELAY 64
    VTRPOSE.XLU  9,  8
    DELAY 64
    VTRPOSE.XLU 10,  9
    DELAY 64
    VTRPOSE.XLU 11, 10
    DELAY 64
    VTRPOSE.XLU 12, 11
    DELAY 64
    VTRPOSE.XLU 13, 12
    DELAY 64
    VTRPOSE.XLU 14, 13
    DELAY 64
    VTRPOSE.XLU 15, 14
    DELAY 64
    VTRPOSE.XLU 16, 15
    DELAY 64
    VTRPOSE.XLU 17, 16
    DELAY 64
    VTRPOSE.XLU 18, 17
    DELAY 64
    VTRPOSE.XLU 19, 18
    DELAY 64
    VTRPOSE.XLU 20, 19
    DELAY 64
    VTRPOSE.XLU 21, 20
    DELAY 64
    VTRPOSE.XLU 22, 21
    DELAY 64
    VTRPOSE.XLU 23, 22
    DELAY 64
    VTRPOSE.XLU 24, 23
    DELAY 64
    VTRPOSE.XLU 25, 24
    DELAY 64
    VTRPOSE.XLU 26, 25
    DELAY 64
    VTRPOSE.XLU 27, 26
    DELAY 64
    VTRPOSE.XLU 28, 27
    DELAY 64
    VTRPOSE.XLU 29, 28
    DELAY 64
    VTRPOSE.XLU 30, 29
    DELAY 64
    VTRPOSE.XLU 31, 30
    DELAY 64
    VTRPOSE.XLU 32, 31
    DELAY 64
    VTRPOSE.XLU 33, 32
    DELAY 64
    VTRPOSE.XLU 34, 33
    DELAY 64
    VTRPOSE.XLU 35, 34
    DELAY 64
    VTRPOSE.XLU 36, 35
    DELAY 64
    VTRPOSE.XLU 37, 36
    DELAY 64
    VTRPOSE.XLU 38, 37
    DELAY 64
    VTRPOSE.XLU 39, 38
    DELAY 64
    VTRPOSE.XLU 40, 39
    DELAY 64
    VTRPOSE.XLU 41, 40
    DELAY 64
    VTRPOSE.XLU 42, 41
    DELAY 64
    VTRPOSE.XLU 43, 42
    DELAY 64
    VTRPOSE.XLU 44, 43
    DELAY 64
    VTRPOSE.XLU 45, 44
    DELAY 64
    VTRPOSE.XLU 46, 45
    DELAY 64
    VTRPOSE.XLU 47, 46
    DELAY 64
    VTRPOSE.XLU 48, 47
    DELAY 64
    VTRPOSE.XLU 49, 48
    DELAY 64
    VTRPOSE.XLU 50, 49
    DELAY 64
    VTRPOSE.XLU 51, 50
    DELAY 64
    VTRPOSE.XLU 52, 51
    DELAY 64
    VTRPOSE.XLU 53, 52
    DELAY 64
    VTRPOSE.XLU 54, 53
    DELAY 64
    VTRPOSE.XLU 55, 54
    DELAY 64
    VTRPOSE.XLU 56, 55
    DELAY 64
    VTRPOSE.XLU 57, 56
    DELAY 64
    VTRPOSE.XLU 58, 57
    DELAY 64
    VTRPOSE.XLU 59, 58
    DELAY 64
    VTRPOSE.XLU 60, 59
    DELAY 64
    VTRPOSE.XLU 61, 60
    DELAY 64
    VTRPOSE.XLU 62, 61
    DELAY 64
    VTRPOSE.XLU 63, 62
    DELAY 64

# ----------------------------------------------------------------------
# PHASE 2: VPU pair-op rotor. Each VMOV reads pair (b-2, b-1) and
#          writes pair (b, b+1). Going b = 2, 4, ..., 62 walks every
#          even pair across the bank space, so the tracker sees VPU-
#          source reads on (0,1)..(60,61) and VPU-source writes on
#          (2,3)..(62,63).
# ----------------------------------------------------------------------
    VMOV  2,  0
    DELAY 64
    VMOV  4,  2
    DELAY 64
    VMOV  6,  4
    DELAY 64
    VMOV  8,  6
    DELAY 64
    VMOV 10,  8
    DELAY 64
    VMOV 12, 10
    DELAY 64
    VMOV 14, 12
    DELAY 64
    VMOV 16, 14
    DELAY 64
    VMOV 18, 16
    DELAY 64
    VMOV 20, 18
    DELAY 64
    VMOV 22, 20
    DELAY 64
    VMOV 24, 22
    DELAY 64
    VMOV 26, 24
    DELAY 64
    VMOV 28, 26
    DELAY 64
    VMOV 30, 28
    DELAY 64
    VMOV 32, 30
    DELAY 64
    VMOV 34, 32
    DELAY 64
    VMOV 36, 34
    DELAY 64
    VMOV 38, 36
    DELAY 64
    VMOV 40, 38
    DELAY 64
    VMOV 42, 40
    DELAY 64
    VMOV 44, 42
    DELAY 64
    VMOV 46, 44
    DELAY 64
    VMOV 48, 46
    DELAY 64
    VMOV 50, 48
    DELAY 64
    VMOV 52, 50
    DELAY 64
    VMOV 54, 52
    DELAY 64
    VMOV 56, 54
    DELAY 64
    VMOV 58, 56
    DELAY 64
    VMOV 60, 58
    DELAY 64
    VMOV 62, 60
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
    VSTORE 63, x7, 0
    DELAY 33

    LI    x8, 256                # VMEM word base of output region
    LI    x10, 0x1000            # DRAM byte offset
    DMA.STORE x10, x8, x12, 1
    DMA.WAIT 1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

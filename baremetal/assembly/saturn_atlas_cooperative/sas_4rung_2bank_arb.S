# sas_4rung_2bank_arb.S - 4-rung priority ladder across two legacy-named VMEM regions
#
# Legacy b3 region: vector LSU + DMA + Saturn TL (per T1 shape)
# Legacy b4 region: scalar LSU + DMA + Saturn TL (new rung coverage)
#
# Per-bank arbitration runs the static priority ladder:
#   scalar-store > VSTORE > scalar-load > VLOAD > DMA-W > DMA-R > TL-W > TL-R
#
# Under the current 256 KiB banking, the symbolic b3/b4 regions again map to
# distinct physical banks. The host driver still refers to them symbolically,
# so the labels are retained here.
#
# Legacy region base words / current decode:
#   b3 base word = 196608 -> current bank 3
#   b4 base word = 262144 -> current bank 4
#
# Sub-region layout:
#   b3 +0x000 (word 196608)   vector LSU (1 KiB / 1 mreg tile)
#   b3 +0x400 (word 196864)   DMA ch0    (1 KiB)
#   b3 +0x800 (word 197120)   Saturn TL  (1 KiB, host-side only)
#   b4 +0x000 (byte 0x100000) scalar LSU (32 words used in pattern)
#   b4 +0x400 (word 262400)   DMA ch1    (1 KiB)
#   b4 +0x800 (word 262656)   Saturn TL  (1 KiB, host-side only)
#
# DRAM (DMA base 0x00000001):
#   IN_b3:  0x0000  (1024 B, legacy b3-region DMA input tile)
#   IN_b4:  0x0400  (1024 B, legacy b4-region DMA input tile)
#   OUT_b3: 0x2000  (1024 B, legacy b3-region DMA output, expected = IN_b3)
#   OUT_b4: 0x2400  (1024 B, legacy b4-region DMA output, expected = IN_b4)
#
# Outer 4 x inner 128 = 512 LSU+DMA cycles, ~250K Atlas cycles.

    LI    x5, 0x00000000  # upper 32 bits of 64-bit DRAM pointer (HW: Cat(dmaBaseReg, rs1/rd))
    DMA.CONFIG x5, 0
    LI    x12, 1024              # DMA transfer size

    LI    x27, 4                 # outer round cap
    LI    x26, 0                 # outer round counter

outer:
    LI    x13, 128               # inner iter cap
    LI    x11, 0                 # inner iter counter

inner:
    # ── DMA.LOAD on both legacy regions (different channels; banks 3 and 4) ──
    LI    x6,  196864            # legacy b3 DMA word (current bank 3)
    LI    x9, 0x90000000  # DRAM IN_b3
    DMA.LOAD x6, x9, x12, 0
    LI    x7,  262400            # legacy b4 DMA word (current bank 4)
    LI    x10, 0x90000400  # DRAM IN_b4
    DMA.LOAD x7, x10, x12, 1

    # ── Legacy b3 vector LSU (mreg 0; current bank 3) ──
    LI    x6,  196608
    VLOAD 0, x6, 0
    DELAY 33

    # ── Legacy b4 scalar LSU: load byte 0, store to byte 8 (current bank 4) ──
    LI    x16, 0x100000          # legacy b4 byte 0 (VMEM-relative)
    LW    x17, x16, 0
    DELAY 2
    SW    x17, x16, 8
    DELAY 1

    VSTORE 0, x6, 0
    DELAY 33

    DMA.WAIT 0
    DMA.WAIT 1

    # ── DMA.STORE on both legacy regions ──
    LI    x6,  196864
    LI    x9, 0x90002000  # DRAM OUT_b3
    DMA.STORE x9, x6, x12, 0
    LI    x7,  262400
    LI    x10, 0x90002400  # DRAM OUT_b4
    DMA.STORE x10, x7, x12, 1

    # ── Legacy b3 vector LSU (mreg 1; current bank 3) ──
    LI    x6,  196608
    VLOAD 1, x6, 0
    DELAY 33

    # ── Legacy b4 scalar LSU: load byte 4, store to byte 12 (current bank 4) ──
    LI    x16, 0x100000
    LW    x17, x16, 4
    DELAY 2
    SW    x17, x16, 12
    DELAY 1

    VSTORE 1, x6, 0
    DELAY 33

    DMA.WAIT 0
    DMA.WAIT 1

    ADDI  x11, x11, 1
    BLT   x11, x13, inner
    NOP

    ADDI  x26, x26, 1
    BLT   x26, x27, outer
    NOP

# ── Done: dbg0 = 1, ECALL ─────────────────────────────────────────────
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

# sas_vmem_dram_dual.S - Atlas kernel for T5 (saturn_atlas_vmem_dram_dual)
#
# T5 closes the Saturn dual-target coverage gap: Saturn issues
# concurrent vload/vstore to BOTH VMEM (legacy b3 region + 0x1000 =
# byte 0xC1000, current bank 3, 4 KiB) and a DRAM ring buffer (64 KiB
# rotated 4 KiB per host iter), while Atlas runs LSU + DMA traffic on
# the two legacy b3/b4 regions below.
#
# Atlas kernel shape == sas_4rung_2bank_arb.S (T1.5):
#   legacy b3 region: vector LSU + DMA + Saturn TL (host-side @ +0x1000)
#   legacy b4 region: scalar LSU + DMA
#
# This file is intentionally a decoupled copy of sas_4rung_2bank_arb.S
# so future T5 tweaks (e.g., shifting toward T2-style 8-channel pump)
# don't perturb T1.5's calibration baseline.
#
# Per-bank arbitration runs the static priority ladder:
#   scalar-store > VSTORE > scalar-load > VLOAD > DMA-W > DMA-R
#                                        > TL-W > TL-R
#
# Under the current 256 KiB banking, the symbolic b3/b4 regions again map to
# distinct physical banks. The legacy labels are retained because the host
# driver and golden files still use them symbolically.
#
# Legacy region base words / current decode:
#   b3 base word = 196608 -> current bank 3
#   b4 base word = 262144 -> current bank 4
#
# Sub-region layout:
#   b3 +0x000 (word 196608)     vector LSU (1 KiB / 1 mreg tile)
#   b3 +0x400 (word 196864)     DMA ch0    (1 KiB)
#   b3 +0x800 (word 197120)     reserved (T1 host TL region, unused here)
#   b3 +0x1000                  Saturn VMEM RVV region (4 KiB, host-side)
#   b4 +0x000 (byte 0x100000)   scalar LSU (32 words used in pattern)
#   b4 +0x400 (word 262400)     DMA ch1    (1 KiB)
#
# DRAM (DMA base 0x00000001):
#   IN_b3:  0x0000  (1024 B, legacy b3-region DMA input tile)
#   IN_b4:  0x0400  (1024 B, legacy b4-region DMA input tile)
#   OUT_b3: 0x2000  (1024 B, legacy b3-region DMA output, expected = IN_b3)
#   OUT_b4: 0x2400  (1024 B, legacy b4-region DMA output, expected = IN_b4)
#
# Saturn DRAM ring lives in a static .bss array placed by the linker;
# its address is host-runtime, disjoint from the DMA window above.
#
# Outer 4 x inner 128 = 512 LSU+DMA cycles, ~1.5-2 M Atlas cycles.

    LI    x5, 0x00000000
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
    DELAY 40

    # ── Legacy b4 scalar LSU: load byte 0, store to byte 8 (current bank 4) ──
    LI    x16, 0x100000          # legacy b4 byte 0 (VMEM-relative)
    LW    x17, x16, 0
    DELAY 2
    SW    x17, x16, 8
    DELAY 1

    VSTORE 0, x6, 0
    DELAY 40

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
    DELAY 40

    # ── Legacy b4 scalar LSU: load byte 4, store to byte 12 (current bank 4) ──
    LI    x16, 0x100000
    LW    x17, x16, 4
    DELAY 2
    SW    x17, x16, 12
    DELAY 1

    VSTORE 1, x6, 0
    DELAY 40

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

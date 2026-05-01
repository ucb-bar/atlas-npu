# sas_vmem_dram_dual.S - Atlas kernel for T5 (saturn_atlas_vmem_dram_dual)
#
# T5 closes the Saturn dual-target coverage gap: Saturn issues
# concurrent vload/vstore to BOTH VMEM (bank 3 + 0x1000, 4 KiB) and
# a DRAM ring buffer (64 KiB rotated 4 KiB per host iter), while
# Atlas runs LSU + DMA traffic on banks 3 and 4.
#
# Atlas kernel shape == sas_4rung_2bank_arb.S (T1.5):
#   bank 3: vector LSU + DMA + Saturn TL (host-side @ +0x1000)
#   bank 4: scalar LSU + DMA
#
# This file is intentionally a decoupled copy of sas_4rung_2bank_arb.S
# so future T5 tweaks (e.g., shifting toward T2-style 8-channel pump)
# don't perturb T1.5's calibration baseline.
#
# Per-bank arbitration runs the static priority ladder:
#   scalar-store > VSTORE > scalar-load > VLOAD > DMA-W > DMA-R
#                                        > TL-W > TL-R
#
# Two-bank split is mandatory: VMEM.scala asserts on same-bank
# scalar/vector LSU collisions. T1 already proves that a TL master
# (Saturn) hitting the same bank as Atlas LSU is legal — Saturn
# enters via the VMEM TL slave port, not the Atlas LSU port.
#
# VMEM bank base words (line index = word_idx >> 3, bank = line[14:12]):
#   bank 3 base word = 3 * 32768 = 98304
#   bank 4 base word = 4 * 32768 = 131072
#
# Sub-region layout (Atlas-visible only; Saturn @ +0x1000 is host-driven):
#   bank 3 +0x000 (word 98304)  vector LSU (1 KiB / 1 mreg tile)
#   bank 3 +0x400 (word 98560)  DMA ch0    (1 KiB)
#   bank 3 +0x800 (word 98816)  reserved (T1 host TL region, unused here)
#   bank 3 +0x1000              Saturn VMEM RVV region (4 KiB, host-side)
#   bank 4 +0x000 (byte 0x80000) scalar LSU (32 words used in pattern)
#   bank 4 +0x400 (word 131328) DMA ch1    (1 KiB)
#
# DRAM (DMA base 0x90000000):
#   IN_b3:  0x0000  (1024 B, bank 3 DMA input tile)
#   IN_b4:  0x0400  (1024 B, bank 4 DMA input tile)
#   OUT_b3: 0x2000  (1024 B, bank 3 DMA output, expected = IN_b3)
#   OUT_b4: 0x2400  (1024 B, bank 4 DMA output, expected = IN_b4)
#
# Saturn DRAM ring lives in a static .bss array placed by the linker;
# its address is host-runtime, disjoint from the DMA window above.
#
# Outer 4 x inner 128 = 512 LSU+DMA cycles, ~1.5-2 M Atlas cycles.

    LI    x5,  0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 1024              # DMA transfer size

    LI    x27, 4                 # outer round cap
    LI    x26, 0                 # outer round counter

outer:
    LI    x13, 128               # inner iter cap
    LI    x11, 0                 # inner iter counter

inner:
    # ── DMA.LOAD on both banks (different channels) ──
    LI    x6,  98560             # bank 3 DMA word
    LI    x9,  0x0000            # DRAM IN_b3
    DMA.LOAD x6, x9, x12, 0
    LI    x7,  131328            # bank 4 DMA word
    LI    x10, 0x0400            # DRAM IN_b4
    DMA.LOAD x7, x10, x12, 1

    # ── Bank 3 vector LSU (mreg 0) ──
    LI    x6,  98304
    VLOAD 0, x6, 0
    DELAY 40

    # ── Bank 4 scalar LSU: load byte 0, store to byte 8 (proves LW fired) ──
    LI    x16, 0x80000           # bank 4 byte 0 (VMEM-relative)
    LW    x17, x16, 0
    DELAY 2
    SW    x17, x16, 8
    DELAY 1

    VSTORE 0, x6, 0
    DELAY 40

    DMA.WAIT 0
    DMA.WAIT 1

    # ── DMA.STORE on both banks ──
    LI    x6,  98560
    LI    x9,  0x2000            # DRAM OUT_b3
    DMA.STORE x9, x6, x12, 0
    LI    x7,  131328
    LI    x10, 0x2400            # DRAM OUT_b4
    DMA.STORE x10, x7, x12, 1

    # ── Bank 3 vector LSU (mreg 1) ──
    LI    x6,  98304
    VLOAD 1, x6, 0
    DELAY 40

    # ── Bank 4 scalar LSU: load byte 4, store to byte 12 ──
    LI    x16, 0x80000
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
    NOP

    ADDI  x26, x26, 1
    BLT   x26, x27, outer
    NOP
    NOP

# ── Done: dbg0 = 1, ECALL ─────────────────────────────────────────────
    LI    x30, 1
    CSRW  x30, 0xC10
    ECALL

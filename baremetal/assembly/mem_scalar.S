# mem_scalar.S — Scalar load/store and SELD functional test
# @TIMEOUT 10000
# @DRAM_BASE 0x90000000
#
# All scalar VMEM access goes through the LSU.
# Scalar loads now have a 3-cycle software-visible latency, and scalar stores
# are pipelined masked writes, so this test spaces load/use and store->load
# checks accordingly.
#
# VMEM byte layout after DMA load (little-endian):
#   byte 0=0x01, 1=0xA2, 2=0x84, 3=0xFF
#   word0 = 0xFF84A201
#   bytes 4-31 = 0x00
#
# @DRAM 0  0x00000000000000000000000000000000000000000000000000000000FF84A201
#
# After all stores:
#   SW 0xAABBCCDD at byte 0: bytes 0-3 = [DD CC BB AA]
#   SB 0x42       at byte 1: byte 1 = 42  -> [DD 42 BB AA]
#   SH 0x1234     at byte 2: bytes 2-3 = [34 12] -> [DD 42 34 12]
# @CHECK_DRAM 32 0x00000000000000000000000000000000000000000000000000000000123442DD

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 32

# ── Step 1: DMA load DRAM[0] -> VMEM[word 0] ─────────────
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x28, x28, 1       # x28 = 2

# ── Test 2: LW byte 0 = 0xFF84A201 ──────────────────────
    ADDI  x8, x0, 0
    lw x9, 0(x8)
    DELAY 2
    LI    x30, 0xFF84A201
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 3

# ── Test 3: LB byte 0 = sign_ext(0x01) = 1 ──────────────
    lb x9, 0(x8)
    DELAY 2
    ADDI  x30, x0, 1
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 4

# ── Test 4: LB byte 2 = sign_ext(0x84) = 0xFFFFFF84 ─────
    lb x9, 2(x8)
    DELAY 2
    LI    x30, 0xFFFFFF84
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 5

# ── Test 5: LBU byte 2 = 0x84 ───────────────────────────
    lbu x9, 2(x8)
    DELAY 2
    ADDI  x30, x0, 0x84
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 6

# ── Test 6: LH byte 0 = sign_ext(0xA201) = 0xFFFFA201 ───
    lh x9, 0(x8)
    DELAY 2
    LI    x30, 0xFFFFA201
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 7

# ── Test 7: LHU byte 0 = 0xA201 ─────────────────────────
    lhu x9, 0(x8)
    DELAY 2
    LI    x30, 0xA201
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 8

# ── Test 8: SELD byte 0 -> e[3] (no crash) ──────────────
    seld e3, x8
    DELAY 2
    ADDI  x28, x28, 1       # x28 = 9

# ── Test 9: SW then immediate LW (RAW hazard test) ──────
    LI    x10, 0xAABBCCDD
    sw x10, 0(x8)
    lw x9, 0(x8)
    DELAY 2
    LI    x30, 0xAABBCCDD
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 10

# ── Test 10: SB then LBU at same byte ───────────────────
    ADDI  x11, x0, 0x42
    sb x11, 1(x8)
    lbu x9, 1(x8)
    DELAY 2
    ADDI  x30, x0, 0x42
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 11

# ── Test 11: SH then LHU at byte 2 ──────────────────────
    LI    x12, 0x1234
    sh x12, 2(x8)
    lhu x9, 2(x8)
    DELAY 2
    LI    x30, 0x1234
    BNE   x9, x30, fail
    ADDI  x28, x28, 1       # x28 = 12

# ── Step 12: DMA store final VMEM -> DRAM word 32 ───────
    ADDI  x6, x0, 0
    LI    x3, 0x90000400
    dma.store x3, x6, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL
    

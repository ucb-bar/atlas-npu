# saturn_atlas_two_phase.S — Saturn ↔ Atlas cooperative kernel (ECALL pause/resume).
#
# PC resets to 0 on every execControl=1 write. This kernel reads dbg1 at
# entry and dispatches between phases so a single IMEM-resident program
# covers both halves of the ECALL pause / host-work / resume protocol.
#
# Phase 0 (dbg1 == 0):
#   Seed VMEM[i*4] = 0xA0000000 | i for i in 0..15.
#   Write dbg0 = 1, ECALL.
#
# Between phases, the host (Saturn on Shuttle):
#   - reads VMEM via MMIO at ATLAS_VMEM_BASE (0x20000000) and verifies seed,
#   - transforms each word: VMEM[i*4] += 0x10000,
#   - writes dbg0 = 0, dbg1 = 1, execControl = 1.
#
# Phase 1 (dbg1 == 1):
#   Read VMEM[i*4] and verify equals (0xA0010000 | i).
#   On full success: dbg0 = 2, ECALL.
#   On mismatch at index i: dbg0 = 100 + i, ECALL.
#
# @TIMEOUT 200000

_start:
    CSRR  x1, 0xC11              # x1 = dbg1 (phase selector)
    BEQ   x1, x0, phase_0
    NOP
    NOP
    JAL   x0, phase_1
    NOP
    NOP

phase_0:
    LI    x2, 0xA0000000         # seed base
    ADDI  x3, x0, 0              # VMEM byte addr
    ADDI  x4, x0, 0              # i
    ADDI  x5, x0, 16             # count

p0_loop:
    OR    x6, x2, x4             # data = seed_base | i
    SW    x6, x3, 0              # VMEM[addr] = data
    DELAY 1
    ADDI  x3, x3, 4
    ADDI  x4, x4, 1
    BLT   x4, x5, p0_loop
    NOP
    NOP

    ADDI  x1, x0, 1
    CSRW  x1, 0xC10              # dbg0 = 1 (phase 0 done)
    ECALL

phase_1:
    LI    x2, 0xA0010000         # expected base = seed + 0x10000
    ADDI  x3, x0, 0
    ADDI  x4, x0, 0
    ADDI  x5, x0, 16

p1_loop:
    LW    x6, x3, 0              # got
    DELAY 1
    OR    x7, x2, x4             # expected = base | i
    BNE   x6, x7, p1_fail
    NOP
    NOP
    ADDI  x3, x3, 4
    ADDI  x4, x4, 1
    BLT   x4, x5, p1_loop
    NOP
    NOP

    ADDI  x1, x0, 2
    CSRW  x1, 0xC10              # dbg0 = 2 (PASS)
    ECALL

p1_fail:
    ADDI  x8, x4, 100            # dbg0 = 100 + failing i
    CSRW  x8, 0xC10
    ECALL

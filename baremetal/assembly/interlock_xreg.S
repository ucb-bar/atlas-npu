# interlock_xreg.S — the scalar-load interlock waits only for the x-registers an instruction reads
# @TIMEOUT 2000
#
# A scalar load writes rd 3 cycles after issue. An instruction that reads that register, or writes
# it, waits; stall_data_xreg (CSR 0xCF4) counts the cycles. Fields that hold an immediate, an
# e-register or an m-register name no x-register, so they do not wait.
# DBG0 = 1 if all checks pass, else the number of the failed check:
#   2  seli and delay right after a load to the register that their immediate bits name in the
#      rs1 field (x3) and in the rs2 field (x4): no stall
#   3  lui (immediate bits in the rs1 field name x5) and vload (vd bits in the rd field name x6)
#      right after a load to that register: no stall
#   4  addi that reads the loaded register: it waits 3 cycles

# ── 2: seli and delay ──
    ADDI  x28, x0, 2
    CSRRS x10, 0xCF4, x0
    LW    x3, 0(x0)
    seli  e1, 1
    DELAY 1
    LW    x4, 0(x0)
    seli  e1, 0x41
    DELAY 0x41
    CSRRS x11, 0xCF4, x0
    BNE   x11, x10, fail

# ── 3: lui and vload ──
    ADDI  x28, x28, 1
    LW    x5, 0(x0)
    LUI   x7, 0x28
    LW    x6, 0(x0)
    vload m6, 0(x0)
    DELAY 40
    CSRRS x11, 0xCF4, x0
    BNE   x11, x10, fail

# ── 4: a load-use stall ──
    ADDI  x28, x28, 1
    LW    x8, 0(x0)
    ADDI  x9, x8, 0
    CSRRS x11, 0xCF4, x0
    SUB   x12, x11, x10
    ADDI  x13, x0, 3
    BNE   x12, x13, fail

pass:
    ADDI  x2, x0, 1
    CSRRW x0, 0x800, x2
    ECALL

fail:
    CSRRW x0, 0x801, x11
    CSRRW x0, 0x800, x28
    ECALL

# csr_isa_perf.S — the scalar core's ISA CSR, performance-counter CSRs and pure CSR reads
# @TIMEOUT 2000
#
# DBG1 = the ISA CSR (0xCC0). DBG0 = 1 if all checks pass, else the number of the failed check:
#   2  ISA CSR: version 1.0, hardware-interlock and inner-product-tree bits set, bits 15:4 zero
#   3  csrrc with x0, csrrsi and csrrci with 0 read the ISA CSR as csrrs with x0 does
#   4  PERF_NUM (0xCC1) is 27
#   5  two back-to-back csrrs of cycle (0xC00) differ by 1: a read with x0 does not write
#   6  the same with csrrsi and csrrci with 0
#   7  two back-to-back csrrs of instret (0xC01) differ by 1
#   8  counter cycles (0xCC2) differs by 1 back to back, its bits 63:32 (0xCC3) are 0, and
#      counter instret (0xCC4) differs by 3 over two NOPs
#   9  a transpose behind a busy XLU stalls: stall_unit_xlu (0xCE8) counts the stalled cycles, and
#      stall_unit (0xCC6) counts the same cycles
#  10  a load-use stall: stall_data_xreg (0xCF4) counts the stalled cycles, and stall_data (0xCC8)
#      counts the same cycles

# ── 2: ISA CSR ──
    ADDI  x28, x0, 2
    CSRRS x1, 0xCC0, x0
    SRLI  x2, x1, 16
    LI    x3, 0x0100
    BNE   x2, x3, fail
    NOP
    ANDI  x2, x1, 5
    ADDI  x3, x0, 5
    BNE   x2, x3, fail
    NOP
    LI    x3, 0xFFF0
    AND   x2, x1, x3
    BNE   x2, x0, fail
    NOP

# ── 3: the other read forms ──
    ADDI   x28, x28, 1
    CSRRC  x2, 0xCC0, x0
    BNE    x2, x1, fail
    NOP
    CSRRSI x2, 0xCC0, 0
    BNE    x2, x1, fail
    NOP
    CSRRCI x2, 0xCC0, 0
    BNE    x2, x1, fail
    NOP

# ── 4: PERF_NUM ──
    ADDI  x28, x28, 1
    CSRRS x2, 0xCC1, x0
    ADDI  x3, x0, 27
    BNE   x2, x3, fail
    NOP

# ── 5: cycle, csrrs with x0 ──
    ADDI  x28, x28, 1
    ADDI  x5, x0, 1
    CSRRS x2, 0xC00, x0
    CSRRS x3, 0xC00, x0
    SUB   x4, x3, x2
    BNE   x4, x5, fail
    NOP

# ── 6: cycle, csrrsi / csrrci with 0 ──
    ADDI   x28, x28, 1
    CSRRSI x2, 0xC00, 0
    CSRRCI x3, 0xC00, 0
    SUB    x4, x3, x2
    BNE    x4, x5, fail
    NOP

# ── 7: instret ──
    ADDI  x28, x28, 1
    CSRRS x2, 0xC01, x0
    CSRRS x3, 0xC01, x0
    SUB   x4, x3, x2
    BNE   x4, x5, fail
    NOP

# ── 8: counters cycles and instret ──
    ADDI  x28, x28, 1
    CSRRS x2, 0xCC2, x0
    CSRRS x3, 0xCC2, x0
    SUB   x4, x3, x2
    BNE   x4, x5, fail
    NOP
    CSRRS x2, 0xCC3, x0
    BNE   x2, x0, fail
    NOP
    CSRRS x2, 0xCC4, x0
    NOP
    NOP
    CSRRS x3, 0xCC4, x0
    SUB   x4, x3, x2
    ADDI  x6, x0, 3
    BNE   x4, x6, fail
    NOP

# ── 9: unit stall on the XLU ──
    ADDI  x28, x28, 1
    CSRRS x10, 0xCE8, x0
    CSRRS x11, 0xCC6, x0
    vtrpose.xlu m1, m0
    vtrpose.xlu m3, m2
    CSRRS x12, 0xCE8, x0
    CSRRS x13, 0xCC6, x0
    SUB   x14, x12, x10
    SUB   x15, x13, x11
    BEQ   x14, x0, fail
    NOP
    BNE   x14, x15, fail
    NOP

# ── 10: data stall on a scalar load ──
    ADDI  x28, x28, 1
    CSRRS x10, 0xCF4, x0
    CSRRS x11, 0xCC8, x0
    LW    x7, 0(x0)
    ADDI  x8, x7, 0
    CSRRS x12, 0xCF4, x0
    CSRRS x13, 0xCC8, x0
    SUB   x14, x12, x10
    SUB   x15, x13, x11
    BEQ   x14, x0, fail
    NOP
    BNE   x14, x15, fail
    NOP

pass:
    CSRRW x0, 0xC11, x1
    ADDI  x2, x0, 1
    CSRRW x0, 0xC10, x2
    ECALL

fail:
    CSRRW x0, 0xC11, x1
    CSRRW x0, 0xC10, x28
    ECALL

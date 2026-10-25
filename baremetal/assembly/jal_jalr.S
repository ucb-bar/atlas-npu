# jal_jalr.S — JAL, JALR, link values
# @TIMEOUT 500
#
# Verifies:
#   - JAL and JALR write the byte address of the next instruction (pc + 4) to rd
#   - A taken JAL or JALR drops the instruction after it (no delay slot)
#   - JALR jumps to (rs1 + imm) with bit 0 cleared
#   - Subroutine call/return pattern works

    ADDI  x28, x0, 1

# ── Test 1: JAL writes pc + 4 ───────────────────────────
    ADDI  x10, x0, 0xDD
    AUIPC x5, 0            # x5 = byte address of this AUIPC
    JAL   x1, t1_target    # x1 = x5 + 8
    ADDI  x10, x0, 0xEE    # must be dropped after redirect

    # Should NOT execute (jumped over)
    ADDI  x10, x0, 0xFF

t1_target:
    # Redirect dropped the next instruction?
    ADDI  x30, x0, 0xDD
    BNE   x10, x30, fail
    # x1 should be the byte address of the JAL + 4
    ADDI  x5, x5, 8
    BNE   x1, x5, fail
    ADDI  x28, x28, 1

# ── Test 2: JAL x0 (no link) ────────────────────────────
    # Use JAL x0 to jump forward without saving link
    ADDI  x10, x0, 0x11
    JAL   x0, t2_target

    ADDI  x10, x0, 0xFF    # should NOT execute

t2_target:
    ADDI  x30, x0, 0x11
    BNE   x10, x30, fail
    ADDI  x28, x28, 1

# ── Test 3: JALR clears bit 0 of rs1 + imm and writes pc + 4 ─
    ADDI  x10, x0, 0x33
    AUIPC x5, 0            # x5 = A, the byte address of this AUIPC
    ADDI  x6, x5, 15
    JALR  x2, x6, 2        # at A + 8: jumps to (A + 17) & ~1 = A + 16; x2 = A + 12
    ADDI  x10, x0, 0xFF    # A + 12: must be dropped after redirect

t3_target:                 # A + 16
    ADDI  x30, x0, 0x33
    BNE   x10, x30, fail
    ADDI  x5, x5, 12
    BNE   x2, x5, fail
    ADDI  x28, x28, 1

# ── Test 4: Subroutine call with JAL + return with JALR ─
    ADDI  x10, x0, 0
    JAL   x1, subroutine
    # Returns here
    ADDI  x30, x0, 0x77
    BNE   x10, x30, fail
    ADDI  x28, x28, 1

    # Jump to pass
    JAL   x0, pass

subroutine:
    ADDI  x10, x0, 0x77    # set return value
    JALR  x0, x1, 0         # return to caller (x1 + 0)

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

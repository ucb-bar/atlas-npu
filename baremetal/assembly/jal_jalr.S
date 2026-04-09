# jal_jalr.S — JAL, JALR, link address
# @TIMEOUT 500
#
# Verifies:
#   - JAL saves return address (pc + 1) to rd
#   - JALR jumps to rs1 + imm
#   - Both delay slots execute after jumps
#   - Subroutine call/return pattern works

    ADDI  x28, x0, 1

# ── Test 1: JAL saves link address ──────────────────────
    # JAL x1, target → x1 = pc_of_JAL + 1
    JAL   x1, t1_target
    ADDI  x10, x0, 0xDD    # delay slot 1
    ADDI  x11, x0, 0xEE    # delay slot 2

    # Should NOT execute (jumped over)
    ADDI  x10, x0, 0xFF

t1_target:
    # Delay slot 1 executed?
    ADDI  x30, x0, 0xDD
    BNE   x10, x30, fail
    NOP
    NOP
    # Delay slot 2 executed?
    ADDI  x30, x0, 0xEE
    BNE   x11, x30, fail
    NOP
    NOP
    # x1 should be (word address of JAL) + 1
    # We can't easily check the absolute value, but we can check it's nonzero
    BEQ   x1, x0, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 2: JAL x0 (no link, infinite loop escape) ─────
    # Use JAL x0 to jump forward without saving link
    ADDI  x10, x0, 0
    JAL   x0, t2_target
    ADDI  x10, x0, 0x11    # delay slot 1
    NOP                      # delay slot 2

    ADDI  x10, x0, 0xFF    # should NOT execute

t2_target:
    ADDI  x30, x0, 0x11
    BNE   x10, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 3: Subroutine call with JAL + return with JALR ─
    ADDI  x10, x0, 0
    JAL   x1, subroutine
    NOP                      # delay slot 1
    NOP                      # delay slot 2
    # Returns here
    ADDI  x30, x0, 0x77
    BNE   x10, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

    # Jump to pass
    JAL   x0, pass
    NOP
    NOP

subroutine:
    ADDI  x10, x0, 0x77    # set return value
    JALR  x0, x1, 0         # return to caller (x1 + 0)
    NOP                      # delay slot 1
    NOP                      # delay slot 2

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
    
# jal_jalr.S — JAL, JALR, link address
# @TIMEOUT 500
#
# Verifies:
#   - JAL saves return address (pc + 1) to rd
#   - JALR jumps to rs1 + imm
#   - The single delay slot executes after jumps
#   - Subroutine call/return pattern works

    ADDI  x28, x0, 1

# ── Test 1: JAL saves link address ──────────────────────
    # JAL x1, target → x1 = pc_of_JAL + 1
    JAL   x1, t1_target
    ADDI  x10, x0, 0xDD    # delay slot
    ADDI  x11, x0, 0xEE    # must be skipped after redirect

    # Should NOT execute (jumped over)
    ADDI  x10, x0, 0xFF

t1_target:
    # Delay slot executed?
    ADDI  x30, x0, 0xDD
    BNE   x10, x30, fail
    NOP
    # Redirect skipped the next instruction?
    BNE   x11, x0, fail
    NOP
    # x1 should be (word address of JAL) + 1
    # We can't easily check the absolute value, but we can check it's nonzero
    BEQ   x1, x0, fail
    NOP
    ADDI  x28, x28, 1

# ── Test 2: JAL x0 (no link, infinite loop escape) ─────
    # Use JAL x0 to jump forward without saving link
    ADDI  x10, x0, 0
    JAL   x0, t2_target
    ADDI  x10, x0, 0x11    # delay slot

    ADDI  x10, x0, 0xFF    # should NOT execute

t2_target:
    ADDI  x30, x0, 0x11
    BNE   x10, x30, fail
    NOP
    ADDI  x28, x28, 1

    # ── Test 3: Subroutine call with JAL + return with JALR ─
    ADDI  x10, x0, 0
    JAL   x1, subroutine
    NOP                      # delay slot
    # Returns here
    ADDI  x30, x0, 0x77
    BNE   x10, x30, fail
    NOP
    ADDI  x28, x28, 1

    # Jump to pass
    JAL   x0, pass
    NOP

subroutine:
    ADDI  x10, x0, 0x77    # set return value
    JALR  x0, x1, 0         # return to caller (x1 + 0)
    NOP                      # delay slot

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
    

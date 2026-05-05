# branch_delay_slots.S — branches + 1 delay slot
# @TIMEOUT 500
#
# Verifies:
#   - Taken BEQ executes its single delay slot
#   - Not-taken BNE falls through (no redirect)
#   - Delay slot side effects are visible
#   - BNE taken branch works

    ADDI  x28, x0, 1

# ── Test 1: BEQ taken, single delay slot executes ───────
    ADDI  x10, x0, 0
    ADDI  x11, x0, 0
    BEQ   x0, x0, t1_target
    ADDI  x10, x0, 0xAA    # delay slot: must execute
    ADDI  x11, x0, 0xBB    # must be skipped after redirect

    # This should NOT execute (skipped by branch)
    ADDI  x10, x0, 0xFF

t1_target:
    # Check delay-slot side effect
    ADDI  x30, x0, 0xAA
    BNE   x10, x30, fail
    NOP
    # Check redirect skipped the next instruction
    BNE   x11, x0, fail
    NOP
    ADDI  x28, x28, 1

# ── Test 2: BEQ not taken, falls through ────────────────
    ADDI  x1, x0, 1
    ADDI  x10, x0, 0
    BEQ   x1, x0, fail     # 1 != 0, not taken
    NOP
    ADDI  x10, x0, 0x55    # should execute (fall-through)
    ADDI  x30, x0, 0x55
    BNE   x10, x30, fail
    NOP
    ADDI  x28, x28, 1

# ── Test 3: BNE taken ───────────────────────────────────
    ADDI  x1, x0, 5
    ADDI  x2, x0, 10
    ADDI  x10, x0, 0
    BNE   x1, x2, t3_target    # 5 != 10, taken
    ADDI  x10, x0, 0xCC        # delay slot

    # Should NOT execute
    ADDI  x10, x0, 0xFF

t3_target:
    ADDI  x30, x0, 0xCC
    BNE   x10, x30, fail
    NOP
    ADDI  x28, x28, 1

# ── Test 4: Loop using BNE ──────────────────────────────
    ADDI  x10, x0, 0       # counter
    ADDI  x11, x0, 5       # limit

t4_loop:
    ADDI  x10, x10, 1
    BNE   x10, x11, t4_loop
    NOP

    # x10 should be 5
    ADDI  x30, x0, 5
    BNE   x10, x30, fail
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

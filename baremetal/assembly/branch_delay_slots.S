# branch_delay_slots.S — taken branches have no delay slot
# @TIMEOUT 500
#
# Verifies:
#   - Taken BEQ drops the instruction after it
#   - Not-taken BEQ falls through (no redirect)
#   - Work before a taken branch is visible at the target
#   - BNE taken branch works

    ADDI  x28, x0, 1

# ── Test 1: BEQ taken drops the next instruction ───────
    ADDI  x10, x0, 0
    ADDI  x11, x0, 0
    ADDI  x10, x0, 0xAA    # work before the branch
    BEQ   x0, x0, t1_target
    ADDI  x11, x0, 0xBB    # must be dropped after redirect

    # This should NOT execute (skipped by branch)
    ADDI  x10, x0, 0xFF

t1_target:
    # Check the work before the branch
    ADDI  x30, x0, 0xAA
    BNE   x10, x30, fail
    # Check redirect dropped the next instruction
    BNE   x11, x0, fail
    ADDI  x28, x28, 1

# ── Test 2: BEQ not taken, falls through ────────────────
    ADDI  x1, x0, 1
    ADDI  x10, x0, 0
    BEQ   x1, x0, fail     # 1 != 0, not taken
    ADDI  x10, x0, 0x55    # should execute (fall-through)
    ADDI  x30, x0, 0x55
    BNE   x10, x30, fail
    ADDI  x28, x28, 1

# ── Test 3: BNE taken ───────────────────────────────────
    ADDI  x1, x0, 5
    ADDI  x2, x0, 10
    ADDI  x10, x0, 0
    ADDI  x10, x0, 0xCC        # work before the branch
    BNE   x1, x2, t3_target    # 5 != 10, taken

    # Should NOT execute
    ADDI  x10, x0, 0xFF

t3_target:
    ADDI  x30, x0, 0xCC
    BNE   x10, x30, fail
    ADDI  x28, x28, 1

# ── Test 4: Loop using BNE ──────────────────────────────
    ADDI  x10, x0, 0       # counter
    ADDI  x11, x0, 5       # limit

t4_loop:
    ADDI  x10, x10, 1
    BNE   x10, x11, t4_loop

    # x10 should be 5
    ADDI  x30, x0, 5
    BNE   x10, x30, fail

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

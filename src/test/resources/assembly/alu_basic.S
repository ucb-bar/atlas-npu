# alu_basic.S — ALU operations
# @TIMEOUT 500
#
# Tests: ADD, SUB, AND, OR, XOR, SLT, SLL, SRL,
#        ADDI, LUI

    ADDI  x28, x0, 1       # test case counter (starts at 1)

# ── Test 1: ADDI ─────────────────────────────────────────
    ADDI  x31, x0, 42
    ADDI  x30, x0, 42
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 2: ADD ──────────────────────────────────────────
    ADDI  x1, x0, 30
    ADDI  x2, x0, 70
    ADD   x31, x1, x2
    ADDI  x30, x0, 100
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 3: SUB ──────────────────────────────────────────
    SUB   x31, x2, x1      # 70 - 30 = 40
    ADDI  x30, x0, 40
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 4: AND ──────────────────────────────────────────
    ADDI  x1, x0, 0xFF
    ADDI  x2, x0, 0x0F
    AND   x31, x1, x2
    ADDI  x30, x0, 0x0F
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 5: OR ───────────────────────────────────────────
    ADDI  x1, x0, 0xA0
    ADDI  x2, x0, 0x05
    OR    x31, x1, x2
    ADDI  x30, x0, 0xA5
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 6: XOR ──────────────────────────────────────────
    ADDI  x1, x0, 0xFF
    ADDI  x2, x0, 0x0F
    XOR   x31, x1, x2
    ADDI  x30, x0, 0xF0
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 7: SLL ──────────────────────────────────────────
    ADDI  x1, x0, 1
    ADDI  x2, x0, 4
    SLL   x31, x1, x2      # 1 << 4 = 16
    ADDI  x30, x0, 16
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 8: SRL ──────────────────────────────────────────
    ADDI  x1, x0, 256
    ADDI  x2, x0, 4
    SRL   x31, x1, x2      # 256 >> 4 = 16
    ADDI  x30, x0, 16
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 9: SLT ─────────────────────────────────────────
    ADDI  x1, x0, -5       # negative
    ADDI  x2, x0, 10
    SLT   x31, x1, x2      # -5 < 10 → 1
    ADDI  x30, x0, 1
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 10: LUI ────────────────────────────────────────
    LUI   x31, 0x12345
    LUI   x30, 0x12345
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 11: LUI + ADDI (32-bit construction) ──────────
    LUI   x31, 0xABCDE
    ADDI  x31, x31, 0x123
    LUI   x30, 0xABCDE
    ADDI  x30, x30, 0x123
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 12: ADDI negative ──────────────────────────────
    ADDI  x1, x0, 100
    ADDI  x31, x1, -30     # 100 - 30 = 70
    ADDI  x30, x0, 70
    BNE   x31, x30, fail
    NOP
    NOP
    ADDI  x28, x28, 1

# ── Test 13: x0 hardwired to zero ───────────────────────
    ADDI  x0, x0, 999      # should have no effect
    BNE   x0, x0, fail     # x0 must still be 0
    NOP
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1    # DBG0 = 1 (PASS)
    JAL   x0, 0
    NOP
    NOP

fail:
    CSRRW x0, 0xC10, x28   # DBG0 = test case number
    JAL   x0, 0
    NOP
    NOP
    
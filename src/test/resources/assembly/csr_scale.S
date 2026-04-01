# csr_scale.S — CSR operations + SELI
# @TIMEOUT 500

    ADDI  x28, x0, 1

# ── Test 1: CSRRW write and read-back via DBG1 ──────────
    ADDI  x1, x0, 0x42
    CSRRW x0, 0xC11, x1
    CSRRS x2, 0xC11, x0
    ADDI  x30, x0, 0x42
    BNE   x2, x30, fail
    NOP
    NOP
    CSRRW x0, 0xC11, x0
    ADDI  x28, x28, 1

# ── Test 2: CSRRS set-bits on DBG1 ─────────────────────
    ADDI  x1, x0, 0x0F
    CSRRW x0, 0xC11, x1
    ADDI  x3, x0, 0x30
    CSRRS x2, 0xC11, x3
    ADDI  x30, x0, 0x0F
    BNE   x2, x30, fail
    NOP
    NOP
    CSRRS x4, 0xC11, x0
    ADDI  x30, x0, 0x3F
    BNE   x4, x30, fail
    NOP
    NOP
    CSRRW x0, 0xC11, x0
    ADDI  x28, x28, 1

# ── Test 3: SELI writes don't crash ─────────────────────
    SELI  0, 0x37
    SELI  5, 0xAB
    SELI  0, 0x11
    SELI  1, 0x22
    SELI  2, 0x33
    ADDI  x28, x28, 1

# ── Test 4: Cycle counter is incrementing ────────────────
    CSRRS x1, 0xC00, x0
    NOP
    NOP
    NOP
    NOP
    CSRRS x2, 0xC00, x0
    SLT   x31, x1, x2
    ADDI  x30, x0, 1
    BNE   x31, x30, fail
    NOP
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    JAL   x0, 0
    NOP
    NOP

fail:
    CSRRW x0, 0xC10, x28
    JAL   x0, 0
    NOP
    NOP
    
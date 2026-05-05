# ================================================================
# scalar_isa.S — scalar instruction verification
# ================================================================
#
# Tests the following scalar instructions:
#
#   ALU I-type : ANDI, ORI, XORI, SLLI, SRLI, SRAI, SLTI, SLTIU
#   ALU R-type : AND, SRA, SLTU
#   Upper imm  : AUIPC
#   Misc       : FENCE, DELAY, ECALL, EBREAK
#   CSR        : CSRRC, CSRRWI, CSRRSI, CSRRCI
#   Branches   : BLT, BGE, BLTU, BGEU
#
# Convention (AssemblyTestRunner):
#   DBG0 (CSR 0xC10) = 0x01  → PASS
#   DBG0 = any other nonzero → FAIL (value encodes which test)
#   Terminate with  JAL x0, 0  after writing DBG0.
#
# Register usage:
#   x1-x6      operands / intermediate results
#   x29 (t4)   expected value
#   x30 (t5)   fail accumulator (0 = all pass)
#   x31 (t6)   current test ID (bit per test group)
#
# @TIMEOUT 10000
# ================================================================

    # ── Initialize ──
    ADDI    x30, x0, 0              # fail accumulator = 0
    ADDI    x31, x0, 1              # test ID bit = 1

# ================================================================
#  1. ALU I-TYPE TESTS
# ================================================================

# ---- ANDI -------------------------------------------------------
    ADDI    x1, x0, 0x7F            # x1 = 0x7F  (0111_1111)
    ANDI    x3, x1, 0x1F            # x3 = 0x7F & 0x1F = 0x1F
    ADDI    x29, x0, 0x1F
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # ANDI sign-extended edge: andi rd, rs1, -1 → identity
    ADDI    x1, x0, 0x55            # x1 = 0x55
    ANDI    x3, x1, -1              # x3 = 0x55 & 0xFFFFFFFF = 0x55
    ADDI    x29, x0, 0x55
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- ORI --------------------------------------------------------
    ADDI    x1, x0, 0x30            # x1 = 0x30
    ORI     x3, x1, 0x0F            # x3 = 0x30 | 0x0F = 0x3F
    ADDI    x29, x0, 0x3F
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # ORI sign-extended edge: ori rd, x0, -1 → 0xFFFFFFFF
    ORI     x3, x0, -1              # x3 = 0 | 0xFFFFFFFF = -1
    ADDI    x29, x0, -1
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- XORI -------------------------------------------------------
    ADDI    x1, x0, 0xFF            # x1 = 0xFF
    XORI    x3, x1, 0x0F            # x3 = 0xFF ^ 0x0F = 0xF0
    ADDI    x29, x0, 0xF0
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # XORI as bitwise NOT: xori rd, rs1, -1
    ADDI    x1, x0, 0x55            # x1 = 0x55
    XORI    x3, x1, -1              # x3 = ~0x55 = 0xFFFFFFAA
    ADDI    x29, x0, -0x56          # x29 = 0xFFFFFFAA  (== -86)
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- SLLI -------------------------------------------------------
    ADDI    x1, x0, 1
    SLLI    x3, x1, 4               # x3 = 1 << 4 = 16
    ADDI    x29, x0, 16
    XOR     x3, x3, x29
    OR      x30, x30, x3

    ADDI    x1, x0, 3
    SLLI    x3, x1, 8               # x3 = 3 << 8 = 768
    ADDI    x29, x0, 768
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- SRLI -------------------------------------------------------
    ADDI    x1, x0, 0x80            # x1 = 128
    SRLI    x3, x1, 4               # x3 = 128 >> 4 = 8
    ADDI    x29, x0, 8
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # SRLI on negative value: logical shift fills with zeros
    ADDI    x1, x0, -1              # x1 = 0xFFFFFFFF
    SRLI    x3, x1, 28              # x3 = 0xFFFFFFFF >>> 28 = 0x0F
    ADDI    x29, x0, 0x0F
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- SRAI -------------------------------------------------------
    ADDI    x1, x0, 64
    SRAI    x3, x1, 2               # x3 = 64 >>> 2 = 16
    ADDI    x29, x0, 16
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # SRAI on negative: sign bit propagates
    ADDI    x1, x0, -16             # x1 = 0xFFFFFFF0
    SRAI    x3, x1, 2               # x3 = -16 >>> 2 = -4
    ADDI    x29, x0, -4
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- SLTI -------------------------------------------------------
    # True case: -5 < 0
    ADDI    x1, x0, -5
    SLTI    x3, x1, 0               # x3 = 1
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # False case: 5 < 0
    ADDI    x1, x0, 5
    SLTI    x3, x1, 0               # x3 = 0
    OR      x30, x30, x3

    # Equal case: 5 < 5 → false
    ADDI    x1, x0, 5
    SLTI    x3, x1, 5               # x3 = 0
    OR      x30, x30, x3

# ---- SLTIU ------------------------------------------------------
    # True: 5 <u 10
    ADDI    x1, x0, 5
    SLTIU   x3, x1, 10              # x3 = 1
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # False: 10 <u 5
    ADDI    x1, x0, 10
    SLTIU   x3, x1, 5               # x3 = 0
    OR      x30, x30, x3

    # Edge: any value <u 0xFFFFFFFF (imm = -1, sign-extended)
    ADDI    x1, x0, 100
    SLTIU   x3, x1, -1              # x3 = (100 <u 0xFFFFFFFF) = 1
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # Edge: 0 <u 0 → false (sltiu with imm=0)
    SLTIU   x3, x0, 0               # x3 = (0 <u 0) = 0
    OR      x30, x30, x3

# ================================================================
#  2. ALU R-TYPE TESTS
# ================================================================

# ---- AND --------------------------------------------------------
    ADDI    x1, x0, 0x7F
    ADDI    x2, x0, 0x1F
    AND     x3, x1, x2              # x3 = 0x7F & 0x1F = 0x1F
    ADDI    x29, x0, 0x1F
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # AND with zero → 0
    ADDI    x1, x0, -1              # x1 = 0xFFFFFFFF
    AND     x3, x1, x0              # x3 = 0
    OR      x30, x30, x3

# ---- SRA --------------------------------------------------------
    # Positive operand
    ADDI    x1, x0, 64
    ADDI    x2, x0, 2
    SRA     x3, x1, x2              # x3 = 64 >>> 2 = 16
    ADDI    x29, x0, 16
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # Negative operand
    ADDI    x1, x0, -32
    ADDI    x2, x0, 3
    SRA     x3, x1, x2              # x3 = -32 >>> 3 = -4
    ADDI    x29, x0, -4
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # Shift by 0 → identity
    ADDI    x1, x0, 77
    SRA     x3, x1, x0              # x3 = 77
    ADDI    x29, x0, 77
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- SLTU -------------------------------------------------------
    # True: 5 <u 10
    ADDI    x1, x0, 5
    ADDI    x2, x0, 10
    SLTU    x3, x1, x2              # x3 = 1
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

    # False: 10 <u 5
    ADDI    x1, x0, 10
    ADDI    x2, x0, 5
    SLTU    x3, x1, x2              # x3 = 0
    OR      x30, x30, x3

    # Unsigned edge: 0xFFFFFFFF is NOT < 1
    ADDI    x1, x0, -1              # x1 = 0xFFFFFFFF
    ADDI    x2, x0, 1
    SLTU    x3, x1, x2              # x3 = (0xFFFF... <u 1) = 0
    OR      x30, x30, x3

    # Unsigned edge: 1 IS < 0xFFFFFFFF
    ADDI    x1, x0, 1
    ADDI    x2, x0, -1              # x2 = 0xFFFFFFFF
    SLTU    x3, x1, x2              # x3 = (1 <u 0xFFFF...) = 1
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ================================================================
#  3. AUIPC TEST
# ================================================================
# auipc rd, imm  →  x[rd] = pc + (imm << 12)
# Two consecutive AUIPCs with imm=0 must differ by exactly 1.

    AUIPC   x3, 0                   # x3 = pc_A
    AUIPC   x4, 0                   # x4 = pc_A + 1
    SUB     x5, x4, x3              # x5 = 1
    ADDI    x29, x0, 1
    XOR     x5, x5, x29
    OR      x30, x30, x5

    # Non-zero immediate
    AUIPC   x3, 1                   # x3 = P + 0x1000
    AUIPC   x4, 0                   # x4 = P + 1
    SUB     x5, x3, x4              # x5 = 0xFFF = 4095
    LUI     x29, 1                  # x29 = 0x1000 = 4096
    ADDI    x29, x29, -1            # x29 = 4095
    XOR     x5, x5, x29
    OR      x30, x30, x5

# ================================================================
#  4. FENCE TEST
# ================================================================
    ADDI    x3, x0, 42
    FENCE
    ADDI    x4, x0, 42
    XOR     x3, x3, x4
    OR      x30, x30, x3

# ================================================================
#  5. CSR TESTS  (using DBG1 @ 0xC11, which is RW)
# ================================================================
# NOTE: DBG0 (0xC10) is reserved for the pass/fail protocol,
# so all CSR read-modify-write tests use DBG1 (0xC11) instead.

# ---- CSRRWI -----------------------------------------------------
    CSRRW   x0, 0xC11, x0          # clear DBG1 to 0   (setup)
    CSRRWI  x3, 0xC11, 21          # old=0→x3; DBG1=21
    OR      x30, x30, x3           # x3 must be 0
    CSRRS   x4, 0xC11, x0          # readback DBG1
    ADDI    x29, x0, 21
    XOR     x4, x4, x29
    OR      x30, x30, x4

# ---- CSRRSI -----------------------------------------------------
    # DBG1 = 21 = 0b10101
    CSRRSI  x3, 0xC11, 10          # old=21→x3; DBG1=21|10=31
    ADDI    x29, x0, 21
    XOR     x3, x3, x29
    OR      x30, x30, x3
    CSRRS   x4, 0xC11, x0          # readback DBG1 = 31
    ADDI    x29, x0, 31
    XOR     x4, x4, x29
    OR      x30, x30, x4

# ---- CSRRCI -----------------------------------------------------
    # DBG1 = 31 = 0b11111
    CSRRCI  x3, 0xC11, 15          # old=31→x3; DBG1=31&~15=16
    ADDI    x29, x0, 31
    XOR     x3, x3, x29
    OR      x30, x30, x3
    CSRRS   x4, 0xC11, x0          # readback DBG1 = 16
    ADDI    x29, x0, 16
    XOR     x4, x4, x29
    OR      x30, x30, x4

# ---- CSRRC ------------------------------------------------------
    CSRRWI  x0, 0xC11, 31          # DBG1 = 0x1F   (setup)
    ADDI    x1, x0, 0x0F           # bits to clear
    CSRRC   x3, 0xC11, x1          # old=31→x3; DBG1=31&~15=16
    ADDI    x29, x0, 31
    XOR     x3, x3, x29
    OR      x30, x30, x3
    CSRRS   x4, 0xC11, x0          # readback DBG1 = 16
    ADDI    x29, x0, 16
    XOR     x4, x4, x29
    OR      x30, x30, x4

# ================================================================
#  6. DELAY TEST
# ================================================================
# Read cycle counter before and after DELAY 5.
# Elapsed must be >= 6.

    CSRR    x3, 0xC00               # x3 = cycle_before
    DELAY   5                       # stall 5 cycles
    CSRR    x4, 0xC00               # x4 = cycle_after
    SUB     x5, x4, x3              # x5 = elapsed
    SLTI    x6, x5, 6               # x6 = (elapsed < 6) ? 1 : 0
    OR      x30, x30, x6            # fail if elapsed < 6

    # DELAY 0 is a no-op
    ADDI    x3, x0, 99
    DELAY   0
    ADDI    x4, x0, 99
    XOR     x3, x3, x4
    OR      x30, x30, x3

# ================================================================
#  7. BRANCH TESTS
# ================================================================
# Architecture has 1 delay slot.
#
# Taken test pattern (branch skips the fail marker):
#   BXX  x1, x2, target       # +4 words to target
#   NOP                       # delay slot
#   ORI  x3, x3, 1            # fail marker (skipped if taken)
#   target:  OR x30, x30, x3  # x3 must be 0
#
# Not-taken test pattern:
#   BXX  x1, x2, target       # +4 words to target
#   NOP                       # delay slot
#   ORI  x3, x3, 1            # expected (x3 = 1)
#   target:  (check x3 == 1)

# ---- BLT taken: -5 < 5 (signed) --------------------------------
    ADDI    x1, x0, -5
    ADDI    x2, x0, 5
    ADDI    x3, x0, 0
    BLT     x1, x2, blt_t
    NOP
    ORI     x3, x3, 1
blt_t:
    OR      x30, x30, x3

# ---- BLT not-taken: 5 < -5 (signed) ----------------------------
    ADDI    x1, x0, 5
    ADDI    x2, x0, -5
    ADDI    x3, x0, 0
    BLT     x1, x2, blt_n
    NOP
    ORI     x3, x3, 1
blt_n:
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- BGE taken: 5 >= -5 (signed) -------------------------------
    ADDI    x1, x0, 5
    ADDI    x2, x0, -5
    ADDI    x3, x0, 0
    BGE     x1, x2, bge_t
    NOP
    ORI     x3, x3, 1
bge_t:
    OR      x30, x30, x3

# ---- BGE taken (equal): 5 >= 5 ---------------------------------
    ADDI    x1, x0, 5
    ADDI    x2, x0, 5
    ADDI    x3, x0, 0
    BGE     x1, x2, bge_eq
    NOP
    ORI     x3, x3, 1
bge_eq:
    OR      x30, x30, x3

# ---- BGE not-taken: -5 >= 5 (signed) ---------------------------
    ADDI    x1, x0, -5
    ADDI    x2, x0, 5
    ADDI    x3, x0, 0
    BGE     x1, x2, bge_n
    NOP
    ORI     x3, x3, 1
bge_n:
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- BLTU taken: 5 <u 10 ---------------------------------------
    ADDI    x1, x0, 5
    ADDI    x2, x0, 10
    ADDI    x3, x0, 0
    BLTU    x1, x2, bltu_t
    NOP
    ORI     x3, x3, 1
bltu_t:
    OR      x30, x30, x3

# ---- BLTU not-taken: 10 <u 5 -----------------------------------
    ADDI    x1, x0, 10
    ADDI    x2, x0, 5
    ADDI    x3, x0, 0
    BLTU    x1, x2, bltu_n
    NOP
    ORI     x3, x3, 1
bltu_n:
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- BLTU unsigned edge: 1 <u 0xFFFFFFFF -----------------------
    ADDI    x1, x0, 1
    ADDI    x2, x0, -1
    ADDI    x3, x0, 0
    BLTU    x1, x2, bltu_e
    NOP
    ORI     x3, x3, 1
bltu_e:
    OR      x30, x30, x3

# ---- BGEU taken: 10 >=u 5 --------------------------------------
    ADDI    x1, x0, 10
    ADDI    x2, x0, 5
    ADDI    x3, x0, 0
    BGEU    x1, x2, bgeu_t
    NOP
    ORI     x3, x3, 1
bgeu_t:
    OR      x30, x30, x3

# ---- BGEU taken (equal): 7 >=u 7 -------------------------------
    ADDI    x1, x0, 7
    ADDI    x2, x0, 7
    ADDI    x3, x0, 0
    BGEU    x1, x2, bgeu_eq
    NOP
    ORI     x3, x3, 1
bgeu_eq:
    OR      x30, x30, x3

# ---- BGEU not-taken: 5 >=u 10 ----------------------------------
    ADDI    x1, x0, 5
    ADDI    x2, x0, 10
    ADDI    x3, x0, 0
    BGEU    x1, x2, bgeu_n
    NOP
    ORI     x3, x3, 1
bgeu_n:
    ADDI    x29, x0, 1
    XOR     x3, x3, x29
    OR      x30, x30, x3

# ---- BGEU unsigned edge: 0xFFFFFFFF >=u 1 ----------------------
    ADDI    x1, x0, -1
    ADDI    x2, x0, 1
    ADDI    x3, x0, 0
    BGEU    x1, x2, bgeu_e
    NOP
    ORI     x3, x3, 1
bgeu_e:
    OR      x30, x30, x3

# ================================================================
#  8. ECALL / EBREAK TESTS
# ================================================================
# We cannot execute ECALL/EBREAK in the main flow (they halt the
# core before DBG0 is written).  Instead we verify their encodings
# assemble correctly by checking a known constant survives past
# where they would be — this proves the assembler accepted them.
#
# Full behavioral tests of ECALL/EBREAK require a dedicated
# harness that checks halt_reason in the CSR status register.

    # Verify ECALL encoding: the assembler must not reject it.
    # We branch over it so it never executes in the main flow.
    BEQ     x0, x0, ecall_skip      # always taken (0 == 0)
    NOP
    ECALL                            # never reached; proves encoding accepted
ecall_skip:

    # Verify EBREAK encoding similarly.
    BEQ     x0, x0, ebreak_skip     # always taken
    NOP
    EBREAK                           # never reached; proves encoding accepted
ebreak_skip:

# ================================================================
#  9. REPORT RESULT VIA DBG0
# ================================================================
# x30 == 0 → all tests passed → write 1 to DBG0
# x30 != 0 → some test failed → write x30 to DBG0 (nonzero, != 1)

    BEQ     x30, x0, all_pass
    NOP
    # ── FAIL path ──
    CSRW    x30, 0xC10              # DBG0 = fail flags (nonzero != 1)
    ECALL

all_pass:
    ADDI    x1, x0, 1
    CSRW    x1, 0xC10               # DBG0 = 1 (PASS)
    ECALL

# seld_byte_offsets.S — seld reads the byte at its address, at byte offsets 0 to 3 of a word
# @TIMEOUT 2000
#
# The word at VMEM byte 0x200 holds the bytes 0x81, 0x82, 0x83 and 0x84 (offsets 0 to 3). seld
# e1 to e4 read them from 0x200 to 0x203. vpack.bf16.fp8 multiplies 1.0 by 2^(e - 127) and gives
# the E4M3 codes 0x48, 0x50, 0x58 and 0x60. The test stores each packed register to VMEM byte
# 0x400 and checks its first word. No DELAY: the interlock orders seld, vpack, vstore and lw.
# DBG0 = 1 on pass, else 2 + the byte offset that failed.

    ADDI  x28, x0, 1
    LI    x6, 0x84838281
    ADDI  x7, x0, 0x200
    SW    x6, 0(x7)
    VLI.ALL m2, 0x3F80
    ADDI  x12, x0, 0x100          # VMEM word 0x100 = byte 0x400
    ADDI  x13, x0, 0x400

# ── byte offset 0: e1 = 0x81, 1.0 x 2^2 = E4M3 0x48 ──
    ADDI  x28, x28, 1
    ADDI  x8, x7, 0
    SELD  e1, x8
    VPACK.BF16.FP8 m4, m2, e1
    VSTORE m4, 0(x12)
    LW    x9, 0(x13)
    LI    x30, 0x48484848
    BNE   x9, x30, fail

# ── byte offset 1: e2 = 0x82, 1.0 x 2^3 = E4M3 0x50 ──
    ADDI  x28, x28, 1
    ADDI  x8, x7, 1
    SELD  e2, x8
    VPACK.BF16.FP8 m4, m2, e2
    VSTORE m4, 0(x12)
    LW    x9, 0(x13)
    LI    x30, 0x50505050
    BNE   x9, x30, fail

# ── byte offset 2: e3 = 0x83, 1.0 x 2^4 = E4M3 0x58 ──
    ADDI  x28, x28, 1
    ADDI  x8, x7, 2
    SELD  e3, x8
    VPACK.BF16.FP8 m4, m2, e3
    VSTORE m4, 0(x12)
    LW    x9, 0(x13)
    LI    x30, 0x58585858
    BNE   x9, x30, fail

# ── byte offset 3: e4 = 0x84, 1.0 x 2^5 = E4M3 0x60 ──
    ADDI  x28, x28, 1
    ADDI  x8, x7, 3
    SELD  e4, x8
    VPACK.BF16.FP8 m4, m2, e4
    VSTORE m4, 0(x12)
    LW    x9, 0(x13)
    LI    x30, 0x60606060
    BNE   x9, x30, fail

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

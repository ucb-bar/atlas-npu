# vpu_pack.S — VPU FP8 pack/unpack integration test
# Tests: vfp8pack, vfp8unpack with non-trivial scaling
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_pack.py
#
# Pack quantizes BF16 → FP8, unpack dequantizes FP8 → BF16.
# Pack consumes two BF16 mregs and writes one packed FP8 mreg.
# Unpack consumes one packed FP8 mreg and writes two BF16 mregs.
# Pack subtracts the decoded scale exponent before rebiasing to E4M3.
# Unpack adds the decoded scale exponent when reconstructing BF16.
#
# Test values:
#   Input:        2.0 (BF16 0x4000)
#   Pack scale:   scale reg 0 = 0x80 → 2^(128-127) = 2.0
#                 pack subtracts this exponent, so 2.0 → 1.0
#   Pack result:  1.0 → FP8 E4M3 0x38, packed into one output mreg
#   Unpack scale: scale reg 1 = 0x81 → 2^(129-127) = 4.0
#   Unpack result:1.0 × 4.0 = 4.0 → BF16 0x4080
#
# DRAM layout:
#   [0x0000..0x07FF]  Input A (BF16)        (2048 B)
#   [0x0800..0x0BFF]  vpack output (FP8)    (1024 B)
#   [0x0C00..0x0FFF]  unused
#   [0x1000..0x17FF]  vunpack output (BF16) (2048 B)
#
# Mreg: m0,m1 = input BF16    m4 = packed FP8    m8,m9 = unpacked BF16

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

# ── Load Input A (BF16) → m0,m1 ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 0x90000400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

# ================================================================
# Test 1: vfp8pack  m4, m0 with scale reg 0
#   scale reg 0 = 0x80 → 2^(+1)
#   pack subtracts this exponent, so 2.0 → 1.0 → E4M3 0x38
# ================================================================
    SELI  0, 0x80
    VFP8PACK 4, 0, 0
    DELAY 64

    LI    x8, 512
    VSTORE 4, x8, 0
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90000800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

# ================================================================
# Test 2: vfp8unpack  m8, m4 with scale reg 1
#   scale reg 1 = 0x81 → 2^2 = 4.0
#   result = 1.0 × 4.0 = 4.0 → BF16 0x4080
# ================================================================
    SELI  1, 0x81
    VFP8UNPACK 8, 4, 1
    DELAY 65

    LI    x8, 1024
    VSTORE 8, x8, 0
    DELAY 33
    VSTORE 9, x8, 8
    DELAY 33

    LI    x8, 1024
    LI    x3, 0x90001000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x90001400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
    

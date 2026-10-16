# smolvla_requant.S — port of npu-model smolvla_requant
# Tests: BF16 → FP8 unit-scale cast on seeded random 32x32 tile
#   y = cast_bf16_to_e4m3(X)
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_requant.py
#
# VFP8PACK consumes a BF16 (m, m+1) pair and emits a single FP8 mreg
# (half-size output). Scale register is loaded with E8M0 0x7F
# (= 127 biased → 2^0 = 1.0), so the pack subtracts 0 from the BF16
# exponent and behaves as a plain per-lane BF16→E4M3 cast.
#
# DRAM layout:
#   [0x0000..0x03FF]  Input X bank 0   (1024 B BF16)
#   [0x0400..0x07FF]  Input X bank 1   (1024 B BF16)
#   [0x0800..0x0BFF]  Output packed    (1024 B FP8)
#
# Mreg allocation:  m0,m1 = X BF16    m4 = packed FP8

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load Input X → VMEM[0..511] → (m0, m1) ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 256
    LI    x1, 0x90000400
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ── Unit-scale FP8 pack: scale reg 0 = 0x7F (2^0 = 1.0) ──
    seli e0, 0x7F
    vpack.bf16.fp8 m4, m0, e0
    DELAY 64

    LI    x8, 512
    vstore m4, 0(x8)
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90000800
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

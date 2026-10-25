# smolvla_parameterized_fused_silu_gate.S — tiled silu(X) for 64x64 BF16
# Despite the "gate" in the name, Jeremy's parameterized_fused_silu_gate.py
# is single-input silu(x) = x * sigmoid(x). Kept as a separate file for
# parity with the npu-model program namespace; functionally identical to
# smolvla_parameterized_silu.S. See stock smolvla_fused_silu_gate.S for
# the same naming-misnomer explanation.
# @TIMEOUT 2000000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_fused_silu_gate.py
#
# Shape: M=128, N=128 → 16 tiles.
# DRAM layout:
#   [0x00000..0x07FFF]  Input X   (16 tiles x 2048B BF16)
#   [0x08000..0x0FFFF]  Output Y  (16 tiles x 2048B BF16)
#
# Mreg allocation (m2..m5 invariant across tiles):
#   m0,m1   = X              m2,m3   = -1.0 (hoisted)
#   m4,m5   = +1.0 (hoisted) m6,m7   = -X
#   m8,m9   = exp(-X)        m10,m11 = 1+exp(-X)
#   m12,m13 = sigmoid(X)     m14,m15 = silu(X)

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024
    LI    x22, 0x8000

    # ── Invariant constants ──
    vli.all m2, 0xBF80          # -1.0
    DELAY 63
    vli.all m4, 0x3F80          # +1.0
    DELAY 63

    ADDI  x11, x0, 16
    ADDI  x10, x0, 0

    LUI   x15, 0x90000         # FPGA DRAM offset 0x9000_0000
tile_loop:
    SLLI  x20, x10, 11
    ADD   x20, x20, x15        # x20 += FPGA DRAM base

    # ── Load X ──
    ADDI  x1, x20, 0
    ADDI  x6, x0, 0
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x1, x20, 1024
    LI    x6, 256
    dma.load x6, x1, x2, ch0
    dma.wait ch0

    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ── silu chain ──
    vmul.bf16 m6, m0, m2         # -X
    DELAY 64
    vexp.bf16 m8, m6                  # exp(-X)
    DELAY 64
    vadd.bf16 m10, m8, m4         # 1 + exp(-X)
    DELAY 64
    vrecip.bf16 m12, m10         # sigmoid(X)
    DELAY 64
    vmul.bf16 m14, m0, m12        # silu(X)
    DELAY 64

    LI    x8, 512
    vstore m14, 0(x8)
    DELAY 33
    vstore m15, 8(x8)
    DELAY 33

    # ── DMA out ──
    ADD   x3, x22, x20
    LI    x8, 512
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    ADDI  x3, x3, 1024
    LI    x8, 768
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    ADDI  x10, x10, 1
    BNE   x10, x11, tile_loop

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

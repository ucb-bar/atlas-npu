# smolvla_fused_silu_gate.S - port of npu-model fused_silu_gate kernel
#
# Tests: out = silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
#
# Single input, single output. The "gate" in the kernel name is a
# misnomer inherited from the upstream coverage tracker phrase
# "silu + elementwise_mul", where elementwise_mul is the MLIR primitive
# opcode of op_b (the inner sigmoid*x multiply), NOT a separate gate
# operand. The npu-model reference is unambiguous: single-arg func,
# op_a=sigmoid(x), op_b=silu(x).
#
# Functionally equivalent to smolvla_silu; kept under the
# fused_silu_gate name for parity with the npu-model program list.
#
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_fused_silu_gate.py
#
# DRAM layout:
#   [0x0000..0x07FF]  x   (BF16 [32,32], 2 banks)
#   [0x0800..0x0FFF]  out (BF16 [32,32], 2 banks)
#
# Mreg allocation:
#   m0,m1   = x
#   m2,m3   = -1.0 const     m4,m5   = +1.0 const
#   m6,m7   = -x
#   m8,m9   = exp(-x)
#   m10,m11 = 1+exp(-x)
#   m12,m13 = sigmoid(x)
#   m14,m15 = silu(x)

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

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

    vli.all m2, 0xBF80
    DELAY 63
    vli.all m4, 0x3F80
    DELAY 63

    vmul.bf16 m6, m0, m2
    DELAY 64

    vexp.bf16 m8, m6
    DELAY 64

    vadd.bf16 m10, m8, m4
    DELAY 64

    vrecip.bf16 m12, m10
    DELAY 64

    vmul.bf16 m14, m0, m12
    DELAY 64

    LI    x8, 512
    vstore m14, 0(x8)
    DELAY 33
    vstore m15, 8(x8)
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90000800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 768
    LI    x3, 0x90000C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

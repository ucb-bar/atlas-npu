# smolvla_fused_silu_gate.S - port of npu-model fused_silu_gate kernel
#
# Tests: out = silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
#
# Single input, single output. The "gate" in the kernel name is a
# misnomer inherited from the upstream coverage tracker phrase
# "silu + elementwise_mul", where elementwise_mul is the MLIR primitive
# opcode of op_b (the inner sigmoid*x multiply), NOT a separate gate
# operand. Reference at .claude/npu-model/smolvla_fused_silu_gate.py
# is unambiguous: single-arg func, op_a=sigmoid(x), op_b=silu(x).
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

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    VLI.ALL 2, 0xBF80
    DELAY 150
    VLI.ALL 4, 0x3F80
    DELAY 150

    VMUL.BF16  6, 0, 2
    DELAY 150

    VEXP 8, 6
    DELAY 150

    VADD.BF16 10, 8, 4
    DELAY 150

    VRECIP.BF16 12, 10
    DELAY 150

    VMUL.BF16 14, 0, 12
    DELAY 150

    LI    x8, 512
    VSTORE 14, x8, 0
    DELAY 40
    VSTORE 15, x8, 8
    DELAY 40

    LI    x8, 512
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x0C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

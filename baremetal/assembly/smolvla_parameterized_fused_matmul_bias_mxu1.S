# smolvla_parameterized_fused_matmul_bias_mxu1.S — tiled (A_bf16 @ B_fp8) + bias_bf16
# MXU1 (IPT) sibling of smolvla_parameterized_fused_matmul_bias.S.
# Kernel structure matches MXU0 exactly; only the engine mnemonics swap:
#   VMATPUSH.ACC.BF16.MXU0 → .MXU1
#   VMATPOP.FP8.MXU0       → .MXU1
#   VMATPUSH.W.MXU0        → .MXU1
#   VMATMUL.MXU0           → .MXU1
#   VMATPOP.BF16.MXU0      → .MXU1
# MXU0 and MXU1 produce bit-identical FP8 under dequant-on-push / quant-on-pop
# (per sequencer logic), but their BF16 matmul outputs diverge due to SA
# per-PE rounding vs IPT single-rounding at the end. So the gen routes the
# matmul golden through IPTLinearRTLFunction.
#
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_fused_matmul_bias_mxu1.py
#
# Shape, DRAM layout, VMEM staging, and register map all match the MXU0
# sibling — see smolvla_parameterized_fused_matmul_bias.S for commentary.

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x25, 0x1000
    LI    x26, 0x2000
    LI    x27, 0x4000
    ADDI  x11, x0, 4

    ADDI  x10, x0, 0

output_tile_loop:
    SRLI  x12, x10, 1
    ANDI  x13, x10, 1

    SLLI  x20, x12, 11
    SLLI  x22, x13, 11
    ADD   x22, x22, x25
    SLLI  x23, x12, 1
    ADD   x23, x23, x13
    SLLI  x23, x23, 11
    ADD   x23, x23, x26
    SLLI  x24, x12, 1
    ADD   x24, x24, x13
    SLLI  x24, x24, 11
    ADD   x24, x24, x27

    ADDI  x6, x0, 0
    DMA.LOAD  x6, x20, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    ADDI  x1, x20, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    LI    x7, 512
    DMA.LOAD  x7, x22, x2, 0
    DMA.WAIT  0
    LI    x7, 512
    VLOAD 4, x7, 0
    DELAY 40

    LI    x9, 768
    DMA.LOAD  x9, x23, x2, 0
    DMA.WAIT  0
    LI    x9, 1024
    ADDI  x1, x23, 1024
    DMA.LOAD  x9, x1, x2, 0
    DMA.WAIT  0

    LI    x9, 768
    VLOAD 6, x9, 0
    DELAY 40
    VLOAD 7, x9, 8
    DELAY 40

    VTRPOSE.XLU 5, 4
    DELAY 80

    VMATPUSH.ACC.BF16.MXU1 1, 0
    DELAY 80
    SELI 0, 0x7F
    VMATPOP.FP8.MXU1 8, 0, 1
    DELAY 40

    VMATPUSH.W.MXU1 0, 5
    DELAY 40
    VMATMUL.MXU1 0, 8, 0
    DELAY 120
    VMATPOP.BF16.MXU1 10, 0
    DELAY 40

    VADD.BF16 12, 10, 6
    DELAY 150

    LI    x8, 1280
    VSTORE 12, x8, 0
    DELAY 40
    VSTORE 13, x8, 8
    DELAY 40

    LI    x8, 1280
    DMA.STORE x24, x8, x2, 1
    DMA.WAIT  1
    ADDI  x1, x24, 1024
    LI    x8, 1536
    DMA.STORE x1, x8, x2, 1
    DMA.WAIT  1

    ADDI  x10, x10, 1
    BNE   x10, x11, output_tile_loop
    NOP
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

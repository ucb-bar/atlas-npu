# smolvla_parameterized_matmul_mxu1.S — tiled C = A @ B for 64x32x64 FP8 → BF16
# MXU1 (IPT / inner product tree) sibling of smolvla_parameterized_matmul.S.
# Only difference from the MXU0 version is the engine mnemonic — Atlas
# exposes .MXU0 (systolic array) and .MXU1 (IPT) with independent
# accumulator state but identical ISA-level semantics. MXU0 and MXU1 do
# NOT produce bit-identical outputs (SA rounds per-PE, IPT rounds once at
# the end), so golden goes through software_models.mxu1_ipt.
#
# @TIMEOUT 400000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_smolvla_parameterized_matmul_mxu1.py
#
# Shape, DRAM layout, VMEM staging, and register map all match
# smolvla_parameterized_matmul.S. See that file for full commentary.

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x25, 0x1000
    LI    x17, 0x2000
    ADDI  x11, x0, 4

    ADDI  x10, x0, 0

output_tile_loop:
    SRLI  x12, x10, 1
    ANDI  x13, x10, 1

    SLLI  x20, x12, 10
    SLLI  x22, x13, 10
    ADD   x22, x22, x25

    ADDI  x6, x0, 0
    DMA.LOAD  x6, x20, x2, 0
    DMA.WAIT  0

    LI    x6, 256
    DMA.LOAD  x6, x22, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    LI    x6, 256
    VLOAD 4, x6, 0
    DELAY 40

    VTRPOSE.XLU 5, 4
    DELAY 80

    VMATPUSH.W.MXU1 0, 5
    DELAY 40
    VMATMUL.MXU1 0, 0, 0
    DELAY 120
    VMATPOP.BF16.MXU1 2, 0
    DELAY 40

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40
    LI    x8, 768
    VSTORE 3, x8, 0
    DELAY 40

    SLLI  x16, x12, 1
    ADD   x16, x16, x13
    SLLI  x16, x16, 11
    ADD   x16, x16, x17

    LI    x8, 512
    DMA.STORE x16, x8, x2, 1
    DMA.WAIT  1

    ADDI  x16, x16, 1024
    LI    x8, 768
    DMA.STORE x16, x8, x2, 1
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

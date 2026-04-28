# vpu_special_math_roundtrip_classes.S
# @TIMEOUT 700000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_special_math_roundtrip_classes.py
#
# Coverage intent:
#   - Class-directed BF16 inputs for VSQRT, VRECIP.BF16, VLOG2, and VEXP2.
#   - True unary-math dependency chain where VEXP2 consumes VLOG2 output.
#   - DMA -> VMEM -> VLOAD -> VPU -> VSTORE -> DMA end-to-end closure.
#   - Scalar-observable control check on a stored sentinel from the roundtrip result.
#
# DRAM byte layout (generator-backed):
#   [0x0000 .. 0x07FF]  input BF16 tensor pair                 (2048 B)
#   [0x0800 .. 0x0FFF]  VSQRT output tensor pair              (2048 B)
#   [0x1000 .. 0x17FF]  VRECIP output tensor pair             (2048 B)
#   [0x1800 .. 0x1FFF]  VLOG2 output tensor pair              (2048 B)
#   [0x2000 .. 0x27FF]  VEXP2(VLOG2(x)) output tensor pair    (2048 B)
#
# VMEM word layout:
#   0      input lo      (1024 B)
#   256    input hi      (1024 B)
#   512    sqrt lo
#   768    sqrt hi
#   1024   recip lo
#   1280   recip hi
#   1536   log2 lo
#   1792   log2 hi
#   2048   exp2 lo
#   2304   exp2 hi
#   scratch scalar byte region at VMEM byte 12288

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0

    LI    x12, 2048

    # Load input pair to VMEM[0..2047 B]
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0

    # VLOAD input pair into m40/m41
    ADDI  x7, x0, 0
    VLOAD 40, x7, 0
    DELAY 40
    VLOAD 41, x7, 8
    DELAY 40

    # VSQRT input -> m42/m43
    VSQRT 42, 40
    DELAY 150

    # VRECIP input -> m44/m45
    VRECIP.BF16 44, 40
    DELAY 150

    # VLOG2 input -> m46/m47
    VLOG2 46, 40
    DELAY 150

    # VEXP2(log2(x)) -> m48/m49
    VEXP2 48, 46
    DELAY 150

    # Store all result pairs back to VMEM
    LI    x8, 512
    VSTORE 42, x8, 0
    DELAY 40
    VSTORE 43, x8, 8
    DELAY 40

    LI    x8, 1024
    VSTORE 44, x8, 0
    DELAY 40
    VSTORE 45, x8, 8
    DELAY 40

    LI    x8, 1536
    VSTORE 46, x8, 0
    DELAY 40
    VSTORE 47, x8, 8
    DELAY 40

    LI    x8, 2048
    VSTORE 48, x8, 0
    DELAY 40
    VSTORE 49, x8, 8
    DELAY 40

    # Scalar-observable read from stored exp2(log2(x)) result.
    # The first packed word at byte 8192 is legitimately zero because the
    # class-directed inputs begin with +0.0 and -0.0, so probe the next word.
    # VMEM byte address = 2048 words * 4 bytes/word = 8192.
    LI    x20, 8192
    LW    x21, x20, 4
    DELAY 1
    BEQ   x21, x0, fail
    NOP
    NOP
    ADDI  x28, x28, 1

    # Drain all outputs to DRAM.
    LI    x6, 512
    LI    x10, 0x0800
    DMA.STORE x10, x6, x12, 1
    DMA.WAIT 1

    LI    x6, 1024
    LI    x10, 0x1000
    DMA.STORE x10, x6, x12, 2
    DMA.WAIT 2

    LI    x6, 1536
    LI    x10, 0x1800
    DMA.STORE x10, x6, x12, 3
    DMA.WAIT 3

    LI    x6, 2048
    LI    x10, 0x2000
    DMA.STORE x10, x6, x12, 4
    DMA.WAIT 4

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

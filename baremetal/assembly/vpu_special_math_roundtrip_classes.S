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

    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x12, 2048

    # Load input pair to VMEM[0..2047 B]
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    # VLOAD input pair into m40/m41
    ADDI  x7, x0, 0
    vload m40, 0(x7)
    DELAY 33
    vload m41, 8(x7)
    DELAY 33

    # VSQRT input -> m42/m43
    vsqrt.bf16 m42, m40
    DELAY 64

    # VRECIP input -> m44/m45
    vrecip.bf16 m44, m40
    DELAY 64

    # VLOG2 input -> m46/m47
    vlog2.bf16 m46, m40
    DELAY 64

    # VEXP2(log2(x)) -> m48/m49
    vexp2.bf16 m48, m46
    DELAY 64

    # Store all result pairs back to VMEM
    LI    x8, 512
    vstore m42, 0(x8)
    DELAY 33
    vstore m43, 8(x8)
    DELAY 33

    LI    x8, 1024
    vstore m44, 0(x8)
    DELAY 33
    vstore m45, 8(x8)
    DELAY 33

    LI    x8, 1536
    vstore m46, 0(x8)
    DELAY 33
    vstore m47, 8(x8)
    DELAY 33

    LI    x8, 2048
    vstore m48, 0(x8)
    DELAY 33
    vstore m49, 8(x8)
    DELAY 33

    # Scalar-observable read from stored exp2(log2(x)) result.
    # The first packed word at byte 8192 is legitimately zero because the
    # class-directed inputs begin with +0.0 and -0.0, so probe the next word.
    # VMEM byte address = 2048 words * 4 bytes/word = 8192.
    LI    x20, 8192
    lw x21, 4(x20)
    DELAY 4
    BEQ   x21, x0, fail
    NOP
    ADDI  x28, x28, 1

    # Drain all outputs to DRAM.
    LI    x6, 512
    LI    x10, 0x90000800
    dma.store x10, x6, x12, ch1
    dma.wait ch1

    LI    x6, 1024
    LI    x10, 0x90001000
    dma.store x10, x6, x12, ch2
    dma.wait ch2

    LI    x6, 1536
    LI    x10, 0x90001800
    dma.store x10, x6, x12, ch3
    dma.wait ch3

    LI    x6, 2048
    LI    x10, 0x90002000
    dma.store x10, x6, x12, ch4
    dma.wait ch4

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

# mxu1_quant_relu_roundtrip.S
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu1_quant_relu_roundtrip.py
#
# MXU1 -> VPU quantization bridge regression.
#
# DRAM byte layout (generator-backed):
#   [0x0000 .. 0x03FF]  A_fp8 input tile                      (1024 B)
#   [0x0400 .. 0x07FF]  B_fp8 input tile                      (1024 B)
#   [0x0800 .. 0x081F]  scale-byte preload region             (32 B; byte0 used by SELD)
#   [0x1000 .. 0x17FF]  BF16 MXU1 raw output                  (2048 B)
#   [0x1800 .. 0x1BFF]  FP8 packed ReLU output                (1024 B)
#   [0x2000 .. 0x27FF]  BF16 unpacked roundtrip output        (2048 B)
#
# VMEM word layout:
#   0      A_fp8 tile           (byte 0x0000)
#   256    B_fp8 tile           (byte 0x0400)
#   512    scale byte region    (byte 0x0800)
#   1024   raw BF16 out lo      (byte 0x1000)
#   1280   raw BF16 out hi      (byte 0x1400)
#   1536   packed FP8 out       (byte 0x1800)
#   2048   unpacked BF16 lo     (byte 0x2000)
#   2304   unpacked BF16 hi     (byte 0x2400)
#
# MREG allocation:
#   0       A_fp8
#   4       B_fp8
#   8,9     MXU1 raw BF16 output
#   12,13   ReLU BF16 output
#   16      packed FP8 output
#   20,21   unpacked BF16 output

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x12, 1024
    LI    x13, 2048
    LI    x14, 32

    # ------------------------------------------------------------------
    # Preload VMEM from DRAM
    # ------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    ADDI  x6, x0, 256
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1
    dma.wait ch1

    ADDI  x6, x0, 512
    LI    x10, 0x90000800
    dma.load x6, x10, x14, ch2
    dma.wait ch2

    # Load scale register from VMEM byte address 0x0800.
    LI    x7, 2048
    seld e0, 0(x7)
    DELAY 2

    # ------------------------------------------------------------------
    # Stage FP8 tiles into MREGs
    # ------------------------------------------------------------------
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33

    ADDI  x6, x0, 256
    vload m4, 0(x6)
    DELAY 33

    # ------------------------------------------------------------------
    # MXU1 matmul: C = A @ B
    # Generator preloads B^T because MXU computes act @ weight^T.
    # ------------------------------------------------------------------
    vmatpush.weight.mxu1 w0, m4
    DELAY 31

    vmatmul.mxu1 acc0, m0, w0
    DELAY 34

    vmatpop.bf16.acc.mxu1 m8, acc0
    DELAY 31

    # ------------------------------------------------------------------
    # VPU post-processing and quantization round-trip
    # ------------------------------------------------------------------
    vrelu.bf16 m12, m8
    DELAY 64

    vpack.bf16.fp8 m16, m12, e0
    DELAY 64

    vunpack.fp8.bf16 m20, m16, e0
    DELAY 65

    # ------------------------------------------------------------------
    # Drain raw BF16 output to DRAM [0x1000..0x17FF]
    # ------------------------------------------------------------------
    ADDI  x6, x0, 1024
    vstore m8, 0(x6)
    DELAY 33

    ADDI  x6, x0, 1280
    vstore m9, 0(x6)
    DELAY 33

    LI    x10, 0x90001000
    ADDI  x6, x0, 1024
    dma.store x10, x6, x13, ch3
    dma.wait ch3

    # ------------------------------------------------------------------
    # Drain packed FP8 output to DRAM [0x1800..0x1BFF]
    # ------------------------------------------------------------------
    ADDI  x6, x0, 1536
    vstore m16, 0(x6)
    DELAY 33

    LI    x10, 0x90001800
    ADDI  x6, x0, 1536
    dma.store x10, x6, x12, ch4
    dma.wait ch4

    # ------------------------------------------------------------------
    # Drain unpacked BF16 output to DRAM [0x2000..0x27FF]
    # ------------------------------------------------------------------
    LI    x6, 2048
    vstore m20, 0(x6)
    DELAY 33

    LI    x6, 2304
    vstore m21, 0(x6)
    DELAY 33

    LI    x10, 0x90002000
    LI    x6, 2048
    dma.store x10, x6, x13, ch5
    dma.wait ch5

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

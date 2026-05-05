# mxu1_quant_relu_roundtrip.S
# @TIMEOUT 500000
# @DRAM_BASE 0x00000001
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

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0

    LI    x12, 1024
    LI    x13, 2048
    LI    x14, 32

    # ------------------------------------------------------------------
    # Preload VMEM from DRAM
    # ------------------------------------------------------------------
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0

    ADDI  x6, x0, 256
    LI    x10, 1024
    DMA.LOAD x6, x10, x12, 1
    DMA.WAIT 1

    ADDI  x6, x0, 512
    LI    x10, 2048
    DMA.LOAD x6, x10, x14, 2
    DMA.WAIT 2

    # Load scale register from VMEM byte address 0x0800.
    LI    x7, 2048
    SELD  0, x7, 0
    DELAY 2

    # ------------------------------------------------------------------
    # Stage FP8 tiles into MREGs
    # ------------------------------------------------------------------
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40

    ADDI  x6, x0, 256
    VLOAD 4, x6, 0
    DELAY 40

    # ------------------------------------------------------------------
    # MXU1 matmul: C = A @ B
    # Generator preloads B^T because MXU computes act @ weight^T.
    # ------------------------------------------------------------------
    VMATPUSH.W.MXU1 0, 4
    DELAY 40

    VMATMUL.MXU1 0, 0, 0
    DELAY 120

    VMATPOP.BF16.MXU1 8, 0
    DELAY 40

    # ------------------------------------------------------------------
    # VPU post-processing and quantization round-trip
    # ------------------------------------------------------------------
    VRELU 12, 8
    DELAY 150

    VFP8PACK 16, 12, 0
    DELAY 150

    VFP8UNPACK 20, 16, 0
    DELAY 150

    # ------------------------------------------------------------------
    # Drain raw BF16 output to DRAM [0x1000..0x17FF]
    # ------------------------------------------------------------------
    ADDI  x6, x0, 1024
    VSTORE 8, x6, 0
    DELAY 40

    ADDI  x6, x0, 1280
    VSTORE 9, x6, 0
    DELAY 40

    LI    x10, 4096
    ADDI  x6, x0, 1024
    DMA.STORE x10, x6, x13, 3
    DMA.WAIT 3

    # ------------------------------------------------------------------
    # Drain packed FP8 output to DRAM [0x1800..0x1BFF]
    # ------------------------------------------------------------------
    ADDI  x6, x0, 1536
    VSTORE 16, x6, 0
    DELAY 40

    LI    x10, 6144
    ADDI  x6, x0, 1536
    DMA.STORE x10, x6, x12, 4
    DMA.WAIT 4

    # ------------------------------------------------------------------
    # Drain unpacked BF16 output to DRAM [0x2000..0x27FF]
    # ------------------------------------------------------------------
    ADDI  x6, x0, 2048
    VSTORE 20, x6, 0
    DELAY 40

    ADDI  x6, x0, 2304
    VSTORE 21, x6, 0
    DELAY 40

    LI    x10, 8192
    ADDI  x6, x0, 2048
    DMA.STORE x10, x6, x13, 5
    DMA.WAIT 5

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

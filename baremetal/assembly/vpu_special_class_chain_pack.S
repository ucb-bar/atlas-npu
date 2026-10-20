# vpu_special_class_chain_pack.S
# @TIMEOUT 300000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_special_class_chain_pack.py
#
# Coverage intent:
#   - Class-directed BF16 inputs stressing VRECIP.BF16 and VSQRT on zeros,
#     tiny values, powers of two, mantissa boundaries, negatives, and specials.
#   - True downstream dependency: multiply recip(x) * sqrt(x) after both unary ops.
#   - Quantization bridge coverage: SELD-loaded scale feeds VFP8PACK of the product.
#   - End-to-end DMA -> VMEM -> VLOAD -> VPU unary/binary/pack -> VSTORE -> DMA path.
#
# DRAM byte layout (generator-backed):
#   [0x0000 .. 0x07FF]  BF16 input tensor pair x                (2048 B)
#   [0x0800 .. 0x0FFF]  recip(x) output tensor pair            (2048 B)
#   [0x1000 .. 0x17FF]  sqrt(x) output tensor pair             (2048 B)
#   [0x1800 .. 0x1FFF]  recip(x)*sqrt(x) BF16 tensor pair      (2048 B)
#   [0x2000 .. 0x23FF]  packed FP8 product tensor              (1024 B)
#   [0x2400 .. 0x241F]  scale byte table for SELD              (32 B)
#
# VMEM word layout:
#   0      input bf16 bank 0      (byte 0x0000)
#   256    input bf16 bank 1      (byte 0x0400)
#   512    recip out bank 0       (byte 0x0800)
#   768    recip out bank 1       (byte 0x0C00)
#   1024   sqrt out bank 0        (byte 0x1000)
#   1280   sqrt out bank 1        (byte 0x1400)
#   1536   mul out bank 0         (byte 0x1800)
#   1792   mul out bank 1         (byte 0x1C00)
#   2048   packed fp8 bank        (byte 0x2000)
#   2304   scale table            (byte 0x2400)

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x12, 2048              # BF16 tensor-pair bytes
    LI    x13, 1024              # FP8 bank bytes
    LI    x14, 32                # scale table bytes

    # VMEM word addresses
    LI    x6, 0                  # input base
    LI    x7, 512                # recip output base
    LI    x8, 1024               # sqrt output base
    LI    x9, 1536               # mul output base
    LI    x10, 2048              # fp8 packed output base
    LI    x11, 2304              # scale table base

    # DRAM byte offsets
    LI    x20, 0x90000000  # input bf16 pair preload
    LI    x21, 0x90002400  # scale table preload
    LI    x22, 0x90000800  # recip output store
    LI    x23, 0x90001000  # sqrt output store
    LI    x24, 0x90001800  # mul output store
    LI    x25, 0x90002000  # packed fp8 store

    # Stage input tensor pair into VMEM.
    dma.load x6, x20, x12, ch0
    dma.wait ch0

    # Stage scale byte table into a disjoint VMEM region for SELD.
    dma.load x11, x21, x14, ch1
    dma.wait ch1

    # Load scale register e3 from VMEM byte address 0x2400.
    LI    x15, 0x2400
    seld e3, 0(x15)
    DELAY 1

    # Load BF16 input pair into m40/m41. The VLOAD path is single-issue in the
    # current RTL, so the second load must wait for the first to drain.
    vload m40, 0(x6)
    DELAY 33
    vload m41, 8(x6)
    DELAY 33

    # Unary math on the same input pair into disjoint destinations.
    vrecip.bf16 m42, m40
    DELAY 64
    vsqrt.bf16 m44, m40
    DELAY 64

    # Consume both unary outputs in a true dependent binary op.
    vmul.bf16 m46, m42, m44
    DELAY 64

    # Quantize the product under the SELD-loaded scale.
    vpack.bf16.fp8 m48, m46, e3
    DELAY 64

    # Spill all produced tensors back to VMEM.
    vstore m42, 0(x7)
    DELAY 33
    vstore m43, 8(x7)
    DELAY 33

    vstore m44, 0(x8)
    DELAY 33
    vstore m45, 8(x8)
    DELAY 33

    vstore m46, 0(x9)
    DELAY 33
    vstore m47, 8(x9)
    DELAY 33

    vstore m48, 0(x10)
    DELAY 33

    # Drain results to DRAM with serialized DMA traffic.
    dma.store x22, x7, x12, ch2
    dma.wait ch2

    dma.store x23, x8, x12, ch3
    dma.wait ch3

    dma.store x24, x9, x12, ch4
    dma.wait ch4

    dma.store x25, x10, x13, ch5
    dma.wait ch5

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

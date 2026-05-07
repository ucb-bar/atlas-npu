# vpu_special_class_chain_pack.S
# @TIMEOUT 300000
# @DRAM_BASE 0x00000001
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

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0

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
    LI    x20, 0x0000            # input bf16 pair preload
    LI    x21, 0x2400            # scale table preload
    LI    x22, 0x0800            # recip output store
    LI    x23, 0x1000            # sqrt output store
    LI    x24, 0x1800            # mul output store
    LI    x25, 0x2000            # packed fp8 store

    # Stage input tensor pair into VMEM.
    DMA.LOAD x6, x20, x12, 0
    DMA.WAIT 0

    # Stage scale byte table into a disjoint VMEM region for SELD.
    DMA.LOAD x11, x21, x14, 1
    DMA.WAIT 1

    # Load scale register e3 from VMEM byte address 0x2400.
    LI    x15, 0x2400
    SELD  3, x15, 0
    DELAY 1

    # Load BF16 input pair into m40/m41. The VLOAD path is single-issue in the
    # current RTL, so the second load must wait for the first to drain.
    VLOAD 40, x6, 0
    DELAY 33
    VLOAD 41, x6, 8
    DELAY 33

    # Unary math on the same input pair into disjoint destinations.
    VRECIP.BF16 42, 40
    DELAY 64
    VSQRT 44, 40
    DELAY 64

    # Consume both unary outputs in a true dependent binary op.
    VMUL.BF16 46, 42, 44
    DELAY 64

    # Quantize the product under the SELD-loaded scale.
    VFP8PACK 48, 46, 3
    DELAY 64

    # Spill all produced tensors back to VMEM.
    VSTORE 42, x7, 0
    DELAY 33
    VSTORE 43, x7, 8
    DELAY 33

    VSTORE 44, x8, 0
    DELAY 33
    VSTORE 45, x8, 8
    DELAY 33

    VSTORE 46, x9, 0
    DELAY 33
    VSTORE 47, x9, 8
    DELAY 33

    VSTORE 48, x10, 0
    DELAY 33

    # Drain results to DRAM with serialized DMA traffic.
    DMA.STORE x22, x7, x12, 2
    DMA.WAIT 2

    DMA.STORE x23, x8, x12, 3
    DMA.WAIT 3

    DMA.STORE x24, x9, x12, 4
    DMA.WAIT 4

    DMA.STORE x25, x10, x13, 5
    DMA.WAIT 5

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

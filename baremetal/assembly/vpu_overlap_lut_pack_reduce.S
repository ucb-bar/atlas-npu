# vpu_overlap_lut_pack_reduce.S - overlapped VPU LUT/pack/reduce chain
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_overlap_lut_pack_reduce.py
#
# Coverage intent:
#   - Use the VPU overlap path with independent single-input special functions.
#   - Chain LUT-heavy ops into add/mul, row reduction, FP8 pack/unpack, and
#     column reduction so the low-toggle VPU leaves drive later datapaths.
#   - Exercise scale-controlled FP8 saturation/underflow through non-trivial
#     pack/unpack exponents.
#
# DRAM layout:
#   0x0000 input A BF16 pair      2048 B
#   0x0800 input B BF16 pair      2048 B
#   0x1000 packed mix FP8        1024 B
#   0x1400 unpacked mix lo       1024 B
#   0x1800 unpacked mix hi       1024 B
#   0x1C00 row-sum lo            1024 B
#   0x2000 row-sum hi            1024 B
#   0x2400 column-sum lo         1024 B
#   0x2800 column-sum hi         1024 B

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024

    # Load input A into VMEM[0..511] and m0/m1.
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI    x6, 256
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # Load input B into VMEM[512..1023] and m2/m3.
    LI    x6, 512
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI    x6, 768
    LI    x10, 0x90000C00
    dma.load x6, x10, x12, ch0
    dma.wait ch0
    LI    x6, 512
    vload m2, 0(x6)
    DELAY 33
    vload m3, 8(x6)
    DELAY 33

    # Overlap slot exercise: log2(A) and sqrt(B) are independent lane boxes.
    vlog2.bf16 m8, m0
    vsqrt.bf16 m10, m2
    DELAY 65

    # Data-dependent special-function chain, again using the overlap slot.
    vexp2.bf16 m12, m8
    vtanh.bf16 m14, m10
    DELAY 65

    # Drive reciprocal and sin/cos paths from prior special-function outputs.
    vrecip.bf16 m16, m12
    vsin.bf16 m18, m14
    DELAY 65
    vcos.bf16 m20, m12
    DELAY 64

    # Mix the special-function products through hardfloat add/mul trees.
    vmul.bf16 m22, m16, m18
    DELAY 64
    vadd.bf16 m24, m22, m20
    DELAY 64

    # Row reduction on the mixed BF16 pair.
    vredsum.row.bf16 m26, m24
    DELAY 37

    # Pack/unpack with scale exponents chosen to hit normal, zero, and clip paths.
    seli e3, 0x83
    seli e4, 0x86
    vpack.bf16.fp8 m28, m24, e3
    DELAY 64
    vunpack.fp8.bf16 m30, m28, e4
    DELAY 65

    # Column sum after unpack expands the packed stream back to a BF16 pair.
    vredsum.bf16 m32, m30
    DELAY 128

    # Store packed result.
    LI    x8, 1024
    vstore m28, 0(x8)
    DELAY 33
    LI    x10, 0x90001000
    dma.store x10, x8, x12, ch1
    dma.wait ch1

    # Store unpacked pair.
    LI    x8, 1280
    vstore m30, 0(x8)
    DELAY 33
    LI    x10, 0x90001400
    dma.store x10, x8, x12, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m31, 0(x8)
    DELAY 33
    LI    x10, 0x90001800
    dma.store x10, x8, x12, ch1
    dma.wait ch1

    # Store row-reduction pair.
    LI    x8, 1792
    vstore m26, 0(x8)
    DELAY 33
    LI    x10, 0x90001C00
    dma.store x10, x8, x12, ch1
    dma.wait ch1
    LI    x8, 2048
    vstore m27, 0(x8)
    DELAY 33
    LI    x10, 0x90002000
    dma.store x10, x8, x12, ch1
    dma.wait ch1

    # Store column-reduction pair.
    LI    x8, 2304
    vstore m32, 0(x8)
    DELAY 33
    LI    x10, 0x90002400
    dma.store x10, x8, x12, ch1
    dma.wait ch1
    LI    x8, 2560
    vstore m33, 0(x8)
    DELAY 33
    LI    x10, 0x90002800
    dma.store x10, x8, x12, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

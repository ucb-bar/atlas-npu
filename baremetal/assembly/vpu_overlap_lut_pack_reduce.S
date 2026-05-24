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
    DMA.CONFIG x5, 0
    LI    x12, 1024

    # Load input A into VMEM[0..511] and m0/m1.
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0
    LI    x6, 256
    LI    x10, 0x90000400
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Load input B into VMEM[512..1023] and m2/m3.
    LI    x6, 512
    LI    x10, 0x90000800
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0
    LI    x6, 768
    LI    x10, 0x90000C00
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0
    LI    x6, 512
    VLOAD 2, x6, 0
    DELAY 33
    VLOAD 3, x6, 8
    DELAY 33

    # Overlap slot exercise: log2(A) and sqrt(B) are independent lane boxes.
    VLOG2 8, 0
    VSQRT 10, 2
    DELAY 65

    # Data-dependent special-function chain, again using the overlap slot.
    VEXP2 12, 8
    VTANH 14, 10
    DELAY 65

    # Drive reciprocal and sin/cos paths from prior special-function outputs.
    VRECIP.BF16 16, 12
    VSIN 18, 14
    DELAY 65
    VCOS 20, 12
    DELAY 64

    # Mix the special-function products through hardfloat add/mul trees.
    VMUL.BF16 22, 16, 18
    DELAY 64
    VADD.BF16 24, 22, 20
    DELAY 64

    # Row reduction on the mixed BF16 pair.
    VREDSUM.ROW.BF16 26, 24
    DELAY 37

    # Pack/unpack with scale exponents chosen to hit normal, zero, and clip paths.
    SELI  3, 0x7B
    SELI  4, 0x86
    VFP8PACK 28, 24, 3
    DELAY 64
    VFP8UNPACK 30, 28, 4
    DELAY 65

    # Column sum after unpack expands the packed stream back to a BF16 pair.
    VREDSUM.BF16 32, 30
    DELAY 128

    # Store packed result.
    LI    x8, 1024
    VSTORE 28, x8, 0
    DELAY 33
    LI    x10, 0x90001000
    DMA.STORE x10, x8, x12, 1
    DMA.WAIT 1

    # Store unpacked pair.
    LI    x8, 1280
    VSTORE 30, x8, 0
    DELAY 33
    LI    x10, 0x90001400
    DMA.STORE x10, x8, x12, 1
    DMA.WAIT 1
    LI    x8, 1536
    VSTORE 31, x8, 0
    DELAY 33
    LI    x10, 0x90001800
    DMA.STORE x10, x8, x12, 1
    DMA.WAIT 1

    # Store row-reduction pair.
    LI    x8, 1792
    VSTORE 26, x8, 0
    DELAY 33
    LI    x10, 0x90001C00
    DMA.STORE x10, x8, x12, 1
    DMA.WAIT 1
    LI    x8, 2048
    VSTORE 27, x8, 0
    DELAY 33
    LI    x10, 0x90002000
    DMA.STORE x10, x8, x12, 1
    DMA.WAIT 1

    # Store column-reduction pair.
    LI    x8, 2304
    VSTORE 32, x8, 0
    DELAY 33
    LI    x10, 0x90002400
    DMA.STORE x10, x8, x12, 1
    DMA.WAIT 1
    LI    x8, 2560
    VSTORE 33, x8, 0
    DELAY 33
    LI    x10, 0x90002800
    DMA.STORE x10, x8, x12, 1
    DMA.WAIT 1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

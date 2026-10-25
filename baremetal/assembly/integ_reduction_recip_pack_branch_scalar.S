# integ_reduction_recip_pack_branch_scalar.S
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_integ_reduction_recip_pack_branch_scalar.py
#
# Coverage intent:
#   - Reduction result reused as a true intermediate: VREDSUM.ROW -> VRECIP -> VMUL.
#   - Quantization bridge with scale loaded from VMEM via SELD, then VFP8PACK.
#   - Scalar-observable validation and branch control after tensor storeback.
#   - End-to-end DMA -> VMEM -> VLOAD -> VPU reduce/unary/binary/pack -> VSTORE -> DMA.
#
# DRAM byte layout (generator-backed):
#   [0x0000 .. 0x07FF]  input BF16 tensor pair                 (2048 B)
#   [0x0800 .. 0x081F]  scale-byte preload region             (32 B; byte0 used by SELD)
#   [0x1000 .. 0x17FF]  normalized BF16 tensor pair output    (2048 B)
#   [0x1800 .. 0x1BFF]  normalized FP8 packed output          (1024 B)
#
# VMEM layout:
#   word 0     / byte 0x0000   : input low half staging       (1024 B)
#   word 256   / byte 0x0400   : input high half staging      (1024 B)
#   word 512   / byte 0x0800   : scale-byte region            (32 B used, DMA-aligned load)
#   word 1024  / byte 0x1000   : BF16 output low half         (1024 B)
#   word 1280  / byte 0x1400   : BF16 output high half        (1024 B)
#   word 1536  / byte 0x1800   : FP8 packed output            (1024 B)
#   byte 0x3000                : scalar scratch / readback    (disjoint from DMA store sources)
#
# MREG allocation:
#   m40,m41 : input tile
#   m44,m45 : row-sum broadcast
#   m48,m49 : reciprocal(row-sum)
#   m52,m53 : normalized BF16 output
#   m56     : normalized FP8 packed output

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0

    # Sizes
    LI    x20, 1024
    LI    x21, 2048
    LI    x22, 32

    # VMEM word addresses
    LI    x6, 0           # input low
    LI    x7, 256         # input high
    LI    x8, 512         # scale region
    LI    x9, 1024        # bf16 out low
    LI    x10, 1280       # bf16 out high
    LI    x11, 1536       # fp8 out

    # DRAM byte offsets
    LI    x12, 0x90000000  # input pair
    LI    x13, 0x90000400  # input high half source
    LI    x14, 0x90000800  # scale preload region
    LI    x15, 0x90001000  # bf16 output dst
    LI    x16, 0x90001400  # bf16 output high dst
    LI    x17, 0x90001800  # fp8 output dst

    # DMA input low half -> VMEM[0x0000]
    dma.load x6, x12, x20, ch0
    dma.wait ch0

    # DMA input high half -> VMEM[0x0400]
    dma.load x7, x13, x20, ch1
    dma.wait ch1

    # DMA scale region -> VMEM[0x0800]
    dma.load x8, x14, x22, ch2
    dma.wait ch2

    # Load BF16 pair into upper-mid banks.
    vload m40, 0(x6)
    DELAY 33
    vload m41, 0(x7)
    DELAY 33

    # row_sum = broadcast(sum(row(input)))
    vredsum.row.bf16 m44, m40
    DELAY 37

    # recip_sum = recip(row_sum)
    vrecip.bf16 m48, m44
    DELAY 64

    # normalized = input * recip_sum
    vmul.bf16 m52, m40, m48
    DELAY 64

    # Load pack scale from VMEM byte address 0x0800 into e0.
    LI    x18, 0x0800
    seld e0, x18
    DELAY 2

    # packed = fp8pack(normalized, e0)
    vpack.bf16.fp8 m56, m52, e0
    DELAY 64

    # Store BF16 normalized tensor pair to VMEM.
    vstore m52, 0(x9)
    DELAY 33
    vstore m53, 0(x10)
    DELAY 33

    # Store FP8 packed result to VMEM.
    vstore m56, 0(x11)
    DELAY 33

    # Scalar-observable readback from BF16 output footprint, then branch on nonzero.
    # This is a read only; scalar scratch stores remain disjoint from DMA.STORE regions.
    LI    x19, 0x1000
    lw x23, 0(x19)
    DELAY 2
    BEQ   x23, x0, fail
    NOP

    # Branch coverage with a not-taken path determined by the loaded result.
    ADDI  x24, x0, 1
    BEQ   x23, x24, scalar_marker
    NOP

after_scalar_branch:
    # Drain BF16 outputs to DRAM.
    dma.store x15, x9, x20, ch3
    dma.wait ch3
    dma.store x16, x10, x20, ch4
    dma.wait ch4

    # Drain FP8 packed output to DRAM.
    dma.store x17, x11, x20, ch5
    dma.wait ch5

    JAL   x0, pass
    NOP

scalar_marker:
    # Write a disjoint scalar scratch marker, then fall through to common storeback.
    LI    x25, 0x3000
    sw x24, 0(x25)
    JAL   x0, after_scalar_branch
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

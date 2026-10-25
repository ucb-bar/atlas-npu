# vpu_vx_vr.S — the .vx and .vr forms of the two-input BF16 VPU instructions
# @TIMEOUT 50000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_vx_vr.py
#
# .vx takes its second operand from the BF16 bits x[rs2][31:16], .vr from row 0 of T(vs2). Each
# result must equal the .bf16 instruction with that operand broadcast to a tile (the generator).
#
# DRAM (from the generator):
#   0x0000  A = T(m0)    0x0800  B = T(m2) = T(m32)    0x1000  0x3FC0BEEF (1.5 in bits 31:16)
#   0x2000 + 0x800 k    output k
#
#   k  instruction                        notes
#   0  vadd.bf16.vx m4, m0, x7            x7 loaded by lw right before it: the interlock waits
#   1  vsub.bf16.vx m6, m0, x8            x8 = lui 0xc0a00: -5.0
#   2  vmul.bf16.vx m8, m0, x0            x0: +0
#   3  vminimum.bf16.vx m10, m0, x7
#   4  vmaximum.bf16.vx m12, m0, x7
#   5  vadd.bf16.vr m14, m0, m32          vs2 in the physical banks of vs1: legal
#   6  vsub.bf16.vr m16, m0, m2
#   7  vmul.bf16.vr m2, m0, m2            vd = vs2
#   8  vminimum.bf16.vr m18, m0, m32
#   9  vmaximum.bf16.vr m20, m0, m32

    ADDI  x28, x0, 1
    LI    x5, 0
    dma.config x5, ch0
    LI    x2, 2048
    LI    x1, 0x90000000
    dma.load x0, x1, x2, ch0         # A -> VMEM byte 0
    LI    x6, 512
    LI    x1, 0x90000800
    dma.load x6, x1, x2, ch1         # B -> VMEM byte 2048
    LI    x6, 1024
    LI    x1, 0x90001000
    ADDI  x3, x0, 32
    dma.load x6, x1, x3, ch2         # the scalar word -> VMEM byte 4096
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    vload m0, 0(x0)
    vload m1, 8(x0)
    LI    x9, 512
    vload m2, 0(x9)
    vload m3, 8(x9)
    vload m32, 0(x9)
    vload m33, 8(x9)
    LUI   x8, 0xc0a00
    LI    x10, 4096
    lw    x7, 0(x10)
    vadd.bf16.vx m4, m0, x7
    vsub.bf16.vx m6, m0, x8
    vmul.bf16.vx m8, m0, x0
    vminimum.bf16.vx m10, m0, x7
    vmaximum.bf16.vx m12, m0, x7
    vadd.bf16.vr m14, m0, m32
    vsub.bf16.vr m16, m0, m2
    vmul.bf16.vr m2, m0, m2
    vminimum.bf16.vr m18, m0, m32
    vmaximum.bf16.vr m20, m0, m32

    # Outputs -> VMEM byte 8192 + 2048 k -> DRAM
    LI    x20, 2048
    vstore m4, 0(x20)
    vstore m5, 8(x20)
    vstore m6, 16(x20)
    vstore m7, 24(x20)
    vstore m8, 32(x20)
    vstore m9, 40(x20)
    vstore m10, 48(x20)
    vstore m11, 56(x20)
    vstore m12, 64(x20)
    vstore m13, 72(x20)
    vstore m14, 80(x20)
    vstore m15, 88(x20)
    vstore m16, 96(x20)
    vstore m17, 104(x20)
    vstore m2, 112(x20)
    vstore m3, 120(x20)
    vstore m18, 128(x20)
    vstore m19, 136(x20)
    vstore m20, 144(x20)
    vstore m21, 152(x20)
    LI    x6, 2048
    LI    x1, 0x90002000
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 2560
    LI    x1, 0x90002800
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 3072
    LI    x1, 0x90003000
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 3584
    LI    x1, 0x90003800
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 4096
    LI    x1, 0x90004000
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 4608
    LI    x1, 0x90004800
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 5120
    LI    x1, 0x90005000
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 5632
    LI    x1, 0x90005800
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 6144
    LI    x1, 0x90006000
    dma.store x1, x6, x2, ch3
    dma.wait ch3
    LI    x6, 6656
    LI    x1, 0x90006800
    dma.store x1, x6, x2, ch3
    dma.wait ch3

    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

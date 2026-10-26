# vpu_rows.S — VPU instructions at row counts M (vsetm)
# @TIMEOUT 60000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_rows.py
#
# An instruction of M rows writes rows 0..M-1 of each register of its destination (vli.col and
# vli.one: of m[vd]); the other rows keep their values. Each destination starts with random data.
# A column reduction reduces rows 0..M-1 of both registers of its source pair. vpack and vunpack
# take all 32 rows at M = 3.
#
#   M   instruction                         M   instruction
#   3   vadd.bf16 m8, m0, m2                4   vli.all m24, 0x3f80
#   1   vmul.bf16.vx m10, m0, x7            2   vli.col m26, 0x4000
#   5   vsub.bf16.vr m12, m0, m2            3   vli.row m28, 0xbf80
#   7   vexp.bf16 m14, m0                   5   vli.one m30, 0x4040
#   2   vgelu.bf16 m16, m2                  3   vpack.bf16.fp8 m32, m0, e0
#   6   vredsum.row.bf16 m18, m0            3   vunpack.fp8.bf16 m34, m4, e0
#   3   vredmax.bf16 m20, m0               31   vmov m36, m2
#   4   vredsum.bf16 m22, m2                2   vminimum.bf16 m38, m0, m2
#
# DRAM (32-byte words, from the generator): 0 T(m0), 64 T(m2), 128 the E4M3 register m4,
# 160 + 32 (r - 8) the start of m_r for r = 8..39, 2048 + 32 (r - 8) the result m_r.
# VMEM: the DRAM inputs at byte 0 (m0-m4 at KiB 0-4, m_r at KiB r - 3); the results at KiB 56 + r.

    LI    x5, 0
    dma.config x5, ch0
    LI    x2, 4096
    LI    x6, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    LI    x6, 1024
    LI    x1, 0x90001000
    dma.load x6, x1, x2, ch1
    LI    x6, 2048
    LI    x1, 0x90002000
    dma.load x6, x1, x2, ch2
    LI    x6, 3072
    LI    x1, 0x90003000
    dma.load x6, x1, x2, ch3
    LI    x6, 4096
    LI    x1, 0x90004000
    dma.load x6, x1, x2, ch4
    LI    x6, 5120
    LI    x1, 0x90005000
    dma.load x6, x1, x2, ch5
    LI    x6, 6144
    LI    x1, 0x90006000
    dma.load x6, x1, x2, ch6
    LI    x6, 7168
    LI    x1, 0x90007000
    dma.load x6, x1, x2, ch0
    LI    x6, 8192
    LI    x1, 0x90008000
    dma.load x6, x1, x2, ch1
    LI    x6, 9216
    LI    x1, 0x90009000
    LI    x3, 1024
    dma.load x6, x1, x3, ch2
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6
    vload m0, 0(x0)
    vload m1, 8(x0)
    vload m2, 16(x0)
    vload m3, 24(x0)
    vload m4, 32(x0)
    vload m8, 40(x0)
    vload m9, 48(x0)
    vload m10, 56(x0)
    vload m11, 64(x0)
    vload m12, 72(x0)
    vload m13, 80(x0)
    vload m14, 88(x0)
    vload m15, 96(x0)
    vload m16, 104(x0)
    vload m17, 112(x0)
    vload m18, 120(x0)
    vload m19, 128(x0)
    vload m20, 136(x0)
    vload m21, 144(x0)
    vload m22, 152(x0)
    vload m23, 160(x0)
    vload m24, 168(x0)
    vload m25, 176(x0)
    vload m26, 184(x0)
    vload m27, 192(x0)
    vload m28, 200(x0)
    vload m29, 208(x0)
    vload m30, 216(x0)
    vload m31, 224(x0)
    vload m32, 232(x0)
    vload m33, 240(x0)
    vload m34, 248(x0)
    vload m35, 256(x0)
    vload m36, 264(x0)
    vload m37, 272(x0)
    vload m38, 280(x0)
    vload m39, 288(x0)
    seli  e0, 127
    LUI   x7, 0x3fc00

    vsetm 3
    vadd.bf16 m8, m0, m2
    vsetm 1
    vmul.bf16.vx m10, m0, x7
    vsetm 5
    vsub.bf16.vr m12, m0, m2
    vsetm 7
    vexp.bf16 m14, m0
    vsetm 2
    vgelu.bf16 m16, m2
    vsetm 6
    vredsum.row.bf16 m18, m0
    vsetm 3
    vredmax.bf16 m20, m0
    vsetm 4
    vredsum.bf16 m22, m2
    vli.all m24, 0x3f80
    vsetm 2
    vli.col m26, 0x4000
    vsetm 3
    vli.row m28, 0xbf80
    vsetm 5
    vli.one m30, 0x4040
    vsetm 3
    vpack.bf16.fp8 m32, m0, e0
    vunpack.fp8.bf16 m34, m4, e0
    vsetm 31
    vmov m36, m2
    vsetm 2
    vminimum.bf16 m38, m0, m2

    # Results -> VMEM KiB 64.. -> DRAM word 2048
    vsetm 32
    vstore m8, 512(x0)
    vstore m9, 520(x0)
    vstore m10, 528(x0)
    vstore m11, 536(x0)
    vstore m12, 544(x0)
    vstore m13, 552(x0)
    vstore m14, 560(x0)
    vstore m15, 568(x0)
    vstore m16, 576(x0)
    vstore m17, 584(x0)
    vstore m18, 592(x0)
    vstore m19, 600(x0)
    vstore m20, 608(x0)
    vstore m21, 616(x0)
    vstore m22, 624(x0)
    vstore m23, 632(x0)
    vstore m24, 640(x0)
    vstore m25, 648(x0)
    vstore m26, 656(x0)
    vstore m27, 664(x0)
    vstore m28, 672(x0)
    vstore m29, 680(x0)
    vstore m30, 688(x0)
    vstore m31, 696(x0)
    vstore m32, 704(x0)
    vstore m33, 712(x0)
    vstore m34, 720(x0)
    vstore m35, 728(x0)
    vstore m36, 736(x0)
    vstore m37, 744(x0)
    vstore m38, 752(x0)
    vstore m39, 760(x0)
    LI    x6, 16384
    LI    x1, 0x90010000
    dma.store x1, x6, x2, ch0
    LI    x6, 17408
    LI    x1, 0x90011000
    dma.store x1, x6, x2, ch1
    LI    x6, 18432
    LI    x1, 0x90012000
    dma.store x1, x6, x2, ch2
    LI    x6, 19456
    LI    x1, 0x90013000
    dma.store x1, x6, x2, ch3
    LI    x6, 20480
    LI    x1, 0x90014000
    dma.store x1, x6, x2, ch4
    LI    x6, 21504
    LI    x1, 0x90015000
    dma.store x1, x6, x2, ch5
    LI    x6, 22528
    LI    x1, 0x90016000
    dma.store x1, x6, x2, ch6
    LI    x6, 23552
    LI    x1, 0x90017000
    dma.store x1, x6, x2, ch0
    dma.wait ch0
    dma.wait ch1
    dma.wait ch2
    dma.wait ch3
    dma.wait ch4
    dma.wait ch5
    dma.wait ch6

    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

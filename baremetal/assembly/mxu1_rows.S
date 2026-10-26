# mxu1_rows.S — mxu1 at row counts M (vsetm)
# @TIMEOUT 60000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu1_rows.py
#
# A command of M rows moves rows 0..M-1 of its accumulator and m-registers; the other rows keep
# their values. A weight push moves all rows. The matmul chains at M = 1 and 2 are below mxu1's
# matmul spacing (4 cycles): each matmul.acc issues only after the previous one on its
# accumulator has written row 0, and reads what it wrote.
#
#   M   commands
#   32  w0 = m3; acc0 = T(m4); acc1 = T(m6)
#   1   acc0 += A0 W^T, A1 W^T, A2 W^T (A_k = m_k), back to back
#   2   acc1 += A0 W^T; acc0 += A1 W^T; acc1 += A2 W^T, alternating
#   3   T(m8) = BF16(acc0); m10 = E4M3(acc1 x 2^-7)
#   5   T(m12) = BF16(acc1 x 2^4)
#   2   m16..m19 = acc0 (FP32)
#   32  m20..m23 = acc0; m24..m27 = acc1
#   4   acc0 = T(m28); 6: acc1 = m30 (E4M3); 3: acc1 = m32..m35 (FP32); 7: acc0 = A1 W^T
#   32  m36..m39 = acc0; m40..m43 = acc1
#
# DRAM (32-byte words, from the generator): m_r at 32 r for r = 0..7 (A, W, C, D) and for the
# start values of r = 8..43; the results m_r at 2048 + 32 (r - 8).
# VMEM: m_r at KiB r; the results at KiB 56 + r.

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
    dma.load x6, x1, x2, ch2
    LI    x6, 10240
    LI    x1, 0x9000a000
    dma.load x6, x1, x2, ch3
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
    vload m5, 40(x0)
    vload m6, 48(x0)
    vload m7, 56(x0)
    vload m8, 64(x0)
    vload m9, 72(x0)
    vload m10, 80(x0)
    vload m11, 88(x0)
    vload m12, 96(x0)
    vload m13, 104(x0)
    vload m14, 112(x0)
    vload m15, 120(x0)
    vload m16, 128(x0)
    vload m17, 136(x0)
    vload m18, 144(x0)
    vload m19, 152(x0)
    vload m20, 160(x0)
    vload m21, 168(x0)
    vload m22, 176(x0)
    vload m23, 184(x0)
    vload m24, 192(x0)
    vload m25, 200(x0)
    vload m26, 208(x0)
    vload m27, 216(x0)
    vload m28, 224(x0)
    vload m29, 232(x0)
    vload m30, 240(x0)
    vload m31, 248(x0)
    vload m32, 256(x0)
    vload m33, 264(x0)
    vload m34, 272(x0)
    vload m35, 280(x0)
    vload m36, 288(x0)
    vload m37, 296(x0)
    vload m38, 304(x0)
    vload m39, 312(x0)
    vload m40, 320(x0)
    vload m41, 328(x0)
    vload m42, 336(x0)
    vload m43, 344(x0)
    seli  e1, 120
    seli  e2, 131

    vmatpush.weight.mxu1 w0, m3
    vmatpush.acc.bf16.mxu1 acc0, m4
    vmatpush.acc.bf16.mxu1 acc1, m6
    vsetm 1
    vmatmul.acc.mxu1 acc0, m0, w0
    vmatmul.acc.mxu1 acc0, m1, w0
    vmatmul.acc.mxu1 acc0, m2, w0
    vsetm 2
    vmatmul.acc.mxu1 acc1, m0, w0
    vmatmul.acc.mxu1 acc0, m1, w0
    vmatmul.acc.mxu1 acc1, m2, w0
    vsetm 3
    vmatpop.bf16.acc.mxu1 m8, acc0
    vmatpop.fp8.acc.mxu1 m10, acc1, e1
    vsetm 5
    vmatpop.bf16.scaled.acc.mxu1 m12, acc1, e2
    vsetm 2
    vmatpop.fp32.acc.mxu1 m16, acc0
    vsetm 32
    vmatpop.fp32.acc.mxu1 m20, acc0
    vmatpop.fp32.acc.mxu1 m24, acc1
    vsetm 4
    vmatpush.acc.bf16.mxu1 acc0, m28
    vsetm 6
    vmatpush.acc.fp8.mxu1 acc1, m30
    vsetm 3
    vmatpush.acc.fp32.mxu1 acc1, m32
    vsetm 7
    vmatmul.mxu1 acc0, m1, w0
    vsetm 32
    vmatpop.fp32.acc.mxu1 m36, acc0
    vmatpop.fp32.acc.mxu1 m40, acc1

    # Results -> VMEM KiB 64.. -> DRAM word 2048
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
    vstore m40, 768(x0)
    vstore m41, 776(x0)
    vstore m42, 784(x0)
    vstore m43, 792(x0)
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
    LI    x6, 24576
    LI    x1, 0x90018000
    dma.store x1, x6, x2, ch1
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

# mxu_qk_scores.S — Q=X@WQ, K=X@WK, scores=Q@K^T
# @TIMEOUT 300000
# @PYTHON_GEN gen_mxu_qk_scores.py 900

    LI    x5,  0x60000000
    DMA.CONFIG x5, 0
    LI    x12, 1024
    LI    x15, 1024
    SELI  0, 127
    SELI  1, 127

    # VMEM staging: tile A at word 0, tile B at word 256, result at word 512/768

######################################################################
# Q[:, 0:32] = X @ WQ[:, 0:32]
######################################################################
    ADDI  x6, x0, 0
    LI    x10, 0x800
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    VMATMUL.MXU0  0, 0, 0

    LI    x7, 256
    LI    x10, 0xC00
    DMA.LOAD  x7, x10, x12, 1
    DMA.WAIT  1
    LI    x7, 256
    VLOAD 6, x7, 0
    VTRPOSE.XLU 7, 6
    VMATPUSH.W.MXU0  1, 7

    ADDI  x6, x0, 0
    LI    x10, 0x400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    VMATMUL.ACC.MXU0  0, 0, 1
    SELI  0, 127
    VMATPOP.FP8.MXU0  8, 0, 0

    LI    x8, 512
    VSTORE 8, x8, 0
    LI    x8, 512
    LI    x10, 0x2800
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

######################################################################
# Q[:, 32:64]
######################################################################
    ADDI  x6, x0, 0
    LI    x10, 0x1000
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    VMATMUL.MXU0  0, 0, 0

    LI    x7, 256
    LI    x10, 0x1400
    DMA.LOAD  x7, x10, x12, 1
    DMA.WAIT  1
    LI    x7, 256
    VLOAD 6, x7, 0
    VTRPOSE.XLU 7, 6
    VMATPUSH.W.MXU0  1, 7

    ADDI  x6, x0, 0
    LI    x10, 0x400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    VMATMUL.ACC.MXU0  0, 0, 1
    SELI  0, 127
    VMATPOP.FP8.MXU0  9, 0, 0

    LI    x8, 512
    VSTORE 9, x8, 0
    LI    x8, 512
    LI    x10, 0x2C00
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

######################################################################
# K[:, 0:32]
######################################################################
    ADDI  x6, x0, 0
    LI    x10, 0x1800
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    VMATMUL.MXU0  0, 0, 0

    LI    x7, 256
    LI    x10, 0x1C00
    DMA.LOAD  x7, x10, x12, 1
    DMA.WAIT  1
    LI    x7, 256
    VLOAD 6, x7, 0
    VTRPOSE.XLU 7, 6
    VMATPUSH.W.MXU0  1, 7

    ADDI  x6, x0, 0
    LI    x10, 0x400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    VMATMUL.ACC.MXU0  0, 0, 1
    SELI  1, 127
    VMATPOP.FP8.MXU0  10, 1, 0

    LI    x8, 512
    VSTORE 10, x8, 0
    LI    x8, 512
    LI    x10, 0x3000
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

######################################################################
# K[:, 32:64]
######################################################################
    ADDI  x6, x0, 0
    LI    x10, 0x2000
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    VMATMUL.MXU0  0, 0, 0

    LI    x7, 256
    LI    x10, 0x2400
    DMA.LOAD  x7, x10, x12, 1
    DMA.WAIT  1
    LI    x7, 256
    VLOAD 6, x7, 0
    VTRPOSE.XLU 7, 6
    VMATPUSH.W.MXU0  1, 7

    ADDI  x6, x0, 0
    LI    x10, 0x400
    DMA.LOAD  x6, x10, x12, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    VMATMUL.ACC.MXU0  0, 0, 1
    SELI  1, 127
    VMATPOP.FP8.MXU0  11, 1, 0

    LI    x8, 512
    VSTORE 11, x8, 0
    LI    x8, 512
    LI    x10, 0x3400
    DMA.STORE x10, x8, x12, 2
    DMA.WAIT  2

######################################################################
# scores = Q @ K^T
######################################################################
    ADDI  x6, x0, 0
    LI    x10, 0x3000
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VMATPUSH.W.MXU0  0, 4

    LI    x7, 256
    LI    x10, 0x2800
    DMA.LOAD  x7, x10, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    VMATMUL.MXU0  0, 0, 0

    ADDI  x6, x0, 0
    LI    x10, 0x3400
    DMA.LOAD  x6, x10, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    VMATPUSH.W.MXU0  1, 6

    LI    x7, 256
    LI    x10, 0x2C00
    DMA.LOAD  x7, x10, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    VMATMUL.ACC.MXU0  0, 0, 1
    VMATPOP.BF16.MXU0  2, 0

    LI    x8, 512
    VSTORE 2, x8, 0
    LI    x9, 768
    VSTORE 3, x9, 0

    LI    x8, 512
    LI    x10, 0x4000
    DMA.STORE x10, x8, x15, 2
    DMA.WAIT  2

    LI    x9, 768
    LI    x10, 0x4400
    DMA.STORE x10, x9, x15, 3
    DMA.WAIT  3

    LI    x30, 1
    CSRW  x30, 0xC10
    JAL   x0, 0
    
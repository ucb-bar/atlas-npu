# xlu_fanout_branch_reuse.S
# @TIMEOUT 600000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_xlu_fanout_branch_reuse.py
#
# Coverage intent:
#   - XLU transpose result is reused by multiple downstream consumers.
#   - Scalar branch predicate comes from a stored transpose result.
#   - Common follow-on path uses the transposed tile as MXU0 weight.
#   - End-to-end DMA -> VLOAD -> XLU -> VSTORE/scalar read -> branch -> MXU -> DMA.
#
# DRAM byte layout:
#   [0x0000 .. 0x03FF]  A_fp8 activation tile                (1024 B)
#   [0x0400 .. 0x07FF]  B_fp8 raw tile                       (1024 B)
#   [0x1000 .. 0x13FF]  transposed B_fp8 tile               (1024 B)
#   [0x1800 .. 0x1FFF]  MXU0 BF16 result tile               (2048 B)
#
# VMEM word layout:
#   0      A fp8
#   256    B raw fp8
#   512    B^T fp8
#   1024   MXU result lo
#   1280   MXU result hi

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024

    # Load A and B.
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0x0000
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0

    LI    x6, 256
    LI    x10, 0x0400
    DMA.LOAD x6, x10, x12, 1
    DMA.WAIT 1

    # Load A->m0 and B->m4.
    ADDI  x7, x0, 0
    VLOAD 0, x7, 0
    DELAY 33

    LI    x7, 256
    VLOAD 4, x7, 0
    DELAY 33

    # Transpose B into m8.
    VTRPOSE.XLU 8, 4
    DELAY 64

    # Store transpose to VMEM and DRAM for scalar-observable branch predicate.
    LI    x8, 512
    VSTORE 8, x8, 0
    DELAY 33

    LI    x10, 0x1000
    LI    x6, 512
    DMA.STORE x10, x6, x12, 2
    DMA.WAIT 2

    # Scalar reads first word of transposed tile from VMEM byte addr 2048.
    LI    x20, 2048
    LW    x21, x20, 0
    DELAY 1

    BEQ   x21, x0, fail
    ADDI  x29, x0, 7

    # Use the same transposed bank as MXU0 weight.
    VMATPUSH.W.MXU0 0, 8
    DELAY 31
    VMATMUL.MXU0 0, 0, 0
    DELAY 95
    VMATPOP.BF16.MXU0 12, 0
    DELAY 31

    # Store BF16 result pair.
    LI    x8, 1024
    VSTORE 12, x8, 0
    DELAY 33
    VSTORE 13, x8, 8
    DELAY 33

    LI    x12, 2048
    LI    x10, 0x1800
    LI    x6, 1024
    DMA.STORE x10, x6, x12, 3
    DMA.WAIT 3

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

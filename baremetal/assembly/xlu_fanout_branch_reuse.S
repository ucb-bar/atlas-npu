# xlu_fanout_branch_reuse.S
# @TIMEOUT 600000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_xlu_fanout_branch_reuse.py
#
# Coverage intent:
#   - XLU transpose result is reused by multiple downstream consumers.
#   - Scalar branch predicate comes from a stored transpose result.
#   - Common follow-on path uses the transposed tile as MXU1 weight.
#   - End-to-end DMA -> VLOAD -> XLU -> VSTORE/scalar read -> branch -> MXU -> DMA.
#
# DRAM byte layout:
#   [0x0000 .. 0x03FF]  A_fp8 activation tile                (1024 B)
#   [0x0400 .. 0x07FF]  B_fp8 raw tile                       (1024 B)
#   [0x1000 .. 0x13FF]  transposed B_fp8 tile               (1024 B)
#   [0x1800 .. 0x1FFF]  MXU1 BF16 result tile               (2048 B)
#
# VMEM word layout:
#   0      A fp8
#   256    B raw fp8
#   512    B^T fp8
#   1024   MXU result lo
#   1280   MXU result hi

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024

    # Load A and B.
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    LI    x6, 256
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1
    dma.wait ch1

    # Load A->m0 and B->m4.
    ADDI  x7, x0, 0
    vload m0, 0(x7)
    DELAY 33

    LI    x7, 256
    vload m4, 0(x7)
    DELAY 33

    # Transpose B into m8.
    vtrpose.xlu m8, m4
    DELAY 64

    # Store transpose to VMEM and DRAM for scalar-observable branch predicate.
    LI    x8, 512
    vstore m8, 0(x8)
    DELAY 33

    LI    x10, 0x90001000
    LI    x6, 512
    dma.store x10, x6, x12, ch2
    dma.wait ch2

    # Scalar reads first word of transposed tile from VMEM byte addr 2048.
    # The LW result is not in x21 until 5 cycles after issue (2-cycle LSU
    # latency + scalar-core writeback pipeline). The BEQ fires DELAY+2 cycles
    # after the LW, so DELAY must be >= 3 to avoid reading the pre-load
    # (uninitialized) x21 -- which reads as X in gate-level sims.
    LI    x20, 2048
    lw x21, 0(x20)
    DELAY 4

    BEQ   x21, x0, fail
    ADDI  x29, x0, 7

    # Use the same transposed bank as MXU1 weight.
    vmatpush.weight.mxu1 w0, m8
    DELAY 31
    vmatmul.mxu1 acc0, m0, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m12, acc0
    DELAY 31

    # Store BF16 result pair.
    LI    x8, 1024
    vstore m12, 0(x8)
    DELAY 33
    vstore m13, 8(x8)
    DELAY 33

    LI    x12, 2048
    LI    x10, 0x90001800
    LI    x6, 1024
    dma.store x10, x6, x12, ch3
    dma.wait ch3

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

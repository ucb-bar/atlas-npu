# mreg_busy_state_rotator_cross_engine.S
# @TIMEOUT 700000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mreg_busy_state_rotator_cross_engine.py
#
# Coverage intent:
#   - Temporal MREG busy-state diversity across VLOAD, XLU, MXU1, VPU, and VSTORE.
#   - Mid/upper-bank activity with rotated producer-consumer handoff.
#   - Exact end-to-end numeric checking for both MXU and post-VPU results.
#
# DRAM byte layout:
#   [0x0000 .. 0x03FF]  A_fp8 tile                         (1024 B)
#   [0x0400 .. 0x07FF]  B_fp8 raw tile                     (1024 B)
#   [0x1000 .. 0x17FF]  MXU1 BF16 raw result               (2048 B)
#   [0x1800 .. 0x1FFF]  VRELU BF16 post-process result     (2048 B)
#
# VMEM word layout:
#   0      A fp8
#   256    B raw fp8
#   512    MXU raw lo
#   768    MXU raw hi
#   1024   relu lo
#   1280   relu hi

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 1024

    # DMA serialized by construction.
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    LI    x6, 256
    LI    x10, 0x90000400
    dma.load x6, x10, x12, ch1
    dma.wait ch1

    # Load into separated MREG regions.
    ADDI  x7, x0, 0
    vload m24, 0(x7)
    DELAY 33

    LI    x7, 256
    vload m40, 0(x7)
    DELAY 33

    # XLU transpose in upper bank region.
    vtrpose.xlu m56, m40
    DELAY 64

    # MXU1 consume low-bank activation + upper-bank transposed weight.
    vmatpush.weight.mxu1 w0, m56
    DELAY 31
    vmatmul.mxu1 acc0, m24, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m32, acc0
    DELAY 31

    # Post-process with VPU into another bank pair.
    vrelu.bf16 m48, m32
    DELAY 64

    # Store raw BF16 result.
    LI    x8, 512
    vstore m32, 0(x8)
    DELAY 33
    vstore m33, 8(x8)
    DELAY 33

    # Store relu BF16 result.
    LI    x8, 1024
    vstore m48, 0(x8)
    DELAY 33
    vstore m49, 8(x8)
    DELAY 33

    LI    x12, 2048
    LI    x10, 0x90001000
    LI    x6, 512
    dma.store x10, x6, x12, ch2
    dma.wait ch2

    LI    x10, 0x90001800
    LI    x6, 1024
    dma.store x10, x6, x12, ch3
    dma.wait ch3

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

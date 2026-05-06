# mreg_busy_state_rotator_cross_engine.S
# @TIMEOUT 700000
# @DRAM_BASE 0x00000001
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

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x12, 1024

    # DMA serialized by construction.
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0x0000
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0

    LI    x6, 256
    LI    x10, 0x0400
    DMA.LOAD x6, x10, x12, 1
    DMA.WAIT 1

    # Load into separated MREG regions.
    ADDI  x7, x0, 0
    VLOAD 24, x7, 0
    DELAY 33

    LI    x7, 256
    VLOAD 40, x7, 0
    DELAY 33

    # XLU transpose in upper bank region.
    VTRPOSE.XLU 56, 40
    DELAY 64

    # MXU1 consume low-bank activation + upper-bank transposed weight.
    VMATPUSH.W.MXU1 0, 56
    DELAY 31
    VMATMUL.MXU1 0, 24, 0
    DELAY 34
    VMATPOP.BF16.MXU1 32, 0
    DELAY 31

    # Post-process with VPU into another bank pair.
    VRELU 48, 32
    DELAY 64

    # Store raw BF16 result.
    LI    x8, 512
    VSTORE 32, x8, 0
    DELAY 33
    VSTORE 33, x8, 8
    DELAY 33

    # Store relu BF16 result.
    LI    x8, 1024
    VSTORE 48, x8, 0
    DELAY 33
    VSTORE 49, x8, 8
    DELAY 33

    LI    x12, 2048
    LI    x10, 0x1000
    LI    x6, 512
    DMA.STORE x10, x6, x12, 2
    DMA.WAIT 2

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

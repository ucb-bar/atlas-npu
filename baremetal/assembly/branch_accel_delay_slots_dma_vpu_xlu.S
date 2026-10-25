# branch_accel_delay_slots_dma_vpu_xlu.S
# @TIMEOUT 400000
# @DRAM_BASE 0x90000000
#
# Coverage intent:
#   - Control-flow with architecturally visible accelerator side effects in one delay slot.
#   - DMA launch from a taken-BEQ delay slot, with completion waited on after redirect.
#   - VLI + XLU side effects from a not-taken BNE with a single delay-slot write.
#   - VSTORE side effect from a JAL delay slot, proving the slot executes before transfer.
#   - End-to-end dataflow closed through DMA load/store and exact generator comparison.
#
# DRAM byte layout (generator-backed):
#   [0x0000 .. 0x03FF]  FP8 input tile for BEQ-delay-slot DMA load     (1024 B)
#   [0x0400 .. 0x07FF]  DRAM result of JAL-delay-slot VSTORE           (1024 B)
#   [0x0800 .. 0x0BFF]  DRAM result of BNE-delay-slot XLU transpose    (1024 B)
#
# VMEM layout:
#   word 0    / byte    0  : FP8 staging tile loaded by the BEQ delay slot
#   word 256  / byte 1024  : JAL-delay-slot VSTORE output
#   word 512  / byte 2048  : BNE-delay-slot XLU-transpose output
#
# MREG allocation:
#   m20  = FP8 tile loaded by DMA in the BEQ delay slot
#   m24  = VLI.COL-written source bank for XLU
#   m25  = XLU transpose destination
#
# Key scheduling fix vs. prior failing version:
#   - VLI.COL writes m24. An immediately-following VTRPOSE.XLU 25,24 after the branch
#     read m24 while its write was still pending, tripping the ScalarCore MREG RAW
#     hazard assertion. Keep VLI.COL in the BNE delay slot, then issue VTRPOSE.XLU
#     only after a conservative VPU completion delay in fallthrough.
#
    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x8, 1024          # common 1 KiB transfer size

    # VMEM word-address constants
    LI    x6, 0             # VMEM word 0     -> byte 0
    LI    x9, 256           # VMEM word 256   -> byte 1024
    LI    x10, 512          # VMEM word 512   -> byte 2048

    # DRAM byte-offset constants
    LI    x7, 0x90000000  # input tile
    LI    x11, 0x90000400  # output tile 0
    LI    x12, 0x90000800  # output tile 1

# ----------------------------------------------------------------------
# Case 1: taken BEQ; the single delay slot launches DMA.LOAD.
# The redirect target then waits for completion before consuming the data.
# ----------------------------------------------------------------------
    BEQ   x0, x0, beq_target
    dma.load x6, x7, x8, ch0

    ADDI  x28, x0, 101
    JAL   x0, fail
    NOP

beq_target:
    dma.wait ch0
    vload m20, 0(x6)         # m20 <- loaded FP8 tile
    DELAY 33

# ----------------------------------------------------------------------
# Case 2: not-taken BNE; the single delay slot writes m24 with a column-immediate pattern.
# Fallthrough then waits for VLI completion before issuing XLU transpose.
# ----------------------------------------------------------------------
    BNE   x0, x0, bne_unreached
    vli.col m24, 0x3F80

    DELAY 31
    vtrpose.xlu m25, m24
    DELAY 64
    vstore m25, 0(x10)       # store XLU output to VMEM[2048..3071]
    DELAY 33
    dma.store x12, x10, x8, ch1
    dma.wait ch1
    JAL   x0, after_bne_case
    NOP

bne_unreached:
    ADDI  x28, x0, 102
    JAL   x0, fail
    NOP

after_bne_case:
# ----------------------------------------------------------------------
# Case 3: JAL helper; the single delay slot VSTOREs m20 to VMEM word 256,
# and helper then waits conservatively before DMA.STORE reads VMEM.
# ----------------------------------------------------------------------
    JAL   x1, helper_store
    vstore m20, 0(x9)

    JAL   x0, pass
    NOP

helper_store:
helper_body:
    DELAY 32
    dma.store x11, x9, x8, ch2
    dma.wait ch2
    JALR  x0, x1, 0
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

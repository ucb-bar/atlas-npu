# integ_xlu_multi_consumer_bank_sweep.S — XLU transpose feeding MXU and VMEM across rotated banks
# @TIMEOUT 500000
# @PYTHON_GEN gen_integ_xlu_multi_consumer_bank_sweep.py
#
# Iteration 0:
#   A0 fp8 at DRAM 0x0000 -> VMEM 0      -> m0
#   B0 raw fp8 at DRAM 0x0400 -> VMEM 256 -> m4 -> XLU m8 -> MXU0 weight
#   MXU0 result BF16 -> m12/m13 -> VMEM 4096/4352 -> DRAM 0x2000/0x2400
#
# Iteration 1:
#   A1 fp8 at DRAM 0x0800 -> VMEM 32768   -> m16
#   B1 raw fp8 at DRAM 0x0C00 -> VMEM 33024 -> m20 -> XLU m24 -> VMEM 36864 -> DRAM 0x2800
#
# This keeps DMA serialized and uses distinct VMEM/MREG placements per phase.

    ADDI  x28, x0, 1
    LI    x5, 0x90000000
    DMA.CONFIG x5, 0

    LI    x14, 1024
    LI    x15, 2048

    # -------- Iteration 0: transpose -> MXU0 consumer --------
    ADDI  x10, x0, 0        # A0 src
    ADDI  x11, x0, 0        # VMEM word 0
    DMA.LOAD x11, x10, x14, 0
    DMA.WAIT 0
    VLOAD 0, x11, 0
    DELAY 40

    LI    x10, 0x0400       # B0 src
    LI    x11, 256          # VMEM word 256
    DMA.LOAD x11, x10, x14, 1
    DMA.WAIT 1
    VLOAD 4, x11, 0
    DELAY 40

    VTRPOSE.XLU 8, 4
    DELAY 80

    VMATPUSH.W.MXU0 0, 8
    DELAY 40
    VMATMUL.MXU0 0, 0, 0
    DELAY 120
    VMATPOP.BF16.MXU0 12, 0
    DELAY 40

    LI    x11, 4096
    VSTORE 12, x11, 0
    DELAY 40
    VSTORE 13, x11, 8
    DELAY 40

    LI    x10, 0x2000
    DMA.STORE x10, x11, x15, 2
    DMA.WAIT 2

    # -------- Iteration 1: transpose -> VMEM/DMA consumer --------
    LI    x10, 0x0800       # A1 src, loaded only to vary bank occupancy
    LI    x11, 32768
    DMA.LOAD x11, x10, x14, 0
    DMA.WAIT 0
    VLOAD 16, x11, 0
    DELAY 40

    LI    x10, 0x0C00       # B1 src
    LI    x11, 33024
    DMA.LOAD x11, x10, x14, 1
    DMA.WAIT 1
    VLOAD 20, x11, 0
    DELAY 40

    VTRPOSE.XLU 24, 20
    DELAY 80

    LI    x11, 36864
    VSTORE 24, x11, 0
    DELAY 40
    LI    x10, 0x2800
    DMA.STORE x10, x11, x14, 3
    DMA.WAIT 3

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

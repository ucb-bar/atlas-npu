# integ_xlu_multi_consumer_bank_sweep.S — XLU transpose feeding MXU and VMEM across legacy bank-spaced placements
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_integ_xlu_multi_consumer_bank_sweep.py
#
# Iteration 0:
#   A0 fp8 at DRAM 0x0000 -> VMEM 0      -> m0
#   B0 raw fp8 at DRAM 0x0400 -> VMEM 256 -> m4 -> XLU m8 -> MXU0 weight
#   MXU0 result BF16 -> m12/m13 -> VMEM 4096/4352 -> DRAM 0x2000/0x2400
#
# Iteration 1:
#   A1 fp8 at DRAM 0x0800 -> VMEM 65536   -> m16
#   B1 raw fp8 at DRAM 0x0C00 -> VMEM 65792 -> m20 -> XLU m24 -> VMEM 69632 -> DRAM 0x2800
#
# Under current 256 KiB banking, iteration 0 lives in bank 0 and iteration 1
# lives in bank 1, so the XLU fanout now sweeps real VMEM banks again.

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x14, 1024
    LI    x15, 2048

    # -------- Iteration 0: transpose -> MXU0 consumer --------
    LI    x10, 0x90000000  # A0 src
    ADDI  x11, x0, 0        # VMEM word 0
    dma.load x11, x10, x14, ch0
    dma.wait ch0
    vload m0, 0(x11)
    DELAY 33

    LI    x10, 0x90000400  # B0 src
    LI    x11, 256          # VMEM word 256
    dma.load x11, x10, x14, ch1
    dma.wait ch1
    vload m4, 0(x11)
    DELAY 33

    vtrpose.xlu m8, m4
    DELAY 64

    vmatpush.weight.mxu0 w0, m8
    DELAY 31
    vmatmul.mxu0 acc0, m0, w0
    DELAY 95
    vmatpop.bf16.acc.mxu0 m12, acc0
    DELAY 31

    LI    x11, 4096
    vstore m12, 0(x11)
    DELAY 33
    vstore m13, 8(x11)
    DELAY 33

    LI    x10, 0x90002000
    dma.store x10, x11, x15, ch2
    dma.wait ch2

    # -------- Iteration 1: transpose -> VMEM/DMA consumer --------
    LI    x10, 0x90000800  # A1 src, loaded only to vary bank occupancy
    LI    x11, 65536
    dma.load x11, x10, x14, ch0
    dma.wait ch0
    vload m16, 0(x11)
    DELAY 33

    LI    x10, 0x90000C00  # B1 src
    LI    x11, 65792
    dma.load x11, x10, x14, ch1
    dma.wait ch1
    vload m20, 0(x11)
    DELAY 33

    vtrpose.xlu m24, m20
    DELAY 64

    LI    x11, 69632
    vstore m24, 0(x11)
    DELAY 33
    LI    x10, 0x90002800
    dma.store x10, x11, x14, ch3
    dma.wait ch3

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

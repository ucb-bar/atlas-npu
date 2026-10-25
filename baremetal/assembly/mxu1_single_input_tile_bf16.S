# gen_mxu1_single_input_tile_bf16.S — FP8 32×32 matmul C = A @ B
# @TIMEOUT 200000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu1_single_input_tile_bf16.py 32 42

    LI    x12, 1024
    LI    x5, 0x00000000
    dma.config x5, ch0

    # Load B DRAM[1024] → VMEM[0] → TRF m4
    ADDI  x6, x0, 0
    LI    x1, 0x90000400
    dma.load x6, x1, x12, ch1
    dma.wait ch1
    ADDI  x6, x0, 0
    vload m4, 0(x6)

    # Load A DRAM[0] → VMEM[256] → TRF m0
    LI    x7, 256
    LI    x1, 0x90000000
    dma.load x7, x1, x12, ch0
    dma.wait ch0
    LI    x7, 256
    vload m0, 0(x7)

    vtrpose.xlu m5, m4
    DELAY 64
    vmatpush.weight.mxu1 w0, m5
    DELAY 31
    vmatmul.mxu1 acc0, m0, w0
    DELAY 34
    vmatpop.bf16.acc.mxu1 m2, acc0
    DELAY 31

    # VSTORE m2 → VMEM[512], m3 → VMEM[768]
    LI    x8, 512
    vstore m2, 0(x8)
    DELAY 33
    LI    x9, 768
    vstore m3, 0(x9)
    DELAY 33

    # DMA store VMEM[512] → DRAM[2048]
    LI    x8, 512
    LI    x3, 0x90000800
    dma.store x3, x8, x12, ch2
    dma.wait ch2

    # DMA store VMEM[768] → DRAM[3072]
    LI    x9, 768
    LI    x3, 0x90000C00
    dma.store x3, x9, x12, ch3
    dma.wait ch3

    ADDI  x30, x0, 1
    CSRRW x0, 0x800, x30
    ECALL

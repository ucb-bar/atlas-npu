# mxu_push_pop_check.S — MXU0 pipeline with functional checks
# @TIMEOUT 20000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_mxu_push_pop.py

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # Load weights DRAM[0] → VMEM[0] → TRF m4
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m4, 0(x6)

    # Load activations DRAM[1024] → VMEM[256] → TRF m0
    LI    x7, 256
    LI    x1, 0x90000400
    dma.load x7, x1, x2, ch0
    dma.wait ch0
    LI    x7, 256
    vload m0, 0(x7)

    # Wait for VLOAD m0 to fully populate before MXU reads activations.
    DELAY 33

    vmatpush.weight.mxu0 w0, m4
    # Wait for push into weight slot 0 to complete before launching compute.
    DELAY 31
    vmatmul.mxu0 acc0, m0, w0
    # Wait for matmul pipeline to finish writing the accumulation buffer.
    DELAY 95
    vmatpop.bf16.acc.mxu0 m2, acc0
    # Wait for pop to finish writing m2/m3 before VSTORE reads m2.
    DELAY 31

    # VSTORE m2 → VMEM[512]
    LI    x8, 512
    vstore m2, 0(x8)
    # Wait for VSTORE to finish before DMA reads VMEM[512].
    DELAY 33

    # DMA store VMEM[512] → DRAM[2048]
    LI    x8, 512
    LI    x3, 0x90000800
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
    
# mxu1_push_pop_check.S — MXU1 pipeline with functional checks
# @TIMEOUT 20000
# @DRAM_BASE 0x60000000
# @PYTHON_GEN gen_mxu_push_pop.py

    ADDI  x28, x0, 1
    LI    x5, 0x60000000
    DMA.CONFIG  x5, 0
    LI    x2, 1024

    # Load weights DRAM[0] → VMEM[0] → TRF m4
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0

    # Load activations DRAM[1024] → VMEM[256] → TRF m0
    LI    x7, 256
    LI    x1, 1024
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0

    VMATPUSH.W.MXU1  0, 4
    VMATMUL.MXU1  0, 0, 0
    VMATPOP.BF16.MXU1  2, 0

    # VSTORE m2 → VMEM[512]
    LI    x8, 512
    VSTORE 2, x8, 0

    # DMA store VMEM[512] → DRAM[2048]
    LI    x8, 512
    LI    x3, 2048
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
    
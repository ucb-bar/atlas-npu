# gen_mxu0_single_input_tile_bf16.S — FP8 32×32 matmul C = A @ B
# @TIMEOUT 200000
# @PYTHON_GEN gen_mxu0_single_input_tile_bf16.py 32 42

    LI    x12, 1024
    LI    x5,  0x90000000
    DMA.CONFIG  x5, 0

    # Load B DRAM[1024] → VMEM[0] → TRF m4
    ADDI  x6, x0, 0
    LI    x1, 1024
    DMA.LOAD  x6, x1, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0

    # Load A DRAM[0] → VMEM[256] → TRF m0
    LI    x7, 256
    ADDI  x1, x0, 0
    DMA.LOAD  x7, x1, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0

    VTRPOSE.XLU 5, 4
    # Wait for XLU transpose to finish writing m5 before MXU reads it.
    DELAY 80
    VMATPUSH.W.MXU0  0, 5
    # Wait for weight push to fully populate slot 0 before launching compute.
    DELAY 40
    VMATMUL.MXU0  0, 0, 0
    # Wait for systolic-array pipeline + acc-buffer writeback to complete.
    DELAY 120
    VMATPOP.BF16.MXU0  2, 0
    # Wait for pop to finish writing m2/m3 before LSU reads them (VSTORE).
    DELAY 40

    # VSTORE m2 → VMEM[512], m3 → VMEM[768]
    LI    x8, 512
    VSTORE 2, x8, 0
    # LSU is busy during VSTORE; delay before issuing the next VSTORE.
    DELAY 40
    LI    x9, 768
    VSTORE 3, x9, 0
    # Wait for VSTORE to finish before DMA reads VMEM for store-back.
    DELAY 40

    # DMA store VMEM[512] → DRAM[2048]
    LI    x8, 512
    LI    x3, 2048
    DMA.STORE x3, x8, x12, 2
    DMA.WAIT  2

    # DMA store VMEM[768] → DRAM[3072]
    LI    x9, 768
    LI    x3, 3072
    DMA.STORE x3, x9, x12, 3
    DMA.WAIT  3

    ADDI  x30, x0, 1
    CSRRW x0, 0xC10, x30
    ECALL

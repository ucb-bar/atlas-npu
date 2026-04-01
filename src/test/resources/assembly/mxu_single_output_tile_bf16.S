# gen_mxu_single_output_tile_bf16.S — C_tile = A_tile @ B_tile + bias (K=2048)
# @TIMEOUT 500000
# @PYTHON_GEN gen_mxu_single_output_tile_bf16.py 42 1

    LI    x10, 0             # tile counter
    LI    x11, 64            # total K tiles
    LI    x12, 1024          # bytes per tile
    LI    x16, 0             # act DRAM offset
    LI    x17, 65536         # wgt DRAM offset
    LI    x18, 131072        # bias DRAM offset
    LI    x23, 132096        # result DRAM offset
    LI    x24, 1024          # result bytes per bank
    LI    x29, 1             # weight buffer selector
    LI    x30, 1             # const 1

    LI    x5,  0x60000000
    DMA.CONFIG  x5, 0

    # VMEM staging: word 0=weight, 256=activation, 512=bias/result, 768=result

# === Bias preload ============================================
    LI    x6, 512
    DMA.LOAD  x6, x18, x12, 3
    DMA.WAIT  3
    LI    x6, 512
    VLOAD 8, x6, 0
    VMATPUSH.ACC.FP8.MXU0  0, 8

# === Prologue: tile 0 ========================================
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    VMATMUL.ACC.MXU0  0, 0, 0

    ADD   x16, x16, x12
    ADD   x17, x17, x12
    ADDI  x10, x10, 1

# === Main loop ================================================
tile_loop:
    BEQ   x10, x11, epilogue
    NOP
    NOP
    BEQ   x29, x0, path_buf0
    NOP
    NOP

path_buf1:
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 6, x6, 0
    VTRPOSE.XLU 7, 6
    VMATPUSH.W.MXU0  1, 7

    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    VMATMUL.ACC.MXU0  0, 0, 1
    BEQ   x0, x0, advance
    NOP
    NOP

path_buf0:
    ADDI  x6, x0, 0
    DMA.LOAD  x6, x17, x12, 1
    DMA.WAIT  1
    ADDI  x6, x0, 0
    VLOAD 4, x6, 0
    VTRPOSE.XLU 5, 4
    VMATPUSH.W.MXU0  0, 5

    LI    x7, 256
    DMA.LOAD  x7, x16, x12, 0
    DMA.WAIT  0
    LI    x7, 256
    VLOAD 0, x7, 0
    VMATMUL.ACC.MXU0  0, 0, 0

advance:
    ADD   x16, x16, x12
    ADD   x17, x17, x12
    ADDI  x10, x10, 1
    XOR   x29, x29, x30
    BEQ   x0, x0, tile_loop
    NOP
    NOP

# === Epilogue =================================================
epilogue:
    VMATPOP.BF16.MXU0  2, 0

    LI    x8, 512
    VSTORE 2, x8, 0
    LI    x9, 768
    VSTORE 3, x9, 0

    LI    x8, 512
    DMA.STORE x23, x8, x24, 2
    DMA.WAIT  2

    LI    x9, 768
    ADD   x25, x23, x24
    DMA.STORE x25, x9, x24, 3
    DMA.WAIT  3

    ADDI  x30, x0, 1
    CSRRW x0, 0xC10, x30
    JAL   x0, 0
    NOP
    NOP

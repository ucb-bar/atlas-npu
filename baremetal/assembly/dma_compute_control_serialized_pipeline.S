# dma_compute_control_serialized_pipeline.S
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_dma_compute_control_serialized_pipeline.py
#
# Coverage intent:
#   - Serialized DMA software contract inside a scalar loop.
#   - Per-iteration VPU compute followed by scalar-observable predicate handling.
#   - Repeated DMA idle/busy transitions with dependent compute and store-back.
#
# Two iterations, each on one BF16 tile pair:
#   iter0 input at DRAM 0x0000 -> output at 0x2000
#   iter1 input at DRAM 0x0800 -> output at 0x2800
#
# VMEM word layout (reused each iteration):
#   0      input lo
#   256    input hi
#   512    rowsum lo
#   768    rowsum hi
#   1024   final lo
#   1280   final hi

    ADDI  x28, x0, 1

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0

    LI    x2, 0              # loop counter
    LI    x3, 2              # trip count
    LI    x16, 0x0000        # src DRAM byte offset
    LI    x17, 0x2000        # dst DRAM byte offset
    LI    x12, 2048          # bytes per BF16 tile pair
    LI    x18, 0x0800        # per-iter DRAM stride (reg: 0x800 > ADDI signed-12 max → would sign-extend to -2048).

loop:
    BEQ   x2, x3, done
    NOP
    NOP

    ADDI  x6, x0, 0
    DMA.LOAD x6, x16, x12, 0
    DMA.WAIT 0

    ADDI  x7, x0, 0
    VLOAD 0, x7, 0
    DELAY 40
    VLOAD 1, x7, 8
    DELAY 40

    VREDSUM.ROW.BF16 2, 0
    DELAY 150

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40
    VSTORE 3, x8, 8
    DELAY 40

    # Scalar predicate from row-sum footprint.
    LI    x20, 2048
    LW    x21, x20, 0
    DELAY 1

    BLT   x21, x0, use_mov
    ADDI  x29, x0, 3
    ADDI  x30, x0, 4

    VRELU 4, 2
    DELAY 150
    JAL   x0, store_iter
    NOP
    NOP

use_mov:
    VMOV  4, 2
    DELAY 150

store_iter:
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40

    LI    x6, 1024
    DMA.STORE x17, x6, x12, 1
    DMA.WAIT 1

    ADD   x16, x16, x18
    ADD   x17, x17, x18
    ADDI  x2, x2, 1
    JAL   x0, loop
    NOP
    NOP

done:
pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

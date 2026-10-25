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

    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x2, 0              # loop counter
    LI    x3, 2              # trip count
    LI    x16, 0x90000000  # src DRAM byte offset
    LI    x17, 0x90002000  # dst DRAM byte offset
    LI    x12, 2048          # bytes per BF16 tile pair
    LI    x18, 0x0800        # per-iter DRAM stride (reg: 0x800 > ADDI signed-12 max → would sign-extend to -2048).

loop:
    BEQ   x2, x3, done
    NOP

    ADDI  x6, x0, 0
    dma.load x6, x16, x12, ch0
    dma.wait ch0

    ADDI  x7, x0, 0
    vload m0, 0(x7)
    DELAY 33
    vload m1, 8(x7)
    DELAY 33

    vredsum.row.bf16 m2, m0
    DELAY 37

    LI    x8, 512
    vstore m2, 0(x8)
    DELAY 33
    vstore m3, 8(x8)
    DELAY 33

    # Scalar predicate from row-sum footprint.
    LI    x20, 2048
    lw x21, 0(x20)
    DELAY 1

    BLT   x21, x0, use_mov
    ADDI  x29, x0, 3

    vrelu.bf16 m4, m2
    DELAY 64
    JAL   x0, store_iter
    NOP

use_mov:
    vmov m4, m2
    DELAY 64

store_iter:
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33

    LI    x6, 1024
    dma.store x17, x6, x12, ch1
    dma.wait ch1

    ADD   x16, x16, x18
    ADD   x17, x17, x18
    ADDI  x2, x2, 1
    JAL   x0, loop
    NOP

done:
pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

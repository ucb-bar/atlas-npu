# branch_on_accel_result_two_path_compute.S
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_branch_on_accel_result_two_path_compute.py
#
# Coverage intent:
#   - Accelerator result influences scalar control path.
#   - VREDSUM.ROW.BF16 result is stored, scalar-read, and branched on.
#   - Taken path launches a dependent VRELU; untaken path is present but not selected.
#   - Includes a target-independent marker write just before the branch.
#
# DRAM byte layout:
#   [0x0000 .. 0x07FF]  input BF16 tensor pair              (2048 B)
#   [0x0800 .. 0x0FFF]  row-sum broadcast tensor pair       (2048 B)
#   [0x1000 .. 0x17FF]  final chosen-path tensor pair       (2048 B)
#
# VMEM word layout:
#   0      input lo
#   256    input hi
#   512    rowsum lo
#   768    rowsum hi
#   1024   final lo
#   1280   final hi

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x12, 2048

    # Load input tile.
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    # Load input pair into m0/m1.
    ADDI  x7, x0, 0
    vload m0, 0(x7)
    DELAY 33
    vload m1, 8(x7)
    DELAY 33

    # Row reduction -> m2/m3.
    vredsum.row.bf16 m2, m0
    DELAY 37

    # Store reduction result to VMEM so scalar can observe it.
    LI    x8, 512
    vstore m2, 0(x8)
    DELAY 33
    vstore m3, 8(x8)
    DELAY 33

    # Drain the reduction result to DRAM as an intermediate oracle.
    LI    x10, 0x90000800
    LI    x6, 512
    dma.store x10, x6, x12, ch1
    dma.wait ch1

    # Scalar reads first BF16 pair word of the stored rowsum tile.
    LI    x20, 2048          # byte address of VMEM word base 512
    lw x21, 0(x20)
    DELAY 4

    # Generator makes row 0 sum positive, so branch to relu_path is deterministic.
    ADDI  x29, x0, 1         # marker only, on both paths
    BLT   x21, x0, neg_path

relu_path:
    vrelu.bf16 m4, m2
    DELAY 64
    JAL   x0, store_final

neg_path:
    vmov m4, m2
    DELAY 64

store_final:
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33

    LI    x10, 0x90001000
    LI    x6, 1024
    dma.store x10, x6, x12, ch2
    dma.wait ch2

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

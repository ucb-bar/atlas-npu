# branch_on_accel_result_two_path_compute.S
# @TIMEOUT 500000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_branch_on_accel_result_two_path_compute.py
#
# Coverage intent:
#   - Accelerator result influences scalar control path.
#   - VREDSUM.ROW.BF16 result is stored, scalar-read, and branched on.
#   - Taken path launches a dependent VRELU; untaken path is present but not selected.
#   - Includes architecturally visible delay-slot side effects that are target-independent.
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

    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x12, 2048

    # Load input tile.
    ADDI  x6, x0, 0
    ADDI  x10, x0, 0
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0

    # Load input pair into m0/m1.
    ADDI  x7, x0, 0
    VLOAD 0, x7, 0
    DELAY 40
    VLOAD 1, x7, 8
    DELAY 40

    # Row reduction -> m2/m3.
    VREDSUM.ROW.BF16 2, 0
    DELAY 150

    # Store reduction result to VMEM so scalar can observe it.
    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 40
    VSTORE 3, x8, 8
    DELAY 40

    # Drain the reduction result to DRAM as an intermediate oracle.
    LI    x10, 0x0800
    LI    x6, 512
    DMA.STORE x10, x6, x12, 1
    DMA.WAIT 1

    # Scalar reads first BF16 pair word of the stored rowsum tile.
    LI    x20, 2048          # byte address of VMEM word base 512
    LW    x21, x20, 0
    DELAY 1

    # Generator makes row 0 sum positive, so branch to relu_path is deterministic.
    BLT   x21, x0, neg_path
    ADDI  x29, x0, 1         # delay-slot marker only
    ADDI  x30, x0, 2         # delay-slot marker only

relu_path:
    VRELU 4, 2
    DELAY 150
    JAL   x0, store_final
    NOP
    NOP

neg_path:
    VMOV  4, 2
    DELAY 150

store_final:
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40

    LI    x10, 0x1000
    LI    x6, 1024
    DMA.STORE x10, x6, x12, 2
    DMA.WAIT 2

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

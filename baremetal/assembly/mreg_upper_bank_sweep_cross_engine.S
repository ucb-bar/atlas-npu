# mreg_upper_bank_sweep_cross_engine.S — upper-bank MREG sweep across DMA, XLU, MXU1, and VPU
# @TIMEOUT 500000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_mreg_upper_bank_sweep_cross_engine.py
#
# Coverage intent:
#   - Drive reads/writes across the underused upper MREG bank range m48..m63.
#   - Exercise DMA -> VMEM -> VLOAD into high banks.
#   - Exercise XLU transpose on a high FP8 bank pair source/destination.
#   - Exercise MXU1 using high-bank activation + transposed high-bank weight.
#   - Exercise VPU pair op (VRELU) on a BF16 result pair in the upper bank space.
#   - Store two distinct results back to DRAM for exact generator comparison.
#
# DRAM byte layout (generator-backed):
#   0x0000 .. 0x03FF   A_fp8                (1024 B)
#   0x0400 .. 0x07FF   B_fp8 raw            (1024 B)
#   0x0800 .. 0x0FFF   Seed BF16 pair lo/hi (2048 B)
#   0x1000 .. 0x13FF   bank58 passthrough   (1024 B expected)
#   0x1400 .. 0x17FF   bank59 passthrough   (1024 B expected)
#   0x1800 .. 0x1FFF   MXU1 BF16 output     (2048 B expected)
#   0x2000 .. 0x27FF   ReLU output          (2048 B expected)
#
# VMEM word-address layout:
#   0      A_fp8               (1024 B)
#   256    B_fp8 raw           (1024 B)
#   512    seed BF16 low       (1024 B)
#   768    seed BF16 high      (1024 B)
#   1024   scratch store m58   (1024 B)
#   1280   scratch store m59   (1024 B)
#   1536   store MXU BF16 low  (1024 B)
#   1792   store MXU BF16 high (1024 B)
#   2048   store ReLU low      (1024 B)
#   2304   store ReLU high     (1024 B)

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0

    LI    x2, 1024            # common 1 KiB DMA size
    LI    x3, 2048            # common 2 KiB DMA size

    # Conservative branch-delay-slot sanity case with accelerator-independent work.
    ADDI  x20, x0, 0
    BEQ   x0, x0, setup_done
    ADDI  x20, x20, 3
    ADDI  x21, x0, 3
    BNE   x20, x21, fail
    NOP
    ADDI  x28, x28, 1

setup_done:
    # DMA A_fp8 to VMEM[0 words]
    ADDI  x6, x0, 0          # VMEM word addr
    ADDI  x7, x0, 0          # DRAM byte offset
    DMA.LOAD x6, x7, x2, 0
    DMA.WAIT 0

    # DMA B_fp8 raw to VMEM[256 words]
    ADDI  x6, x0, 256
    LI    x7, 0x0400
    DMA.LOAD x6, x7, x2, 0
    DMA.WAIT 0

    # DMA BF16 seed pair to VMEM[512 words] (2048 B covers low+high banks)
    ADDI  x6, x0, 512
    LI    x7, 0x0800
    DMA.LOAD x6, x7, x3, 0
    DMA.WAIT 0

    # VLOAD all upper banks we intend to touch.
    ADDI  x10, x0, 0
    VLOAD 48, x10, 0         # A_fp8 -> m48
    DELAY 40

    ADDI  x10, x0, 256
    VLOAD 50, x10, 0         # B raw -> m50
    DELAY 40

    ADDI  x10, x0, 512
    VLOAD 58, x10, 0         # BF16 seed low -> m58
    DELAY 40
    VLOAD 59, x10, 8         # BF16 seed high -> m59
    DELAY 40

    # High-bank passthrough stores to ensure direct read+write coverage on m58/m59.
    ADDI  x11, x0, 1024
    VSTORE 58, x11, 0
    DELAY 40
    VSTORE 59, x11, 8
    DELAY 40

    # XLU transpose the raw FP8 weight tile from m50 into high bank m52.
    VTRPOSE.XLU 52, 50
    DELAY 80

    # MXU1 consumes upper-bank activation and transposed weight.
    VMATPUSH.W.MXU1 0, 52
    DELAY 40
    VMATMUL.MXU1 0, 48, 0
    DELAY 120
    VMATPOP.BF16.MXU1 60, 0
    DELAY 40

    # VPU post-process on upper BF16 pair.
    VRELU 62, 60
    DELAY 150

    # Store MXU BF16 result pair and VPU ReLU pair to VMEM.
    ADDI  x12, x0, 1536
    VSTORE 60, x12, 0
    DELAY 40
    VSTORE 61, x12, 8
    DELAY 40

    ADDI  x12, x0, 2048
    VSTORE 62, x12, 0
    DELAY 40
    VSTORE 63, x12, 8
    DELAY 40

    # Drain all checked regions to DRAM.
    LI    x13, 0x1000
    ADDI  x14, x0, 1024
    DMA.STORE x13, x14, x2, 1
    DMA.WAIT 1

    LI    x13, 0x1400
    ADDI  x14, x0, 1280
    DMA.STORE x13, x14, x2, 1
    DMA.WAIT 1

    LI    x13, 0x1800
    ADDI  x14, x0, 1536
    DMA.STORE x13, x14, x3, 1
    DMA.WAIT 1

    LI    x13, 0x2000
    ADDI  x14, x0, 2048
    DMA.STORE x13, x14, x3, 1
    DMA.WAIT 1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

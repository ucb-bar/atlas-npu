# vpu_sqcb_branch_blast.S - VSQUARE/VCUBE every-branch lattice
# @TIMEOUT 700000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_vpu_sqcb_branch_blast.py
#
# Coverage intent (cov_overall.modinfo.txt):
#   - core.vpu.core.sqcb   branch 23.30%   cond 42.86%   <-- primary target
#
# The square/cube lane box (`SquareCubeVec.scala` + `VectorParam.squareCube`)
# has two parallel result chains, one for square and one for cube, each a
# 3-arm `when/elsewhen/otherwise` over inputs and per-arm result flags:
#
#   when (isInputZero || isInputSubnormal || isInputNaN || isResultZero) ...
#   elsewhen (isInputInf || isResultInf) ...
#   otherwise (normal path) ...
#
# Plus internal `for (i <- 0 until N) { when(mant(i)) { highestIdx := i.U } }`
# bit-walks over `squareMant` (16 bits) and `cubeMant` (24 bits), and
# multi-arm `when` chains for `squareExtraExp` (2 arms) and `cubeExtraExp`
# (3 arms).  Existing tests run square/cube on a single mixed-class tile,
# which lights up the normal arms but rarely fires the result-zero /
# result-inf branches simultaneously with the input-class branches the
# helper fans across.
#
# Strategy: 4 input tiles, each 32x32 BF16, each constructed to *force*
# specific helper conditions, then run VSQUARE.BF16 AND VCUBE.BF16 on
# every tile (8 passes total).
#
#   Tile 0  (DRAM 0x0000): dense mantissa walk, mid-range exponents,
#                          both signs                       -> normal arms
#   Tile 1  (DRAM 0x0800): tiny magnitudes (biased exp 1..63), both
#                          signs, dense mantissa walk      -> isSquareResultZero AND
#                                                              isCubeResultZero arms
#   Tile 2  (DRAM 0x1000): huge magnitudes (biased exp 192..254), both
#                          signs, dense mantissa walk      -> isSquareResultInf AND
#                                                              isCubeResultInf arms
#   Tile 3  (DRAM 0x1800): special-class lattice
#                            +/-zero, +/-subnormal, +/-inf, NaN payloads,
#                            x near 1.0, threshold mantissas -> isInputZero,
#                                                                isInputSubnormal,
#                                                                isInputNaN,
#                                                                isInputInf arms
#
# For every tile we run:
#       op_idx 0:  VSQUARE.BF16 4, 0
#       op_idx 1:  VCUBE.BF16   4, 0
# Output DRAM byte = 0x4000 + (tile_idx * 2 + op_idx) * 0x800
#
# VMEM word layout:
#   0      input tile staging  (512 words = 2 KiB) feeds m0,m1
#   1024   output pair staging (512 words = 2 KiB) drained from m4,m5

    ADDI  x28, x0, 1
    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 2048

# ======================================================================
# TILE 0 - dense mantissa walk, mid-range exponents
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x0000
    DMA.LOAD x6, x1, x2, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Tile 0 / VSQUARE.BF16
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x4000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 0 / VCUBE.BF16
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x4800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

# ======================================================================
# TILE 1 - tiny magnitudes (drives isSquareResultZero, isCubeResultZero)
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x0800
    DMA.LOAD x6, x1, x2, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Tile 1 / VSQUARE.BF16
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x5000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 1 / VCUBE.BF16
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x5800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

# ======================================================================
# TILE 2 - huge magnitudes (drives isSquareResultInf, isCubeResultInf)
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x1000
    DMA.LOAD x6, x1, x2, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Tile 2 / VSQUARE.BF16
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x6000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 2 / VCUBE.BF16
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x6800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

# ======================================================================
# TILE 3 - special-class lattice (zero/subnormal/inf/NaN/etc.)
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x1800
    DMA.LOAD x6, x1, x2, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Tile 3 / VSQUARE.BF16
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x7000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 3 / VCUBE.BF16
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x7800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

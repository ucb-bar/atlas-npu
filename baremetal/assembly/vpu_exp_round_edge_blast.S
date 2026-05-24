# vpu_exp_round_edge_blast.S - drive HardFloat rounding edges in VPU EXP
# @TIMEOUT 700000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_exp_round_edge_blast.py
#
# Coverage intent (cov_overall.modinfo.txt):
#   - core.vpu.core.exp                  toggle 41.78%
#   - core.vpu.core.exp.lut              toggle 11.66%
#   - core.vpu.core.exp.roundToRecFn_*   cond   49.3% (16 instances)
#
# The 16 `roundToRecFn_*` instances inside `core.vpu.core.exp` are
# HardFloat `roundAnyRawFNToRecFN` rounders. Their conditional coverage
# only climbs when stimulus drives every data-dependent branch inside
# the rounder: overflow detected before vs after rounding, signed-zero
# vs subnormal underflow, exact tie vs inexact, sticky bit set vs not,
# negative vs positive result.
#
# Existing tests drive EXP with general-purpose inputs but rarely walk
# the *boundary* regions where these branches flip. This test feeds
# four tiles, each chosen to land EXP/EXP2 outputs on a different
# rounder edge:
#
#   Tile 0 (DRAM 0x0000): overflow boundary
#       VEXP:   x in [88.0, 89.0]   -> result spans the 2^128 cliff
#       VEXP2:  x in [127.0, 128.0] -> result spans the same cliff
#
#   Tile 1 (DRAM 0x0800): underflow boundary
#       VEXP:   x in [-94.0, -91.0] -> result hits BF16 subnormal /
#                                       flush-to-zero region
#       VEXP2:  x in [-149.0, -120.0]
#
#   Tile 2 (DRAM 0x1000): mantissa-bit walk in mid range
#       x = +/- (1 + frac/128) * 2^k  for k in [-3, 3]
#       Visits every 7-bit mantissa value at every reasonable
#       exponent; drives the LUT interpolation `r_lower` field over
#       its full range and the rounder's sticky bit equally over set
#       and unset.
#
#   Tile 3 (DRAM 0x1800): tiny inputs near zero
#       x = +/- 2^-k for k in [-1, 30] -> all results round to ~1.0
#       with very small inexact bits, exercising the rounder's
#       "barely inexact" path that the other tiles do not.
#
# Per-tile op list: VEXP, VEXP2 (2 ops). 4 tiles x 2 ops = 8 passes,
# each writing 2 KiB BF16 output.
#
#   pass_idx   = tile_idx * 2 + op_idx
#   DRAM byte  = 0x4000 + pass_idx * 0x800
#
# VMEM word layout:
#   0      input tile staging  (512 words = 2 KiB) feeds m0,m1
#   1024   output pair staging (512 words = 2 KiB) drained from m4,m5

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 2048

# ======================================================================
# TILE 0 - overflow boundary
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    DMA.LOAD x6, x1, x2, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Tile 0 / VEXP
    VEXP 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90004000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 0 / VEXP2
    VEXP2 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90004800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

# ======================================================================
# TILE 1 - underflow boundary
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x90000800
    DMA.LOAD x6, x1, x2, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Tile 1 / VEXP
    VEXP 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90005000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 1 / VEXP2
    VEXP2 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90005800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

# ======================================================================
# TILE 2 - mantissa-bit walk in mid range
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x90001000
    DMA.LOAD x6, x1, x2, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Tile 2 / VEXP
    VEXP 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90006000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 2 / VEXP2
    VEXP2 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90006800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

# ======================================================================
# TILE 3 - tiny inputs near zero (x ~ 0 -> exp(x) ~ 1)
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x90001800
    DMA.LOAD x6, x1, x2, 0
    DMA.WAIT 0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Tile 3 / VEXP
    VEXP 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90007000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 3 / VEXP2
    VEXP2 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90007800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

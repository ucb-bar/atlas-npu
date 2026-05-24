# vpu_lut_octave_oddexp_sweep.S — wide-octave/oddExp LUT-input sweep
# @TIMEOUT 600000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_lut_octave_oddexp_sweep.py
#
# Coverage intent (cov_overall.modinfo.txt):
#   - core.vpu.core.sqrt.lut    toggle 5.21%   branch 31.82%   <-- target
#   - core.vpu.core.log2.lut    toggle 3.37%
#   - core.vpu.core.rcp.lut     toggle 3.43%
#   - core.vpu.core.exp.lut     toggle 11.66%
#   - core.vpu.core.tanh.lut    toggle 11.66%
#
# Sqrt's `oddExp` flag is `!exponentLowBitIndex`, i.e. flips on every
# unit step of the BF16 biased exponent. Rcp's special-case lattice
# branches differently from sqrt's. Log's `isNeg = exp < bias` has its
# own dedicated arithmetic path. Existing tests (`vpu_raw_lut_class_stress`,
# `vpu_lut_toggle_sweep`) cluster their inputs around 8 fixed exponents,
# leaving most of the 254 valid biased-exponent values unvisited. This
# test deliberately walks ~50 distinct exponents per tile, alternates
# signs every lane, and salts each tile with the special-case lattice
# (signed zero / subnormal / inf / NaN-payload) so every helper branch
# in `lutFixedToBf16{Sqrt,Rcp,Log}` fires repeatedly.
#
# Each tile is 32x32 BF16 = 2 KiB and is built directly from explicit
# `(sign, exp, mant)` triples by the generator; we don't try to express
# them as floats because we want exact bit-pattern control over inputs
# the float-domain helpers cannot represent (e.g. NaN payloads, the
# largest finite subnormal).
#
# DRAM layout:
#     0x0000 .. 0x07FF   tile 0  (oddExp branch dense; |x| << 1)
#     0x0800 .. 0x0FFF   tile 1  (oddExp branch dense; |x| ~ 1, mixed sign)
#     0x1000 .. 0x17FF   tile 2  (full octave ladder, both signs)
#     0x1800 .. 0x1FFF   tile 3  (special-case + saturating lattice)
#     0x4000 .. 0x47FF   tile 0 / VSQRT output    (pass 0)
#     0x4800 .. 0x4FFF   tile 0 / VLOG2 output    (pass 1)
#     0x5000 .. 0x57FF   tile 0 / VRECIP output   (pass 2)
#     0x5800 .. 0x5FFF   tile 0 / VEXP output     (pass 3)
#     0x6000 .. 0x67FF   tile 0 / VTANH output    (pass 4)
#     ... per tile, output stride = 0x2800 (5 ops x 0x800)
#
# pass_idx = tile_idx * 5 + op_idx
# DRAM byte offset = 0x4000 + pass_idx * 0x800
#
# VMEM layout (word addresses):
#     0       input tile staging  (512 words = 2 KiB) feeds m0,m1
#     1024    output pair staging (512 words = 2 KiB) drained from m4,m5
#
# MREG layout:
#     m0,m1   source BF16 pair
#     m4,m5   VPU result pair

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 2048             # 2 KiB per tile / output

# ======================================================================
# TILE 0 - oddExp branch dense, |x| around 1.0
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

    # Tile 0 / VSQRT
    VSQRT 4, 0
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

    # Tile 0 / VLOG2
    VLOG2 4, 0
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

    # Tile 0 / VRECIP.BF16
    VRECIP.BF16 4, 0
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

    # Tile 0 / VEXP
    VEXP 4, 0
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

    # Tile 0 / VTANH
    VTANH 4, 0
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

# ======================================================================
# TILE 1 - oddExp branch dense, mixed sign just below/above 1
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

    # Tile 1 / VSQRT
    VSQRT 4, 0
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

    # Tile 1 / VLOG2
    VLOG2 4, 0
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

    # Tile 1 / VRECIP.BF16
    VRECIP.BF16 4, 0
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

    # Tile 1 / VEXP
    VEXP 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90008000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 1 / VTANH
    VTANH 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90008800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

# ======================================================================
# TILE 2 - full ~50-octave ladder, both signs
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

    # Tile 2 / VSQRT
    VSQRT 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90009000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 2 / VLOG2
    VLOG2 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90009800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 2 / VRECIP.BF16
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x9000A000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 2 / VEXP
    VEXP 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x9000A800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 2 / VTANH
    VTANH 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x9000B000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

# ======================================================================
# TILE 3 - special-case lattice (zero, subnormal, inf, NaN-coded, sat)
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

    # Tile 3 / VSQRT
    VSQRT 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x9000B800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 3 / VLOG2
    VLOG2 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x9000C000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 3 / VRECIP.BF16
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x9000C800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 3 / VEXP
    VEXP 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x9000D000
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

    # Tile 3 / VTANH
    VTANH 4, 0
    DELAY 64
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x9000D800
    DMA.STORE x10, x9, x2, 1
    DMA.WAIT 1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

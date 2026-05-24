# vpu_raw_lut_class_stress.S - raw BF16 VPU LUT/class toggle stress
# @TIMEOUT 900000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_raw_lut_class_stress.py
#
# Coverage intent:
#   - Drive raw BF16 mantissa/exponent/sign space into the weak LUT leaves:
#       sqrt.lut, log2.lut, exp.lut, tanh.lut, rcp.lut
#   - Exercise square/cube special branches: zero, subnormal, normal,
#     underflow, overflow, inf, NaN, and signed cube.
#   - Avoid float-domain stimulus blind spots by using generator-built BF16
#     bit patterns as the test's source of truth.
#
# DRAM layout:
#   Inputs:  five 2 KiB BF16 tensor pairs at 0x0000, 0x0800, ... 0x2000
#   Outputs: 40 result tensor pairs starting at 0x3000, stride 0x0800
#
# Per input tile, op order is:
#   VSQRT, VLOG2, VRECIP.BF16, VEXP, VEXP2, VTANH, VSQUARE.BF16, VCUBE.BF16

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 2048              # input tile bytes
    LI    x3, 2048              # output tile bytes

# ----------------------------------------------------------------------
# Tile 0: full mantissa walk across representative normal exponents
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x90003000
    JAL   x1, store_pair
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x90003800
    JAL   x1, store_pair
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x90004000
    JAL   x1, store_pair
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x90004800
    JAL   x1, store_pair
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x90005000
    JAL   x1, store_pair
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x90005800
    JAL   x1, store_pair
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90006000
    JAL   x1, store_pair
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90006800
    JAL   x1, store_pair
    NOP

# ----------------------------------------------------------------------
# Tile 1: exponent ladder with scrambled mantissas and alternating signs
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x90007000
    JAL   x1, store_pair
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x90007800
    JAL   x1, store_pair
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x90008000
    JAL   x1, store_pair
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x90008800
    JAL   x1, store_pair
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x90009000
    JAL   x1, store_pair
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x90009800
    JAL   x1, store_pair
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000A000
    JAL   x1, store_pair
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000A800
    JAL   x1, store_pair
    NOP

# ----------------------------------------------------------------------
# Tile 2: explicit zeros, subnormals, min/max normal, inf, NaN, sq/cube edges
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x9000B000
    JAL   x1, store_pair
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x9000B800
    JAL   x1, store_pair
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000C000
    JAL   x1, store_pair
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x9000C800
    JAL   x1, store_pair
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x9000D000
    JAL   x1, store_pair
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x9000D800
    JAL   x1, store_pair
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000E000
    JAL   x1, store_pair
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000E800
    JAL   x1, store_pair
    NOP

# ----------------------------------------------------------------------
# Tile 3: tanh address bins plus saturation and tiny-value behavior
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x90001800
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x9000F000
    JAL   x1, store_pair
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x9000F800
    JAL   x1, store_pair
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x90010000
    JAL   x1, store_pair
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x90010800
    JAL   x1, store_pair
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x90011000
    JAL   x1, store_pair
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x90011800
    JAL   x1, store_pair
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90012000
    JAL   x1, store_pair
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90012800
    JAL   x1, store_pair
    NOP

# ----------------------------------------------------------------------
# Tile 4: exp thresholds and square/cube normalization carry edges
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x90002000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x90013000
    JAL   x1, store_pair
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x90013800
    JAL   x1, store_pair
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x90014000
    JAL   x1, store_pair
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x90014800
    JAL   x1, store_pair
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x90015000
    JAL   x1, store_pair
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x90015800
    JAL   x1, store_pair
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90016000
    JAL   x1, store_pair
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90016800
    JAL   x1, store_pair
    NOP

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

store_pair:
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x9, 1024
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT 1
    JALR  x0, x1, 0
    NOP

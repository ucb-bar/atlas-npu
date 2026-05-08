# vpu_raw_lut_class_stress.S - raw BF16 VPU LUT/class toggle stress
# @TIMEOUT 900000
# @DRAM_BASE 0x00000001
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
    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 2048              # input tile bytes
    LI    x3, 2048              # output tile bytes

# ----------------------------------------------------------------------
# Tile 0: full mantissa walk across representative normal exponents
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x0000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x3000
    JAL   x1, store_pair
    NOP
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x3800
    JAL   x1, store_pair
    NOP
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x4000
    JAL   x1, store_pair
    NOP
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x4800
    JAL   x1, store_pair
    NOP
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x5000
    JAL   x1, store_pair
    NOP
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x5800
    JAL   x1, store_pair
    NOP
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x6000
    JAL   x1, store_pair
    NOP
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x6800
    JAL   x1, store_pair
    NOP
    NOP

# ----------------------------------------------------------------------
# Tile 1: exponent ladder with scrambled mantissas and alternating signs
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x0800
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x7000
    JAL   x1, store_pair
    NOP
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x7800
    JAL   x1, store_pair
    NOP
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x8000
    JAL   x1, store_pair
    NOP
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x8800
    JAL   x1, store_pair
    NOP
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x9000
    JAL   x1, store_pair
    NOP
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x9800
    JAL   x1, store_pair
    NOP
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0xA000
    JAL   x1, store_pair
    NOP
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0xA800
    JAL   x1, store_pair
    NOP
    NOP

# ----------------------------------------------------------------------
# Tile 2: explicit zeros, subnormals, min/max normal, inf, NaN, sq/cube edges
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x1000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0xB000
    JAL   x1, store_pair
    NOP
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0xB800
    JAL   x1, store_pair
    NOP
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0xC000
    JAL   x1, store_pair
    NOP
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0xC800
    JAL   x1, store_pair
    NOP
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0xD000
    JAL   x1, store_pair
    NOP
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0xD800
    JAL   x1, store_pair
    NOP
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0xE000
    JAL   x1, store_pair
    NOP
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0xE800
    JAL   x1, store_pair
    NOP
    NOP

# ----------------------------------------------------------------------
# Tile 3: tanh address bins plus saturation and tiny-value behavior
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x1800
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0xF000
    JAL   x1, store_pair
    NOP
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0xF800
    JAL   x1, store_pair
    NOP
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x10000
    JAL   x1, store_pair
    NOP
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x10800
    JAL   x1, store_pair
    NOP
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x11000
    JAL   x1, store_pair
    NOP
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x11800
    JAL   x1, store_pair
    NOP
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x12000
    JAL   x1, store_pair
    NOP
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x12800
    JAL   x1, store_pair
    NOP
    NOP

# ----------------------------------------------------------------------
# Tile 4: exp thresholds and square/cube normalization carry edges
# ----------------------------------------------------------------------
    ADDI  x6, x0, 0
    LI    x10, 0x2000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x13000
    JAL   x1, store_pair
    NOP
    NOP
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x13800
    JAL   x1, store_pair
    NOP
    NOP
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x14000
    JAL   x1, store_pair
    NOP
    NOP
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x14800
    JAL   x1, store_pair
    NOP
    NOP
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x15000
    JAL   x1, store_pair
    NOP
    NOP
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x15800
    JAL   x1, store_pair
    NOP
    NOP
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x16000
    JAL   x1, store_pair
    NOP
    NOP
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x16800
    JAL   x1, store_pair
    NOP
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
    NOP

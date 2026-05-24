# vpu_lut_bitflip_megasweep.S - bit-flip LUT toggle megasweep for VPU coverage
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_lut_bitflip_megasweep.py
#
# Coverage intent (atlasTile_coverage_report.md targets):
#   core.vpu.core.sqrt.lut    TOGGLE  5.21%
#   core.vpu.core.log2.lut    TOGGLE  3.37%
#   core.vpu.core.exp.lut     TOGGLE 11.66%
#   core.vpu.core.tanh.lut    TOGGLE 11.66%
#   core.vpu.core.rcp.lut     TOGGLE  3.43%
#   core.vpu.core.addSubSum.adders_*    TOGGLE ~54%
#   core.vpu.core.csum.adders_*         TOGGLE ~70%
#   core.vpu.core.exp.roundToRecFn_*    TOGGLE ~70%
#
# Strategy:
#   Layers on top of vpu_lut_toggle_sweep.S and vpu_raw_lut_class_stress.S
#   by feeding tiles purpose-built to maximise PER-LANE PER-CYCLE bit
#   transitions in the LUT result registers and the lutFixedToBf16
#   conversion paths. Every column of every input tile drives a single
#   hardware lane through 32 cycles of inputs chosen to:
#     - sweep the 7-bit mantissa LUT address through bit-flipping pairs
#     - alternate the input exponent LSB to toggle SqrtLUT.oddExp
#     - alternate the input sign to toggle RcpLUT.neg
#     - keep wide exponent and class diversity to drive the BF16
#       conversion path in lutFixedToBf16Sqrt/Rcp/Log
#
#   After the unary sweep, a chain phase reuses Tile A through five LUT
#   pipes (VLOG2 + VSQRT + VEXP2 + VRECIP.BF16 + VTANH) and combines the
#   products through VMUL.BF16 / VADD.BF16 / VSUB.BF16,
#   VREDSUM.ROW.BF16, VFP8PACK + VFP8UNPACK, and VREDSUM.BF16 (column
#   reduce). This drives the AddSubSum / ColAdd reduction trees and the
#   FP8 datapath with LUT-derived (not constant) operands.
#
# DRAM layout (DRAM_BASE = 0x90000000):
#
#   Inputs (4 tiles, 2 KiB each):
#     0x00000  Tile A  walk_pos       all-positive mantissa zig-zag
#     0x00800  Tile B  walk_signed    walk_pos with alternating sign
#     0x01000  Tile C  class_lattice  signed special-class lattice
#     0x01800  Tile D  magnitude_alt  alternating large/small magnitudes
#
#   Unary outputs (32 tiles x 2 KiB):
#     base = 0x02000 + (tile_idx * 8 + op_idx) * 0x800
#     op order: VSQRT, VLOG2, VRECIP.BF16, VEXP, VEXP2, VTANH,
#               VSQUARE.BF16, VCUBE.BF16
#
#   Chain outputs (Tile A used for chain inputs):
#     0x12000  m28/m29  VMUL.BF16(VLOG2(A), VSQRT(A))      2 KiB
#     0x12800  m30/m31  VADD.BF16(VEXP2(A), VRECIP.BF16(A)) 2 KiB
#     0x13000  m32/m33  VSUB.BF16(VTANH(A), MUL)            2 KiB
#     0x13800  m34/m35  VREDSUM.ROW.BF16(ADD)               2 KiB
#     0x14000  m36      VFP8PACK(ADD,    scale=0x7B)        1 KiB
#     0x14400  m38/m39  VFP8UNPACK(PACK, scale=0x86)        2 KiB
#     0x14C00  m40/m41  VREDSUM.BF16(UNPACK)                2 KiB
#
# VMEM staging:
#   word 0..511     input tile (2 KiB)  -> m0/m1
#   word 1024..1535 output staging      -> drained from VPU result mreg(s)
#
# MREG allocation:
#   m0/m1   current input bank pair (overwritten per tile load)
#   m4/m5   unary result scratch (target of all 32 unary ops)
#   m8/m9   chain log2(A)
#   m12/m13 chain sqrt(A)
#   m16/m17 chain exp2(A)
#   m20/m21 chain rcp(A)
#   m24/m25 chain tanh(A)
#   m28/m29 chain mul(log2, sqrt)
#   m30/m31 chain add(exp2, rcp)
#   m32/m33 chain sub(tanh, mul)
#   m34/m35 chain row-sum(add)
#   m36     chain fp8pack(add)
#   m38/m39 chain fp8unpack(pack)
#   m40/m41 chain col-sum(unpack)
#
# Scale registers:
#   e3 = 0x7B  (pack   scale e8m0)
#   e4 = 0x86  (unpack scale e8m0)

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 2048               # 2 KiB DMA size (input tiles, paired outputs)
    LI    x3, 2048               # 2 KiB DMA size for paired output stores

# ======================================================================
# Tile A (walk_pos) at DRAM 0x0000
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ---- Tile A / VSQRT       -> 0x2000 ----
    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x90002000
    JAL   x1, store_pair
    NOP
    # ---- Tile A / VLOG2       -> 0x2800 ----
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x90002800
    JAL   x1, store_pair
    NOP
    # ---- Tile A / VRECIP.BF16 -> 0x3000 ----
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x90003000
    JAL   x1, store_pair
    NOP
    # ---- Tile A / VEXP        -> 0x3800 ----
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x90003800
    JAL   x1, store_pair
    NOP
    # ---- Tile A / VEXP2       -> 0x4000 ----
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x90004000
    JAL   x1, store_pair
    NOP
    # ---- Tile A / VTANH       -> 0x4800 ----
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x90004800
    JAL   x1, store_pair
    NOP
    # ---- Tile A / VSQUARE.BF16 -> 0x5000 ----
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90005000
    JAL   x1, store_pair
    NOP
    # ---- Tile A / VCUBE.BF16   -> 0x5800 ----
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90005800
    JAL   x1, store_pair
    NOP

# ======================================================================
# Tile B (walk_signed) at DRAM 0x0800
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ---- Tile B / VSQRT       -> 0x6000 ----
    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x90006000
    JAL   x1, store_pair
    NOP
    # ---- Tile B / VLOG2       -> 0x6800 ----
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x90006800
    JAL   x1, store_pair
    NOP
    # ---- Tile B / VRECIP.BF16 -> 0x7000 ----
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x90007000
    JAL   x1, store_pair
    NOP
    # ---- Tile B / VEXP        -> 0x7800 ----
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x90007800
    JAL   x1, store_pair
    NOP
    # ---- Tile B / VEXP2       -> 0x8000 ----
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x90008000
    JAL   x1, store_pair
    NOP
    # ---- Tile B / VTANH       -> 0x8800 ----
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x90008800
    JAL   x1, store_pair
    NOP
    # ---- Tile B / VSQUARE.BF16 -> 0x9000 ----
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90009000
    JAL   x1, store_pair
    NOP
    # ---- Tile B / VCUBE.BF16   -> 0x9800 ----
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90009800
    JAL   x1, store_pair
    NOP

# ======================================================================
# Tile C (class_lattice) at DRAM 0x1000
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ---- Tile C / VSQRT       -> 0xA000 ----
    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x9000A000
    JAL   x1, store_pair
    NOP
    # ---- Tile C / VLOG2       -> 0xA800 ----
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x9000A800
    JAL   x1, store_pair
    NOP
    # ---- Tile C / VRECIP.BF16 -> 0xB000 ----
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000B000
    JAL   x1, store_pair
    NOP
    # ---- Tile C / VEXP        -> 0xB800 ----
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x9000B800
    JAL   x1, store_pair
    NOP
    # ---- Tile C / VEXP2       -> 0xC000 ----
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x9000C000
    JAL   x1, store_pair
    NOP
    # ---- Tile C / VTANH       -> 0xC800 ----
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x9000C800
    JAL   x1, store_pair
    NOP
    # ---- Tile C / VSQUARE.BF16 -> 0xD000 ----
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000D000
    JAL   x1, store_pair
    NOP
    # ---- Tile C / VCUBE.BF16   -> 0xD800 ----
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000D800
    JAL   x1, store_pair
    NOP

# ======================================================================
# Tile D (magnitude_alt) at DRAM 0x1800
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90001800
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ---- Tile D / VSQRT       -> 0xE000 ----
    VSQRT 4, 0
    DELAY 64
    LI    x10, 0x9000E000
    JAL   x1, store_pair
    NOP
    # ---- Tile D / VLOG2       -> 0xE800 ----
    VLOG2 4, 0
    DELAY 64
    LI    x10, 0x9000E800
    JAL   x1, store_pair
    NOP
    # ---- Tile D / VRECIP.BF16 -> 0xF000 ----
    VRECIP.BF16 4, 0
    DELAY 64
    LI    x10, 0x9000F000
    JAL   x1, store_pair
    NOP
    # ---- Tile D / VEXP        -> 0xF800 ----
    VEXP 4, 0
    DELAY 64
    LI    x10, 0x9000F800
    JAL   x1, store_pair
    NOP
    # ---- Tile D / VEXP2       -> 0x10000 ----
    VEXP2 4, 0
    DELAY 64
    LI    x10, 0x90010000
    JAL   x1, store_pair
    NOP
    # ---- Tile D / VTANH       -> 0x10800 ----
    VTANH 4, 0
    DELAY 64
    LI    x10, 0x90010800
    JAL   x1, store_pair
    NOP
    # ---- Tile D / VSQUARE.BF16 -> 0x11000 ----
    VSQUARE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90011000
    JAL   x1, store_pair
    NOP
    # ---- Tile D / VCUBE.BF16   -> 0x11800 ----
    VCUBE.BF16 4, 0
    DELAY 64
    LI    x10, 0x90011800
    JAL   x1, store_pair
    NOP

# ======================================================================
# Chain phase
#   Reload Tile A into m0/m1, then drive five LUT pipes into binary,
#   reduction, FP8 pack/unpack, and column reduce so the AddSubSum,
#   ColAdd, and FP8 datapaths see LUT-derived operands.
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    DMA.LOAD x6, x10, x2, 0
    DMA.WAIT 0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Five LUT pipes off the same Tile A pair. Each is sequenced into
    # its own destination bank pair, separated by a full latency window
    # so the next op can read a fresh m0/m1 / write its own dest.
    VLOG2  8, 0
    DELAY 64
    VSQRT 12, 0
    DELAY 64
    VEXP2 16, 0
    DELAY 64
    VRECIP.BF16 20, 0
    DELAY 64
    VTANH 24, 0
    DELAY 64

    # Binary combinations feed AddSubSum / Mul-Rec adders with non-trivial
    # mantissa/exponent diversity (unlike constant-input arithmetic tests).
    VMUL.BF16 28,  8, 12
    DELAY 64
    VADD.BF16 30, 16, 20
    DELAY 64
    VSUB.BF16 32, 24, 28
    DELAY 64

    # Row reduction across the ADD result.
    VREDSUM.ROW.BF16 34, 30
    DELAY 37

    # FP8 pack / unpack with non-trivial scales hits the saturation,
    # underflow, and clip paths in the dual-path FP8 datapath.
    SELI 3, 0x7B
    SELI 4, 0x86
    VFP8PACK 36, 30, 3
    DELAY 64
    VFP8UNPACK 38, 36, 4
    DELAY 65

    # Column reduction over the unpacked stream drives the csum adder
    # tree with values shaped by every prior LUT and FP8 stage.
    VREDSUM.BF16 40, 38
    DELAY 128

    # ---- chain stores ----
    # m28/m29 (mul) -> 0x12000
    LI    x8, 1024
    VSTORE 28, x8, 0
    DELAY 33
    VSTORE 29, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90012000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT 1

    # m30/m31 (add) -> 0x12800
    LI    x8, 1024
    VSTORE 30, x8, 0
    DELAY 33
    VSTORE 31, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90012800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT 1

    # m32/m33 (sub) -> 0x13000
    LI    x8, 1024
    VSTORE 32, x8, 0
    DELAY 33
    VSTORE 33, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90013000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT 1

    # m34/m35 (row-sum) -> 0x13800
    LI    x8, 1024
    VSTORE 34, x8, 0
    DELAY 33
    VSTORE 35, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90013800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT 1

    # m36 (fp8pack, single 1 KiB mreg) -> 0x14000
    LI    x8, 1024
    VSTORE 36, x8, 0
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90014000
    LI    x11, 1024
    DMA.STORE x10, x9, x11, 1
    DMA.WAIT 1

    # m38/m39 (fp8unpack pair) -> 0x14400
    LI    x8, 1024
    VSTORE 38, x8, 0
    DELAY 33
    VSTORE 39, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90014400
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT 1

    # m40/m41 (col-sum pair) -> 0x14C00
    LI    x8, 1024
    VSTORE 40, x8, 0
    DELAY 33
    VSTORE 41, x8, 8
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90014C00
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT 1

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

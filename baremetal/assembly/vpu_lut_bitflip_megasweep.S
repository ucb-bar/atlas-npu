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
#   VREDSUM.ROW.BF16, VPACK.BF16.FP8 + VUNPACK.FP8.BF16, and VREDSUM.BF16 (column
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
#     0x14000  m36      VPACK.BF16.FP8(ADD, scale=0x83)    1 KiB
#     0x14400  m38/m39  VUNPACK.FP8.BF16(PACK, scale=0x86) 2 KiB
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
#   e3 = 0x83  (pack   scale e8m0)
#   e4 = 0x86  (unpack scale e8m0)

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 2048               # 2 KiB DMA size (input tiles, paired outputs)
    LI    x3, 2048               # 2 KiB DMA size for paired output stores

# ======================================================================
# Tile A (walk_pos) at DRAM 0x0000
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ---- Tile A / VSQRT       -> 0x2000 ----
    vsqrt.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90002000
    JAL   x1, store_pair
    # ---- Tile A / VLOG2       -> 0x2800 ----
    vlog2.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90002800
    JAL   x1, store_pair
    # ---- Tile A / VRECIP.BF16 -> 0x3000 ----
    vrecip.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90003000
    JAL   x1, store_pair
    # ---- Tile A / VEXP        -> 0x3800 ----
    vexp.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90003800
    JAL   x1, store_pair
    # ---- Tile A / VEXP2       -> 0x4000 ----
    vexp2.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90004000
    JAL   x1, store_pair
    # ---- Tile A / VTANH       -> 0x4800 ----
    vtanh.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90004800
    JAL   x1, store_pair
    # ---- Tile A / VSQUARE.BF16 -> 0x5000 ----
    vsquare.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90005000
    JAL   x1, store_pair
    # ---- Tile A / VCUBE.BF16   -> 0x5800 ----
    vcube.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90005800
    JAL   x1, store_pair

# ======================================================================
# Tile B (walk_signed) at DRAM 0x0800
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90000800
    dma.load x6, x10, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ---- Tile B / VSQRT       -> 0x6000 ----
    vsqrt.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90006000
    JAL   x1, store_pair
    # ---- Tile B / VLOG2       -> 0x6800 ----
    vlog2.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90006800
    JAL   x1, store_pair
    # ---- Tile B / VRECIP.BF16 -> 0x7000 ----
    vrecip.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90007000
    JAL   x1, store_pair
    # ---- Tile B / VEXP        -> 0x7800 ----
    vexp.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90007800
    JAL   x1, store_pair
    # ---- Tile B / VEXP2       -> 0x8000 ----
    vexp2.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90008000
    JAL   x1, store_pair
    # ---- Tile B / VTANH       -> 0x8800 ----
    vtanh.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90008800
    JAL   x1, store_pair
    # ---- Tile B / VSQUARE.BF16 -> 0x9000 ----
    vsquare.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90009000
    JAL   x1, store_pair
    # ---- Tile B / VCUBE.BF16   -> 0x9800 ----
    vcube.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90009800
    JAL   x1, store_pair

# ======================================================================
# Tile C (class_lattice) at DRAM 0x1000
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90001000
    dma.load x6, x10, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ---- Tile C / VSQRT       -> 0xA000 ----
    vsqrt.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000A000
    JAL   x1, store_pair
    # ---- Tile C / VLOG2       -> 0xA800 ----
    vlog2.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000A800
    JAL   x1, store_pair
    # ---- Tile C / VRECIP.BF16 -> 0xB000 ----
    vrecip.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000B000
    JAL   x1, store_pair
    # ---- Tile C / VEXP        -> 0xB800 ----
    vexp.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000B800
    JAL   x1, store_pair
    # ---- Tile C / VEXP2       -> 0xC000 ----
    vexp2.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000C000
    JAL   x1, store_pair
    # ---- Tile C / VTANH       -> 0xC800 ----
    vtanh.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000C800
    JAL   x1, store_pair
    # ---- Tile C / VSQUARE.BF16 -> 0xD000 ----
    vsquare.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000D000
    JAL   x1, store_pair
    # ---- Tile C / VCUBE.BF16   -> 0xD800 ----
    vcube.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000D800
    JAL   x1, store_pair

# ======================================================================
# Tile D (magnitude_alt) at DRAM 0x1800
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90001800
    dma.load x6, x10, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ---- Tile D / VSQRT       -> 0xE000 ----
    vsqrt.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000E000
    JAL   x1, store_pair
    # ---- Tile D / VLOG2       -> 0xE800 ----
    vlog2.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000E800
    JAL   x1, store_pair
    # ---- Tile D / VRECIP.BF16 -> 0xF000 ----
    vrecip.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000F000
    JAL   x1, store_pair
    # ---- Tile D / VEXP        -> 0xF800 ----
    vexp.bf16 m4, m0
    DELAY 64
    LI    x10, 0x9000F800
    JAL   x1, store_pair
    # ---- Tile D / VEXP2       -> 0x10000 ----
    vexp2.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90010000
    JAL   x1, store_pair
    # ---- Tile D / VTANH       -> 0x10800 ----
    vtanh.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90010800
    JAL   x1, store_pair
    # ---- Tile D / VSQUARE.BF16 -> 0x11000 ----
    vsquare.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90011000
    JAL   x1, store_pair
    # ---- Tile D / VCUBE.BF16   -> 0x11800 ----
    vcube.bf16 m4, m0
    DELAY 64
    LI    x10, 0x90011800
    JAL   x1, store_pair

# ======================================================================
# Chain phase
#   Reload Tile A into m0/m1, then drive five LUT pipes into binary,
#   reduction, FP8 pack/unpack, and column reduce so the AddSubSum,
#   ColAdd, and FP8 datapaths see LUT-derived operands.
# ======================================================================
    ADDI  x6, x0, 0
    LI    x10, 0x90000000
    dma.load x6, x10, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # Five LUT pipes off the same Tile A pair. Each is sequenced into
    # its own destination bank pair, separated by a full latency window
    # so the next op can read a fresh m0/m1 / write its own dest.
    vlog2.bf16 m8, m0
    DELAY 64
    vsqrt.bf16 m12, m0
    DELAY 64
    vexp2.bf16 m16, m0
    DELAY 64
    vrecip.bf16 m20, m0
    DELAY 64
    vtanh.bf16 m24, m0
    DELAY 64

    # Binary combinations feed AddSubSum / Mul-Rec adders with non-trivial
    # mantissa/exponent diversity (unlike constant-input arithmetic tests).
    vmul.bf16 m28, m8, m12
    DELAY 64
    vadd.bf16 m30, m16, m20
    DELAY 64
    vsub.bf16 m32, m24, m28
    DELAY 64

    # Row reduction across the ADD result.
    vredsum.row.bf16 m34, m30
    DELAY 37

    # FP8 pack / unpack with non-trivial scales hits the saturation,
    # underflow, and clip paths in the dual-path FP8 datapath.
    seli e3, 0x83
    seli e4, 0x86
    vpack.bf16.fp8 m36, m30, e3
    DELAY 64
    vunpack.fp8.bf16 m38, m36, e4
    DELAY 65

    # Column reduction over the unpacked stream drives the csum adder
    # tree with values shaped by every prior LUT and FP8 stage.
    vredsum.bf16 m40, m38
    DELAY 128

    # ---- chain stores ----
    # m28/m29 (mul) -> 0x12000
    LI    x8, 1024
    vstore m28, 0(x8)
    DELAY 33
    vstore m29, 8(x8)
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90012000
    dma.store x10, x9, x3, ch1
    dma.wait ch1

    # m30/m31 (add) -> 0x12800
    LI    x8, 1024
    vstore m30, 0(x8)
    DELAY 33
    vstore m31, 8(x8)
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90012800
    dma.store x10, x9, x3, ch1
    dma.wait ch1

    # m32/m33 (sub) -> 0x13000
    LI    x8, 1024
    vstore m32, 0(x8)
    DELAY 33
    vstore m33, 8(x8)
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90013000
    dma.store x10, x9, x3, ch1
    dma.wait ch1

    # m34/m35 (row-sum) -> 0x13800
    LI    x8, 1024
    vstore m34, 0(x8)
    DELAY 33
    vstore m35, 8(x8)
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90013800
    dma.store x10, x9, x3, ch1
    dma.wait ch1

    # m36 (fp8pack, single 1 KiB mreg) -> 0x14000
    LI    x8, 1024
    vstore m36, 0(x8)
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90014000
    LI    x11, 1024
    dma.store x10, x9, x11, ch1
    dma.wait ch1

    # m38/m39 (fp8unpack pair) -> 0x14400
    LI    x8, 1024
    vstore m38, 0(x8)
    DELAY 33
    vstore m39, 8(x8)
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90014400
    dma.store x10, x9, x3, ch1
    dma.wait ch1

    # m40/m41 (col-sum pair) -> 0x14C00
    LI    x8, 1024
    vstore m40, 0(x8)
    DELAY 33
    vstore m41, 8(x8)
    DELAY 33
    LI    x9, 1024
    LI    x10, 0x90014C00
    dma.store x10, x9, x3, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

store_pair:
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33
    LI    x9, 1024
    dma.store x10, x9, x3, ch1
    dma.wait ch1
    JALR  x0, x1, 0

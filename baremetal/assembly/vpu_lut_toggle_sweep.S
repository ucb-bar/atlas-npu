# vpu_lut_toggle_sweep.S — mantissa/exponent-diverse VPU unary math sweep
# @TIMEOUT 200000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_lut_toggle_sweep.py
#
# Coverage intent (directly targets atlasTile_coverage_report.md findings):
#   - sqrt.lut     toggle 5.21%
#   - log2.lut     toggle 3.37%
#   - exp.lut     toggle 11.66%
#   - tanh.lut    toggle 11.66%
#   - rcp.lut      toggle 3.43%
#
# Strategy:
#   Baseline unary-math tests feed CONSTANT BF16 tiles (all 1.0, all -2.0,
#   etc.), so only 1-2 LUT entries out of 128 ever get read per op. This
#   test feeds three 32x32 BF16 tiles that are deliberately designed so
#   that the mantissa and exponent bits of every lane take MANY distinct
#   values over a single tile, and each tile covers a different exponent
#   range. Running all eight unary math ops on all three tiles reads a
#   very wide subset of each LUT and toggles many more output bits.
#
#   Tile 0 (DRAM 0x0000, 2 KiB): mantissa-sweep, exponent clustered around 0
#          -> dense log2/rcp/sqrt LUT coverage.
#   Tile 1 (DRAM 0x0800, 2 KiB): moderate magnitude, mixed sign
#          -> dense sin/cos/tanh/exp LUT coverage.
#   Tile 2 (DRAM 0x1000, 2 KiB): wide exponent sweep (small and large)
#          -> exponent-diversity for sqrt/log2/exp/exp2 LUT.
#
# For each of the 3 tiles we run 8 VPU unary ops:
#   VEXP, VEXP2, VSIN, VCOS, VTANH, VLOG2, VSQRT, VRECIP.BF16
# giving 24 output tiles of 2 KiB each. Output layout:
#   DRAM output base = 0x2000
#   Per-pass stride  = 0x0800
#   pass_idx = tile_idx * 8 + op_idx
#   Pass DRAM offset = 0x2000 + pass_idx * 0x0800
#
# VMEM layout (word addresses):
#   0      input tile staging  (512 words = 2 KiB)  -> feeds m0,m1
#   1024   output pair staging (512 words = 2 KiB)  -> drained from m4,m5
#
# MREG layout:
#   m0,m1  source BF16 pair
#   m4,m5  VPU result pair

    ADDI  x28, x0, 1
    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 2048           # 2 KiB per tile / output
    LI    x3, 2048           # same, used for DMA.STOREs

# ======================================================================
# TILE 0 — mantissa sweep, clustered exponent
# ======================================================================
    ADDI  x6, x0, 0          # VMEM word addr 0
    LI    x1, 0x0000         # DRAM byte 0x0000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ---------- Tile0 / VEXP ----------
    VEXP  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x2000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile0 / VEXP2 ----------
    VEXP2 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x2800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile0 / VSIN ----------
    VSIN  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x3000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile0 / VCOS ----------
    VCOS  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x3800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile0 / VTANH ----------
    VTANH 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x4000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile0 / VLOG2 ----------
    VLOG2 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x4800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile0 / VSQRT ----------
    VSQRT 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x5000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile0 / VRECIP.BF16 ----------
    VRECIP.BF16 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x5800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

# ======================================================================
# TILE 1 — mid-range values, mixed sign
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x0800
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ---------- Tile1 / VEXP ----------
    VEXP  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x6000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile1 / VEXP2 ----------
    VEXP2 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x6800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile1 / VSIN ----------
    VSIN  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x7000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile1 / VCOS ----------
    VCOS  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x7800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile1 / VTANH ----------
    VTANH 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x8000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile1 / VLOG2 ----------
    VLOG2 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x8800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile1 / VSQRT ----------
    VSQRT 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x9000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile1 / VRECIP.BF16 ----------
    VRECIP.BF16 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0x9800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

# ======================================================================
# TILE 2 — wide exponent sweep
# ======================================================================
    ADDI  x6, x0, 0
    LI    x1, 0x1000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ---------- Tile2 / VEXP ----------
    VEXP  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0xA000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile2 / VEXP2 ----------
    VEXP2 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0xA800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile2 / VSIN ----------
    VSIN  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0xB000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile2 / VCOS ----------
    VCOS  4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0xB800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile2 / VTANH ----------
    VTANH 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0xC000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile2 / VLOG2 ----------
    VLOG2 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0xC800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile2 / VSQRT ----------
    VSQRT 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0xD000
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

    # ---------- Tile2 / VRECIP.BF16 ----------
    VRECIP.BF16 4, 0
    DELAY 160
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x9, 1024
    LI    x10, 0xD800
    DMA.STORE x10, x9, x3, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

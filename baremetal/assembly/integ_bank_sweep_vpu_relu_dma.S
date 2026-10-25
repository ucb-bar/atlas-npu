# integ_bank_sweep_vpu_relu_dma.S — VMEM bank/address sweep with DMA, VPU, and scalar LSU
# @TIMEOUT 800000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_integ_bank_sweep_vpu_relu_dma.py
#
# Coverage intent:
#   - DMA traffic to four distinct VMEM placements spread across the lower
#     VMEM banks of the 1.5 MiB default map
#   - VLOAD/VSTORE on four distinct tensor slots; under current 256 KiB
#     banking these land in banks 0, 0, 1, and 2
#   - VPU pair op (VRELU) on each loaded BF16 tile
#   - Scalar LB/LHU/LW reads from a dedicated scratch region after each phase
#
# DRAM byte layout (generator-backed):
#   input0   0x0000 .. 0x07FF   BF16 tile 0 (2048 B)
#   input1   0x0800 .. 0x0FFF   BF16 tile 1 (2048 B)
#   input2   0x1000 .. 0x17FF   BF16 tile 2 (2048 B)
#   input3   0x1800 .. 0x1FFF   BF16 tile 3 (2048 B)
#   out0     0x2000 .. 0x27FF   ReLU(tile 0)
#   out1     0x2800 .. 0x2FFF   ReLU(tile 1)
#   out2     0x3000 .. 0x37FF   ReLU(tile 2)
#   out3     0x3800 .. 0x3FFF   ReLU(tile 3)
#
# VMEM plan:
#   scratch bytes      0x7000 .. 0x701F   scalar-only scratch, disjoint from DMA.STORE sources
#
#   tile0 in bytes     0x00000 .. 0x007FF   word base 0x0000 /4 =     0, bank 0, VLOAD imm 0/8
#   tile0 out bytes    0x00800 .. 0x00FFF   word base 0x0800 /4 =   512, bank 0, VSTORE imm 0/8
#
#   tile1 in bytes     0x20000 .. 0x207FF   word base 0x20000/4 = 32768, bank 0, VLOAD imm 0/8
#   tile1 out bytes    0x20800 .. 0x20FFF   word base 0x20800/4 = 33280, bank 0, VSTORE imm 0/8
#
#   tile2 in bytes     0x40000 .. 0x407FF   word base 0x40000/4 = 65536, bank 1, VLOAD imm 0/8
#   tile2 out bytes    0x40800 .. 0x40FFF   word base 0x40800/4 = 66048, bank 1, VSTORE imm 0/8
#
#   tile3 in bytes     0x80000 .. 0x807FF   word base 0x80000/4 = 131072, bank 2, VLOAD imm 0/8
#   tile3 out bytes    0x80800 .. 0x80FFF   word base 0x80800/4 = 131584, bank 2, VSTORE imm 0/8
#
# Notes:
#   - Every DMA is serialized with DMA.WAIT.
#   - Every VLOAD/VSTORE pair is separated conservatively.
#   - Scalar stores touch only scratch, never any later DMA.STORE source footprint.

    ADDI  x28, x0, 1

    LI    x5, 0x00000000
    dma.config x5, ch0

    LI    x29, 0x7000        # scratch byte base
    ADDI  x30, x0, 0x5A
    sb x30, 0(x29)
    ADDI  x30, x0, 0x34
    SLLI  x30, x30, 8
    ADDI  x30, x30, 0x12
    sh x30, 2(x29)
    LI    x30, 0x13579BDF
    sw x30, 4(x29)

    LI    x12, 2048          # tile bytes

    # ------------------------------------------------------------------
    # Tile 0: low VMEM region
    # ------------------------------------------------------------------
    LI    x6, 0              # input VMEM word base
    LI    x10, 0x90000000  # input DRAM byte offset
    dma.load x6, x10, x12, ch0
    dma.wait ch0

    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    vrelu.bf16 m2, m0
    DELAY 64

    LI    x7, 512            # output VMEM word base
    vstore m2, 0(x7)
    DELAY 33
    vstore m3, 8(x7)
    DELAY 33

    lb x20, 0(x29)
    DELAY 2
    ADDI  x21, x0, 0x5A
    BNE   x20, x21, fail

    lhu x20, 2(x29)
    DELAY 2
    LI    x21, 0x3412
    BNE   x20, x21, fail

    lw x20, 4(x29)
    DELAY 2
    LI    x21, 0x13579BDF
    BNE   x20, x21, fail

    LI    x11, 0x90002000  # output DRAM byte offset
    dma.store x11, x7, x12, ch1
    dma.wait ch1

    # ------------------------------------------------------------------
    # Tile 1: 0x20000 VMEM region
    # ------------------------------------------------------------------
    LI    x6, 32768          # 0x20000 bytes / 4
    LI    x10, 0x90000800
    dma.load x6, x10, x12, ch2
    dma.wait ch2

    vload m4, 0(x6)
    DELAY 33
    vload m5, 8(x6)
    DELAY 33

    vrelu.bf16 m6, m4
    DELAY 64

    LI    x7, 33280          # 0x20800 bytes / 4
    vstore m6, 0(x7)
    DELAY 33
    vstore m7, 8(x7)
    DELAY 33

    lb x20, 0(x29)
    DELAY 2
    ADDI  x21, x0, 0x5A
    BNE   x20, x21, fail

    lhu x20, 2(x29)
    DELAY 2
    LI    x21, 0x3412
    BNE   x20, x21, fail

    lw x20, 4(x29)
    DELAY 2
    LI    x21, 0x13579BDF
    BNE   x20, x21, fail

    LI    x11, 0x90002800
    dma.store x11, x7, x12, ch3
    dma.wait ch3

    # ------------------------------------------------------------------
    # Tile 2: 0x40000 VMEM region
    # ------------------------------------------------------------------
    LI    x6, 65536          # 0x40000 bytes / 4
    LI    x10, 0x90001000
    dma.load x6, x10, x12, ch4
    dma.wait ch4

    vload m8, 0(x6)
    DELAY 33
    vload m9, 8(x6)
    DELAY 33

    vrelu.bf16 m10, m8
    DELAY 64

    LI    x7, 66048          # 0x40800 bytes / 4
    vstore m10, 0(x7)
    DELAY 33
    vstore m11, 8(x7)
    DELAY 33

    lb x20, 0(x29)
    DELAY 2
    ADDI  x21, x0, 0x5A
    BNE   x20, x21, fail

    lhu x20, 2(x29)
    DELAY 2
    LI    x21, 0x3412
    BNE   x20, x21, fail

    lw x20, 4(x29)
    DELAY 2
    LI    x21, 0x13579BDF
    BNE   x20, x21, fail

    LI    x11, 0x90003000
    dma.store x11, x7, x12, ch5
    dma.wait ch5

    # ------------------------------------------------------------------
    # Tile 3: 0x80000 VMEM region
    # ------------------------------------------------------------------
    LI    x6, 131072         # 0x80000 bytes / 4
    LI    x10, 0x90001800
    dma.load x6, x10, x12, ch6
    dma.wait ch6

    vload m12, 0(x6)
    DELAY 33
    vload m13, 8(x6)
    DELAY 33

    vrelu.bf16 m14, m12
    DELAY 64

    LI    x7, 131584         # 0x80800 bytes / 4
    vstore m14, 0(x7)
    DELAY 33
    vstore m15, 8(x7)
    DELAY 33

    lb x20, 0(x29)
    DELAY 2
    ADDI  x21, x0, 0x5A
    BNE   x20, x21, fail

    lhu x20, 2(x29)
    DELAY 2
    LI    x21, 0x3412
    BNE   x20, x21, fail

    lw x20, 4(x29)
    DELAY 2
    LI    x21, 0x13579BDF
    BNE   x20, x21, fail

    LI    x11, 0x90003800
    dma.store x11, x7, x12, ch7
    dma.wait ch7

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

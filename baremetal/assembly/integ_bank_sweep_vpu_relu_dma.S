# integ_bank_sweep_vpu_relu_dma.S — VMEM bank/address sweep with DMA, VPU, and scalar LSU
# @TIMEOUT 800000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_integ_bank_sweep_vpu_relu_dma.py
#
# Coverage intent:
#   - DMA traffic to four distinct VMEM placements spread across the 1 MiB VMEM
#   - VLOAD/VSTORE on four distinct tensor slots / bank decodes
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
#   tile0 in bytes     0x00000 .. 0x007FF   word base 0x0000 /4 =     0, VLOAD imm 0/8
#   tile0 out bytes    0x00800 .. 0x00FFF   word base 0x0800 /4 =   512, VSTORE imm 0/8
#
#   tile1 in bytes     0x20000 .. 0x207FF   word base 0x20000/4 = 32768, VLOAD imm 0/8
#   tile1 out bytes    0x20800 .. 0x20FFF   word base 0x20800/4 = 33280, VSTORE imm 0/8
#
#   tile2 in bytes     0x40000 .. 0x407FF   word base 0x40000/4 = 65536, VLOAD imm 0/8
#   tile2 out bytes    0x40800 .. 0x40FFF   word base 0x40800/4 = 66048, VSTORE imm 0/8
#
#   tile3 in bytes     0x60000 .. 0x607FF   word base 0x60000/4 = 98304, VLOAD imm 0/8
#   tile3 out bytes    0x60800 .. 0x60FFF   word base 0x60800/4 = 98816, VSTORE imm 0/8
#
# Notes:
#   - Every DMA is serialized with DMA.WAIT.
#   - Every VLOAD/VSTORE pair is separated conservatively.
#   - Scalar stores touch only scratch, never any later DMA.STORE source footprint.

    ADDI  x28, x0, 1

    LI    x5, 0x00000001
    DMA.CONFIG x5, 0

    LI    x29, 0x7000        # scratch byte base
    ADDI  x30, x0, 0x5A
    SB    x30, x29, 0
    ADDI  x30, x0, 0x34
    SLLI  x30, x30, 8
    ADDI  x30, x30, 0x12
    SH    x30, x29, 2
    LI    x30, 0x13579BDF
    SW    x30, x29, 4

    LI    x12, 2048          # tile bytes

    # ------------------------------------------------------------------
    # Tile 0: low VMEM region
    # ------------------------------------------------------------------
    LI    x6, 0              # input VMEM word base
    LI    x10, 0x0000        # input DRAM byte offset
    DMA.LOAD x6, x10, x12, 0
    DMA.WAIT 0

    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    VRELU 2, 0
    DELAY 64

    LI    x7, 512            # output VMEM word base
    VSTORE 2, x7, 0
    DELAY 33
    VSTORE 3, x7, 8
    DELAY 33

    LB    x20, x29, 0
    DELAY 2
    ADDI  x21, x0, 0x5A
    BNE   x20, x21, fail
    NOP

    LHU   x20, x29, 2
    DELAY 2
    LI    x21, 0x3412
    BNE   x20, x21, fail
    NOP

    LW    x20, x29, 4
    DELAY 2
    LI    x21, 0x13579BDF
    BNE   x20, x21, fail
    NOP

    LI    x11, 0x2000        # output DRAM byte offset
    DMA.STORE x11, x7, x12, 1
    DMA.WAIT 1

    # ------------------------------------------------------------------
    # Tile 1: 0x20000 VMEM region
    # ------------------------------------------------------------------
    LI    x6, 32768          # 0x20000 bytes / 4
    LI    x10, 0x0800
    DMA.LOAD x6, x10, x12, 2
    DMA.WAIT 2

    VLOAD 4, x6, 0
    DELAY 33
    VLOAD 5, x6, 8
    DELAY 33

    VRELU 6, 4
    DELAY 64

    LI    x7, 33280          # 0x20800 bytes / 4
    VSTORE 6, x7, 0
    DELAY 33
    VSTORE 7, x7, 8
    DELAY 33

    LB    x20, x29, 0
    DELAY 2
    ADDI  x21, x0, 0x5A
    BNE   x20, x21, fail
    NOP

    LHU   x20, x29, 2
    DELAY 2
    LI    x21, 0x3412
    BNE   x20, x21, fail
    NOP

    LW    x20, x29, 4
    DELAY 2
    LI    x21, 0x13579BDF
    BNE   x20, x21, fail
    NOP

    LI    x11, 0x2800
    DMA.STORE x11, x7, x12, 3
    DMA.WAIT 3

    # ------------------------------------------------------------------
    # Tile 2: 0x40000 VMEM region
    # ------------------------------------------------------------------
    LI    x6, 65536          # 0x40000 bytes / 4
    LI    x10, 0x1000
    DMA.LOAD x6, x10, x12, 4
    DMA.WAIT 4

    VLOAD 8, x6, 0
    DELAY 33
    VLOAD 9, x6, 8
    DELAY 33

    VRELU 10, 8
    DELAY 64

    LI    x7, 66048          # 0x40800 bytes / 4
    VSTORE 10, x7, 0
    DELAY 33
    VSTORE 11, x7, 8
    DELAY 33

    LB    x20, x29, 0
    DELAY 2
    ADDI  x21, x0, 0x5A
    BNE   x20, x21, fail
    NOP

    LHU   x20, x29, 2
    DELAY 2
    LI    x21, 0x3412
    BNE   x20, x21, fail
    NOP

    LW    x20, x29, 4
    DELAY 2
    LI    x21, 0x13579BDF
    BNE   x20, x21, fail
    NOP

    LI    x11, 0x3000
    DMA.STORE x11, x7, x12, 5
    DMA.WAIT 5

    # ------------------------------------------------------------------
    # Tile 3: 0x60000 VMEM region
    # ------------------------------------------------------------------
    LI    x6, 98304          # 0x60000 bytes / 4
    LI    x10, 0x1800
    DMA.LOAD x6, x10, x12, 6
    DMA.WAIT 6

    VLOAD 12, x6, 0
    DELAY 33
    VLOAD 13, x6, 8
    DELAY 33

    VRELU 14, 12
    DELAY 64

    LI    x7, 98816          # 0x60800 bytes / 4
    VSTORE 14, x7, 0
    DELAY 33
    VSTORE 15, x7, 8
    DELAY 33

    LB    x20, x29, 0
    DELAY 2
    ADDI  x21, x0, 0x5A
    BNE   x20, x21, fail
    NOP

    LHU   x20, x29, 2
    DELAY 2
    LI    x21, 0x3412
    BNE   x20, x21, fail
    NOP

    LW    x20, x29, 4
    DELAY 2
    LI    x21, 0x13579BDF
    BNE   x20, x21, fail
    NOP

    LI    x11, 0x3800
    DMA.STORE x11, x7, x12, 7
    DMA.WAIT 7

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

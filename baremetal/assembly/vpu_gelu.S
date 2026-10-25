# vpu_gelu.S — vgelu.bf16 against the eight .bf16 instructions that define it
# @TIMEOUT 400000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_gelu.py
#
# vgelu.bf16 = vcube; vmul c1; vadd x; vmul c2; vtanh; vadd 1; vmul x; vmul 0.5, each step rounded
# as its instruction is. For each of the 64 tiles that hold the 65,536 BF16 codes, vgelu.bf16
# (in place) and the eight instructions (constants from x-registers, .vx) must give the same
# tile, the generator's. vgelu overlaps the sequence's single-input steps in the other VPU slot.
#
# DRAM (from the generator):
#   0x00000 + 0x800 k   tile k: codes 1024 k .. 1024 k + 1023
#   0x20000 + 0x800 k   vgelu.bf16 of tile k
#   0x40000 + 0x800 k   the eight instructions on tile k

    LI    x5, 0
    dma.config x5, ch0
    LUI   x11, 0x3d370               # c1 = 0.044715 in bits 31:16
    LUI   x12, 0x3f4c0               # c2 = sqrt(2/pi)
    LUI   x13, 0x3f800               # 1.0
    LUI   x14, 0x3f000               # 0.5
    LI    x2, 2048
    LI    x15, 512                   # VMEM byte 2048: vgelu result
    LI    x16, 1024                  # VMEM byte 4096: sequence result
    LI    x20, 0x90000000
    LI    x21, 0x90020000
    LI    x22, 0x90040000
    LI    x23, 64
tile:
    dma.load x0, x20, x2, ch0        # tile k -> VMEM byte 0
    dma.wait ch0
    vload m0, 0(x0)
    vload m1, 8(x0)
    vload m2, 0(x0)
    vload m3, 8(x0)
    vgelu.bf16 m2, m2
    vcube.bf16 m8, m0
    vmul.bf16.vx m8, m8, x11
    vadd.bf16 m8, m8, m0
    vmul.bf16.vx m8, m8, x12
    vtanh.bf16 m8, m8
    vadd.bf16.vx m8, m8, x13
    vmul.bf16 m8, m0, m8
    vmul.bf16.vx m8, m8, x14
    vstore m2, 0(x15)
    vstore m3, 8(x15)
    vstore m8, 0(x16)
    vstore m9, 8(x16)
    dma.store x21, x15, x2, ch1
    dma.store x22, x16, x2, ch2
    dma.wait ch1
    dma.wait ch2
    ADD   x20, x20, x2
    ADD   x21, x21, x2
    ADD   x22, x22, x2
    ADDI  x23, x23, -1
    BNE   x23, x0, tile

    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

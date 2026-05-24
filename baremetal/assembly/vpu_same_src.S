# vpu_same_src.S — VPU same-source binary-op regression
# Tests: vadd.bf16, vsub.bf16, vmul.bf16, vmin.bf16, vmax.bf16
#        with the same source register pair used for both operands
# @TIMEOUT 50000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_same_src.py
#
# DRAM layout (set by Python generator):
#   [0x0000..0x03FF]  Input A bank 0    (1024 B)
#   [0x0400..0x07FF]  Input A bank 1    (1024 B)
#   [0x0800..0x0FFF]  vadd output
#   [0x1000..0x17FF]  vsub output
#   [0x1800..0x1FFF]  vmul output
#   [0x2000..0x27FF]  vmin output
#   [0x2800..0x2FFF]  vmax output
#
# Mreg allocation: m0,m1 = A    m2,m3 = result

    ADDI  x28, x0, 1

    # ── DMA config ──
    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input A → VMEM[0..511] ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 0x90000400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    # VLOAD A → m0,m1
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ================================================================
    # Test 1: vadd.bf16  m2, m0, m0
    # ================================================================
    VADD.BF16  2, 0, 0
    DELAY 64

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 33
    VSTORE 3, x8, 8
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90000800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x90000C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 2: vsub.bf16  m2, m0, m0
    # ================================================================
    VSUB.BF16  2, 0, 0
    DELAY 64

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 33
    VSTORE 3, x8, 8
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90001000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x90001400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 3: vmul.bf16  m2, m0, m0
    # ================================================================
    VMUL.BF16  2, 0, 0
    DELAY 64

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 33
    VSTORE 3, x8, 8
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90001800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x90001C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 4: vmin.bf16  m2, m0, m0
    # ================================================================
    VMIN.BF16  2, 0, 0
    DELAY 64

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 33
    VSTORE 3, x8, 8
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90002000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x90002400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 5: vmax.bf16  m2, m0, m0
    # ================================================================
    VMAX.BF16  2, 0, 0
    DELAY 64

    LI    x8, 512
    VSTORE 2, x8, 0
    DELAY 33
    VSTORE 3, x8, 8
    DELAY 33

    LI    x8, 512
    LI    x3, 0x90002800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 768
    LI    x3, 0x90002C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

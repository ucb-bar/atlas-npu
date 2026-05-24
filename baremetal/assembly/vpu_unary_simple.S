# vpu_unary_simple.S — VPU simple unary operations integration test
# Tests: vmov, vrelu.bf16, vrecip.bf16, vsquare.bf16, vcube.bf16
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_unary_simple.py
#
# DRAM layout:
#   [0x0000..0x07FF]  Input A            (2048 B, 2 banks)
#   [0x0800..0x0FFF]  vmov output        (2048 B)
#   [0x1000..0x17FF]  vrelu output       (2048 B)
#   [0x1800..0x1FFF]  vrecip output      (2048 B)
#   [0x2000..0x27FF]  vsquare output     (2048 B)
#   [0x2800..0x2FFF]  vcube output       (2048 B)
#
# Mreg: m0,m1 = A    m4,m5 = result

    ADDI  x28, x0, 1
    LI    x5, 0x00000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input A → m0,m1 ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 0x90000400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # ================================================================
    # Test 1: vmov  m4, m0
    # ================================================================
    VMOV 4, 0
    DELAY 64

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90000800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x90000C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 2: vrelu  m4, m0
    # ================================================================
    VRELU 4, 0
    DELAY 64

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90001000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x90001400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 3: vrecip.bf16  m4, m0
    # ================================================================
    VRECIP.BF16 4, 0
    DELAY 64

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90001800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x90001C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 4: vsquare.bf16  m4, m0
    # ================================================================
    VSQUARE.BF16 4, 0
    DELAY 64

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x90002400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 5: vcube.bf16  m4, m0
    # ================================================================
    VCUBE.BF16 4, 0
    DELAY 64

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 33
    VSTORE 5, x8, 8
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
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

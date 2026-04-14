# vpu_unary_math.S — VPU transcendental / math unary ops
# Tests: vexp, vexp2, vsin, vcos, vtanh, vlog2, vsqrt
# @TIMEOUT 60000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_unary_math.py
#
# DRAM layout:
#   [0x0000..0x07FF]  Input A            (2048 B)
#   [0x0800..0x0FFF]  vexp output        (2048 B)
#   [0x1000..0x17FF]  vexp2 output       (2048 B)
#   [0x1800..0x1FFF]  vsin output        (2048 B)
#   [0x2000..0x27FF]  vcos output        (2048 B)
#   [0x2800..0x2FFF]  vtanh output       (2048 B)
#   [0x3000..0x37FF]  vlog2 output       (2048 B)
#   [0x3800..0x3FFF]  vsqrt output       (2048 B)

    ADDI  x28, x0, 1
    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input A → m0,m1 ──
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ── Macro-like store pattern: result in m4,m5 → VMEM[1024] → DRAM[x3] ──
    # Each block: VPU op, delay, 2×VSTORE, 2×DMA.STORE

    # ================================================================
    # Test 1: vexp
    # ================================================================
    VEXP 4, 0
    DELAY 150
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x0C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 2: vexp2
    # ================================================================
    VEXP2 4, 0
    DELAY 150
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x1000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x1400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 3: vsin
    # ================================================================
    VSIN 4, 0
    DELAY 150
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x1800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x1C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 4: vcos
    # ================================================================
    VCOS 4, 0
    DELAY 150
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x2000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x2400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 5: vtanh
    # ================================================================
    VTANH 4, 0
    DELAY 150
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x2800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x2C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 6: vlog2
    # ================================================================
    VLOG2 4, 0
    DELAY 150
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x3000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x3400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # ================================================================
    # Test 7: vsqrt
    # ================================================================
    VSQRT 4, 0
    DELAY 150
    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x3800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x3C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

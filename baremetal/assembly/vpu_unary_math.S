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
    LI    x5, 0x00000000
    dma.config x5, ch0
    LI    x2, 1024

    # ── Load Input A → m0,m1 ──
    ADDI  x6, x0, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 256
    LI    x1, 0x90000400
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ── Macro-like store pattern: result in m4,m5 → VMEM[1024] → DRAM[x3] ──
    # Each block: VPU op, delay, 2×VSTORE, 2×DMA.STORE

    # ================================================================
    # Test 1: vexp
    # ================================================================
    vexp.bf16 m4, m0
    DELAY 64
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90000800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90000C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    # ================================================================
    # Test 2: vexp2
    # ================================================================
    vexp2.bf16 m4, m0
    DELAY 64
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90001000
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90001400
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    # ================================================================
    # Test 3: vsin
    # ================================================================
    vsin.bf16 m4, m0
    DELAY 64
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90001800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90001C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    # ================================================================
    # Test 4: vcos
    # ================================================================
    vcos.bf16 m4, m0
    DELAY 64
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002000
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90002400
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    # ================================================================
    # Test 5: vtanh
    # ================================================================
    vtanh.bf16 m4, m0
    DELAY 64
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90002C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    # ================================================================
    # Test 6: vlog2
    # ================================================================
    vlog2.bf16 m4, m0
    DELAY 64
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003000
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90003400
    dma.store x3, x8, x2, ch1
    dma.wait ch1

    # ================================================================
    # Test 7: vsqrt
    # ================================================================
    vsqrt.bf16 m4, m0
    DELAY 64
    LI    x8, 1024
    vstore m4, 0(x8)
    DELAY 33
    vstore m5, 8(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003800
    dma.store x3, x8, x2, ch1
    dma.wait ch1
    LI    x8, 1280
    LI    x3, 0x90003C00
    dma.store x3, x8, x2, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

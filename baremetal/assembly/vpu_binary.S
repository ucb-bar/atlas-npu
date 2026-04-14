# vpu_binary.S — VPU binary operations integration test
# Tests: vadd.bf16, vsub.bf16, vmul.bf16, vmin.bf16, vmax.bf16
# @TIMEOUT 50000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_binary.py
#
# DRAM layout (set by Python generator):
#   [0x0000..0x03FF]  Input A bank 0    (1024 B)
#   [0x0400..0x07FF]  Input A bank 1    (1024 B)
#   [0x0800..0x0BFF]  Input B bank 0    (1024 B)
#   [0x0C00..0x0FFF]  Input B bank 1    (1024 B)
#   [0x1000..0x17FF]  vadd output       (2048 B)
#   [0x1800..0x1FFF]  vsub output       (2048 B)
#   [0x2000..0x27FF]  vmul output       (2048 B)
#   [0x2800..0x2FFF]  vmin output       (2048 B)
#   [0x3000..0x37FF]  vmax output       (2048 B)
#
# Mreg allocation:  m0,m1 = A    m2,m3 = B    m4,m5 = result
# VMEM:  [0..255] A0  [256..511] A1  [512..767] B0  [768..1023] B1
#        [1024..1279] out0  [1280..1535] out1

    ADDI  x28, x0, 1

    # ── DMA config ──
    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024              # bank size

    # ── Load Input A → VMEM[0..511] ──
    ADDI  x6, x0, 0             # VMEM[0]
    ADDI  x1, x0, 0             # DRAM[0]
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256               # VMEM[256]
    LI    x1, 1024              # DRAM[1024]
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0

    # VLOAD A → m0,m1
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ── Load Input B → VMEM[512..1023] ──
    LI    x7, 512               # VMEM[512]
    LI    x1, 2048              # DRAM[2048]
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0
    LI    x7, 768               # VMEM[768]
    LI    x1, 3072              # DRAM[3072]
    DMA.LOAD  x7, x1, x2, 0
    DMA.WAIT  0

    # VLOAD B → m2,m3
    LI    x7, 512
    VLOAD 2, x7, 0
    DELAY 40
    VLOAD 3, x7, 8
    DELAY 40

    # ================================================================
    # Test 1: vadd.bf16  m4, m0, m2
    # ================================================================
    VADD.BF16  4, 0, 2
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
    # Test 2: vsub.bf16  m4, m0, m2
    # ================================================================
    VSUB.BF16  4, 0, 2
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
    # Test 3: vmul.bf16  m4, m0, m2
    # ================================================================
    VMUL.BF16  4, 0, 2
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
    # Test 4: vmin.bf16  m4, m0, m2
    # ================================================================
    VMIN.BF16  4, 0, 2
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
    # Test 5: vmax.bf16  m4, m0, m2
    # ================================================================
    VMAX.BF16  4, 0, 2
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

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL

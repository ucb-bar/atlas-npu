# pw_vpu_overlap_pointwise_pack.S — pairwise VPU overlap with pointwise/fp8 first-op cases
# Exhaustive ordered overlap coverage for the current VPU single-input
# overlap window: each case issues op1 and op2 in consecutive cycles
# with no intervening DELAY. Double-port ops are intentionally excluded
# because the current FSM never accepts them into the overlap slot.
# Cases in this file: 234
# @TIMEOUT 600000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_pw_vpu_overlap_pointwise_pack.py
#
# DRAM inputs:
#   [0x0000..0x07FF]  safe slot1 BF16 tensor pair
#   [0x0800..0x0FFF]  signed slot1 BF16 tensor pair
#   [0x1000..0x17FF]  safe slot2 BF16 tensor pair
#   [0x1800..0x1FFF]  signed slot2 BF16 tensor pair
#   [0x2000..0x23FF]  packed slot1 source bank
#   [0x2400..0x27FF]  packed slot2 source bank
# Outputs start at 0x2800; each case stores 4 signatures x 64 B = 256 B:
#   slot1 low, slot1 high, slot2 low, slot2 high

    ADDI  x28, x0, 1
    LI    x5, 0x00000001
    DMA.CONFIG x5, 0
    LI    x2, 1024
    LI    x21, 64

    # Scale registers for fp8pack / fp8unpack cases
    SELI  0, 0x80
    SELI  1, 0x81
    SELI  2, 0x82
    SELI  3, 0x7F

    # Load safe slot1 pair -> m0,m1
    LI    x6, 0
    LI    x1, 0x0000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 0x0400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Load signed slot1 pair -> m2,m3
    LI    x6, 512
    LI    x1, 0x0800
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 768
    LI    x1, 0x0C00
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 512
    VLOAD 2, x6, 0
    DELAY 33
    VLOAD 3, x6, 8
    DELAY 33

    # Load safe slot2 pair -> m4,m5
    LI    x6, 1024
    LI    x1, 0x1000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1280
    LI    x1, 0x1400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1024
    VLOAD 4, x6, 0
    DELAY 33
    VLOAD 5, x6, 8
    DELAY 33

    # Load signed slot2 pair -> m6,m7
    LI    x6, 1536
    LI    x1, 0x1800
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1792
    LI    x1, 0x1C00
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1536
    VLOAD 6, x6, 0
    DELAY 33
    VLOAD 7, x6, 8
    DELAY 33

    # Load packed slot1 bank -> m8
    LI    x6, 2048
    LI    x1, 0x2000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 2048
    VLOAD 8, x6, 0
    DELAY 33

    # Load packed slot2 bank -> m9
    LI    x6, 2304
    LI    x1, 0x2400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 2304
    VLOAD 9, x6, 0
    DELAY 33

    # ----------------------------------------------------------------
    # Case 000: mov || rcp
    # ----------------------------------------------------------------
    VMOV 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x2800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x2840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x2880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x28C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 001: mov || exp
    # ----------------------------------------------------------------
    VMOV 16, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x2900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x2940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x2980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x29C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 002: mov || exp2
    # ----------------------------------------------------------------
    VMOV 16, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x2A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x2A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x2A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x2AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 003: mov || square
    # ----------------------------------------------------------------
    VMOV 16, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x2B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x2B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x2B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x2BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 004: mov || cube
    # ----------------------------------------------------------------
    VMOV 16, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x2C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x2C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x2C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x2CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 005: mov || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VMOV 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x2D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x2D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x2D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x2DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 006: mov || fp8unpack
    # ----------------------------------------------------------------
    VMOV 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x2E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x2E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x2E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x2EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 007: mov || relu
    # ----------------------------------------------------------------
    VMOV 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x2F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x2F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x2F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x2FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 008: mov || sin
    # ----------------------------------------------------------------
    VMOV 16, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x30C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 009: mov || cos
    # ----------------------------------------------------------------
    VMOV 16, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x31C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 010: mov || tanh
    # ----------------------------------------------------------------
    VMOV 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x32C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 011: mov || log
    # ----------------------------------------------------------------
    VMOV 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x33C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 012: mov || sqrt
    # ----------------------------------------------------------------
    VMOV 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x34C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 013: mov || csum
    # ----------------------------------------------------------------
    VMOV 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x35C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 014: mov || cmin
    # ----------------------------------------------------------------
    VMOV 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x36C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 015: mov || cmax
    # ----------------------------------------------------------------
    VMOV 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x37C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 016: mov || vliAll
    # ----------------------------------------------------------------
    VMOV 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x38C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 017: mov || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VMOV 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x39C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 018: mov || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VMOV 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x3AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 019: mov || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VMOV 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x3BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 020: rcp || mov
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x3CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 021: rcp || exp
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x3DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 022: rcp || exp2
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x3EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 023: rcp || square
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x3F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x3F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x3F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x3FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 024: rcp || cube
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x40C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 025: rcp || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VRECIP.BF16 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x41C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 026: rcp || fp8unpack
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x42C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 027: rcp || relu
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x43C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 028: rcp || sin
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x44C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 029: rcp || cos
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x45C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 030: rcp || tanh
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x46C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 031: rcp || log
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x47C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 032: rcp || sqrt
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x48C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 033: rcp || csum
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x49C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 034: rcp || cmin
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x4AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 035: rcp || cmax
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x4BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 036: rcp || vliAll
    # ----------------------------------------------------------------
    VRECIP.BF16 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x4CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 037: rcp || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VRECIP.BF16 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x4DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 038: rcp || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VRECIP.BF16 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x4EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 039: rcp || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VRECIP.BF16 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x4F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x4F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x4F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x4FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 040: exp || mov
    # ----------------------------------------------------------------
    VEXP 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x50C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 041: exp || rcp
    # ----------------------------------------------------------------
    VEXP 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x51C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 042: exp || square
    # ----------------------------------------------------------------
    VEXP 16, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x52C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 043: exp || cube
    # ----------------------------------------------------------------
    VEXP 16, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x53C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 044: exp || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VEXP 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x54C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 045: exp || fp8unpack
    # ----------------------------------------------------------------
    VEXP 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x55C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 046: exp || relu
    # ----------------------------------------------------------------
    VEXP 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x56C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 047: exp || sin
    # ----------------------------------------------------------------
    VEXP 16, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x57C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 048: exp || cos
    # ----------------------------------------------------------------
    VEXP 16, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x58C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 049: exp || tanh
    # ----------------------------------------------------------------
    VEXP 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x59C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 050: exp || log
    # ----------------------------------------------------------------
    VEXP 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x5AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 051: exp || sqrt
    # ----------------------------------------------------------------
    VEXP 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x5BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 052: exp || csum
    # ----------------------------------------------------------------
    VEXP 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x5CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 053: exp || cmin
    # ----------------------------------------------------------------
    VEXP 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x5DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 054: exp || cmax
    # ----------------------------------------------------------------
    VEXP 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x5EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 055: exp || vliAll
    # ----------------------------------------------------------------
    VEXP 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x5F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x5F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x5F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x5FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 056: exp || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VEXP 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x60C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 057: exp || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VEXP 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x61C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 058: exp || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VEXP 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x62C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 059: exp2 || mov
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x63C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 060: exp2 || rcp
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x64C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 061: exp2 || square
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x65C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 062: exp2 || cube
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x66C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 063: exp2 || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VEXP2 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x67C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 064: exp2 || fp8unpack
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x68C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 065: exp2 || relu
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x69C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 066: exp2 || sin
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x6AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 067: exp2 || cos
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x6BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 068: exp2 || tanh
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x6CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 069: exp2 || log
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x6DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 070: exp2 || sqrt
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x6EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 071: exp2 || csum
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x6F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x6F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x6F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x6FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 072: exp2 || cmin
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x70C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 073: exp2 || cmax
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x71C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 074: exp2 || vliAll
    # ----------------------------------------------------------------
    VEXP2 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x72C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 075: exp2 || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VEXP2 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x73C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 076: exp2 || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VEXP2 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x74C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 077: exp2 || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VEXP2 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x75C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 078: square || mov
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x76C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 079: square || rcp
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x77C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 080: square || exp
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x78C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 081: square || exp2
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x79C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 082: square || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSQUARE.BF16 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x7AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 083: square || fp8unpack
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x7BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 084: square || relu
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x7CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 085: square || sin
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x7DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 086: square || cos
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x7EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 087: square || tanh
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x7F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x7F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x7F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x7FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 088: square || log
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x80C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 089: square || sqrt
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x81C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 090: square || csum
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x82C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 091: square || cmin
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x83C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 092: square || cmax
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x84C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 093: square || vliAll
    # ----------------------------------------------------------------
    VSQUARE.BF16 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x85C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 094: square || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSQUARE.BF16 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x86C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 095: square || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSQUARE.BF16 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x87C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 096: square || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSQUARE.BF16 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x88C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 097: cube || mov
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x89C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 098: cube || rcp
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x8AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 099: cube || exp
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x8BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 100: cube || exp2
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x8CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 101: cube || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VCUBE.BF16 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x8DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 102: cube || fp8unpack
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x8EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 103: cube || relu
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x8F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x8F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x8F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x8FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 104: cube || sin
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 105: cube || cos
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x91C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 106: cube || tanh
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x92C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 107: cube || log
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x93C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 108: cube || sqrt
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x94C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 109: cube || csum
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x95C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 110: cube || cmin
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x96C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 111: cube || cmax
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x97C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 112: cube || vliAll
    # ----------------------------------------------------------------
    VCUBE.BF16 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x98C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 113: cube || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VCUBE.BF16 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x99C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 114: cube || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VCUBE.BF16 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 115: cube || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VCUBE.BF16 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 116: fp8pack || mov
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 117: fp8pack || rcp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 118: fp8pack || exp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 119: fp8pack || exp2
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 120: fp8pack || square
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA0C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 121: fp8pack || cube
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA1C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 122: fp8pack || fp8unpack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA2C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 123: fp8pack || relu
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA3C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 124: fp8pack || sin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA4C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 125: fp8pack || cos
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA5C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 126: fp8pack || tanh
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA6C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 127: fp8pack || log
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA7C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 128: fp8pack || sqrt
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA8C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 129: fp8pack || csum
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xA900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xA940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xA980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xA9C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 130: fp8pack || cmin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xAA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xAA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xAA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xAAC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 131: fp8pack || cmax
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xAB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xAB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xAB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xABC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 132: fp8pack || vliAll
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xAC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xAC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xAC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xACC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 133: fp8pack || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ALL 18, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xAD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xAD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xAD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xADC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 134: fp8pack || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ALL 18, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xAE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xAE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xAE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xAEC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 135: fp8pack || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ALL 18, 0x0000
    DELAY 64
    VFP8PACK 16, 0, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xAF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xAF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xAF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xAFC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 136: fp8unpack || mov
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB0C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 137: fp8unpack || rcp
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB1C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 138: fp8unpack || exp
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB2C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 139: fp8unpack || exp2
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB3C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 140: fp8unpack || square
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB4C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 141: fp8unpack || cube
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB5C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 142: fp8unpack || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VFP8UNPACK 16, 8, 1
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB6C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 143: fp8unpack || relu
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB7C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 144: fp8unpack || sin
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB8C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 145: fp8unpack || cos
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xB900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xB940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xB980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xB9C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 146: fp8unpack || tanh
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xBA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xBA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xBA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xBAC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 147: fp8unpack || log
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xBB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xBB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xBB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xBBC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 148: fp8unpack || sqrt
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xBC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xBC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xBC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xBCC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 149: fp8unpack || csum
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xBD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xBD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xBD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xBDC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 150: fp8unpack || cmin
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xBE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xBE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xBE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xBEC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 151: fp8unpack || cmax
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xBF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xBF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xBF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xBFC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 152: fp8unpack || vliAll
    # ----------------------------------------------------------------
    VFP8UNPACK 16, 8, 1
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC0C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 153: fp8unpack || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VFP8UNPACK 16, 8, 1
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC1C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 154: fp8unpack || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VFP8UNPACK 16, 8, 1
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC2C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 155: fp8unpack || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VFP8UNPACK 16, 8, 1
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC3C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 156: relu || mov
    # ----------------------------------------------------------------
    VRELU 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC4C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 157: relu || rcp
    # ----------------------------------------------------------------
    VRELU 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC5C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 158: relu || exp
    # ----------------------------------------------------------------
    VRELU 16, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC6C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 159: relu || exp2
    # ----------------------------------------------------------------
    VRELU 16, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC7C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 160: relu || square
    # ----------------------------------------------------------------
    VRELU 16, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC8C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 161: relu || cube
    # ----------------------------------------------------------------
    VRELU 16, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xC900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xC940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xC980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xC9C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 162: relu || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VRELU 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xCA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xCA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xCA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xCAC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 163: relu || fp8unpack
    # ----------------------------------------------------------------
    VRELU 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xCB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xCB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xCB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xCBC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 164: relu || sin
    # ----------------------------------------------------------------
    VRELU 16, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xCC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xCC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xCC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xCCC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 165: relu || cos
    # ----------------------------------------------------------------
    VRELU 16, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xCD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xCD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xCD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xCDC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 166: relu || tanh
    # ----------------------------------------------------------------
    VRELU 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xCE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xCE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xCE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xCEC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 167: relu || log
    # ----------------------------------------------------------------
    VRELU 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xCF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xCF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xCF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xCFC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 168: relu || sqrt
    # ----------------------------------------------------------------
    VRELU 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD0C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 169: relu || csum
    # ----------------------------------------------------------------
    VRELU 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD1C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 170: relu || cmin
    # ----------------------------------------------------------------
    VRELU 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD2C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 171: relu || cmax
    # ----------------------------------------------------------------
    VRELU 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD3C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 172: relu || vliAll
    # ----------------------------------------------------------------
    VRELU 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD4C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 173: relu || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VRELU 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD5C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 174: relu || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VRELU 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD6C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 175: relu || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VRELU 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD7C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 176: sin || mov
    # ----------------------------------------------------------------
    VSIN 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD8C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 177: sin || rcp
    # ----------------------------------------------------------------
    VSIN 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xD900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xD940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xD980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xD9C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 178: sin || exp
    # ----------------------------------------------------------------
    VSIN 16, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xDA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xDA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xDA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xDAC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 179: sin || exp2
    # ----------------------------------------------------------------
    VSIN 16, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xDB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xDB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xDB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xDBC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 180: sin || square
    # ----------------------------------------------------------------
    VSIN 16, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xDC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xDC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xDC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xDCC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 181: sin || cube
    # ----------------------------------------------------------------
    VSIN 16, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xDD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xDD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xDD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xDDC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 182: sin || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSIN 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xDE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xDE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xDE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xDEC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 183: sin || fp8unpack
    # ----------------------------------------------------------------
    VSIN 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xDF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xDF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xDF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xDFC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 184: sin || relu
    # ----------------------------------------------------------------
    VSIN 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE0C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 185: sin || tanh
    # ----------------------------------------------------------------
    VSIN 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE1C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 186: sin || log
    # ----------------------------------------------------------------
    VSIN 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE2C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 187: sin || sqrt
    # ----------------------------------------------------------------
    VSIN 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE3C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 188: sin || csum
    # ----------------------------------------------------------------
    VSIN 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE4C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 189: sin || cmin
    # ----------------------------------------------------------------
    VSIN 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE5C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 190: sin || cmax
    # ----------------------------------------------------------------
    VSIN 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE6C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 191: sin || vliAll
    # ----------------------------------------------------------------
    VSIN 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE7C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 192: sin || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSIN 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE8C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 193: sin || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSIN 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xE900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xE940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xE980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xE9C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 194: sin || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSIN 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xEA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xEA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xEA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xEAC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 195: cos || mov
    # ----------------------------------------------------------------
    VCOS 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xEB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xEB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xEB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xEBC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 196: cos || rcp
    # ----------------------------------------------------------------
    VCOS 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xEC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xEC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xEC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xECC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 197: cos || exp
    # ----------------------------------------------------------------
    VCOS 16, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xED00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xED40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xED80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xEDC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 198: cos || exp2
    # ----------------------------------------------------------------
    VCOS 16, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xEE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xEE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xEE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xEEC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 199: cos || square
    # ----------------------------------------------------------------
    VCOS 16, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xEF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xEF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xEF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xEFC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 200: cos || cube
    # ----------------------------------------------------------------
    VCOS 16, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF0C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 201: cos || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VCOS 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF1C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 202: cos || fp8unpack
    # ----------------------------------------------------------------
    VCOS 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF2C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 203: cos || relu
    # ----------------------------------------------------------------
    VCOS 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF3C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 204: cos || tanh
    # ----------------------------------------------------------------
    VCOS 16, 0
    VTANH 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF4C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 205: cos || log
    # ----------------------------------------------------------------
    VCOS 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF5C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 206: cos || sqrt
    # ----------------------------------------------------------------
    VCOS 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF6C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 207: cos || csum
    # ----------------------------------------------------------------
    VCOS 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF7C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 208: cos || cmin
    # ----------------------------------------------------------------
    VCOS 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF8C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 209: cos || cmax
    # ----------------------------------------------------------------
    VCOS 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xF900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xF940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xF980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xF9C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 210: cos || vliAll
    # ----------------------------------------------------------------
    VCOS 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xFA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xFA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xFA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xFAC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 211: cos || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VCOS 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xFB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xFB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xFB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xFBC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 212: cos || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VCOS 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xFC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xFC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xFC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xFCC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 213: cos || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VCOS 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xFD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xFD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xFD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xFDC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 214: tanh || mov
    # ----------------------------------------------------------------
    VTANH 16, 0
    VMOV 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xFE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xFE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xFE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xFEC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 215: tanh || rcp
    # ----------------------------------------------------------------
    VTANH 16, 0
    VRECIP.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0xFF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0xFF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0xFF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0xFFC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 216: tanh || exp
    # ----------------------------------------------------------------
    VTANH 16, 0
    VEXP 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x100C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 217: tanh || exp2
    # ----------------------------------------------------------------
    VTANH 16, 0
    VEXP2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x101C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 218: tanh || square
    # ----------------------------------------------------------------
    VTANH 16, 0
    VSQUARE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x102C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 219: tanh || cube
    # ----------------------------------------------------------------
    VTANH 16, 0
    VCUBE.BF16 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x103C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 220: tanh || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VTANH 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x104C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 221: tanh || fp8unpack
    # ----------------------------------------------------------------
    VTANH 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x105C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 222: tanh || relu
    # ----------------------------------------------------------------
    VTANH 16, 0
    VRELU 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x106C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 223: tanh || sin
    # ----------------------------------------------------------------
    VTANH 16, 0
    VSIN 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x107C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 224: tanh || cos
    # ----------------------------------------------------------------
    VTANH 16, 0
    VCOS 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x108C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 225: tanh || log
    # ----------------------------------------------------------------
    VTANH 16, 0
    VLOG2 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x109C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 226: tanh || sqrt
    # ----------------------------------------------------------------
    VTANH 16, 0
    VSQRT 18, 4
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x10AC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 227: tanh || csum
    # ----------------------------------------------------------------
    VTANH 16, 0
    VREDSUM.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x10BC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 228: tanh || cmin
    # ----------------------------------------------------------------
    VTANH 16, 0
    VREDMIN.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x10CC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 229: tanh || cmax
    # ----------------------------------------------------------------
    VTANH 16, 0
    VREDMAX.BF16 18, 6
    DELAY 128
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x10DC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 230: tanh || vliAll
    # ----------------------------------------------------------------
    VTANH 16, 0
    VLI.ALL 18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x10EC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 231: tanh || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VTANH 16, 0
    VLI.ROW 18, 0x4020
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x10F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x10F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x10F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x10FC0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 232: tanh || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VTANH 16, 0
    VLI.COL 19, 0x4070
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x11000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x11040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x11080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x110C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

    # ----------------------------------------------------------------
    # Case 233: tanh || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VTANH 16, 0
    VLI.ONE 19, 0x4090
    DELAY 64
    LI    x8, 1024
    VSTORE 16, x8, 0
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x11100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x11140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x11180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x111C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL


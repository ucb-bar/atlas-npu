# pw_vpu_overlap_reduce_vli.S — pairwise VPU overlap with reduce/VLI first-op cases
# Exhaustive ordered overlap coverage for the current VPU single-input
# overlap window: each case issues op1 and op2 in consecutive cycles
# with no intervening DELAY. Double-port ops are intentionally excluded
# because the current FSM never accepts them into the overlap slot.
# Cases in this file: 168
# @TIMEOUT 600000
# @DRAM_BASE 0x00000001
# @PYTHON_GEN gen_pw_vpu_overlap_reduce_vli.py
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
    # Case 000: log || mov
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VMOV 18, 4
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
    # Case 001: log || rcp
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VRECIP.BF16 18, 4
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
    # Case 002: log || exp
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VEXP 18, 4
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
    # Case 003: log || exp2
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VEXP2 18, 4
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
    # Case 004: log || square
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VSQUARE.BF16 18, 4
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
    # Case 005: log || cube
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VCUBE.BF16 18, 4
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
    # Case 006: log || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VLOG2 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
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
    # Case 007: log || fp8unpack
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 008: log || relu
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VRELU 18, 4
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
    # Case 009: log || sin
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VSIN 18, 4
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
    # Case 010: log || cos
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VCOS 18, 4
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
    # Case 011: log || tanh
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VTANH 18, 4
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
    # Case 012: log || sqrt
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 013: log || csum
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 014: log || cmin
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 015: log || cmax
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 016: log || vliAll
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 017: log || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VLOG2 16, 0
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
    # Case 018: log || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VLOG2 16, 0
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
    # Case 019: log || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VLOG2 16, 0
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
    # Case 020: sqrt || mov
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 021: sqrt || rcp
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VRECIP.BF16 18, 4
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
    # Case 022: sqrt || exp
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VEXP 18, 4
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
    # Case 023: sqrt || exp2
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VEXP2 18, 4
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
    # Case 024: sqrt || square
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VSQUARE.BF16 18, 4
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
    # Case 025: sqrt || cube
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VCUBE.BF16 18, 4
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
    # Case 026: sqrt || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSQRT 16, 0
    VFP8PACK 18, 4, 2
    DELAY 64
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
    # Case 027: sqrt || fp8unpack
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 028: sqrt || relu
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VRELU 18, 4
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
    # Case 029: sqrt || sin
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VSIN 18, 4
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
    # Case 030: sqrt || cos
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VCOS 18, 4
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
    # Case 031: sqrt || tanh
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VTANH 18, 4
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
    # Case 032: sqrt || log
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VLOG2 18, 4
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
    # Case 033: sqrt || csum
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 034: sqrt || cmin
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 035: sqrt || cmax
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 036: sqrt || vliAll
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 037: sqrt || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSQRT 16, 0
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
    # Case 038: sqrt || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSQRT 16, 0
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
    # Case 039: sqrt || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VSQRT 16, 0
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
    # Case 040: csum || mov
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VMOV 18, 4
    DELAY 128
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
    # Case 041: csum || rcp
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VRECIP.BF16 18, 4
    DELAY 128
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
    # Case 042: csum || exp
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VEXP 18, 4
    DELAY 128
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
    # Case 043: csum || exp2
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VEXP2 18, 4
    DELAY 128
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
    # Case 044: csum || square
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VSQUARE.BF16 18, 4
    DELAY 128
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
    # Case 045: csum || cube
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VCUBE.BF16 18, 4
    DELAY 128
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
    # Case 046: csum || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDSUM.BF16 16, 2
    VFP8PACK 18, 4, 2
    DELAY 128
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
    # Case 047: csum || fp8unpack
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VFP8UNPACK 18, 9, 3
    DELAY 128
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
    # Case 048: csum || relu
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VRELU 18, 4
    DELAY 128
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
    # Case 049: csum || sin
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VSIN 18, 4
    DELAY 128
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
    # Case 050: csum || cos
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VCOS 18, 4
    DELAY 128
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
    # Case 051: csum || tanh
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VTANH 18, 4
    DELAY 128
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
    # Case 052: csum || log
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VLOG2 18, 4
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
    # Case 053: csum || sqrt
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VSQRT 18, 4
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
    # Case 054: csum || cmin
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VREDMIN.BF16 18, 6
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
    # Case 055: csum || cmax
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VREDMAX.BF16 18, 6
    DELAY 128
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
    # Case 056: csum || vliAll
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VLI.ALL 18, 0x3FA0
    DELAY 128
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
    # Case 057: csum || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDSUM.BF16 16, 2
    VLI.ROW 18, 0x4020
    DELAY 128
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
    # Case 058: csum || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDSUM.BF16 16, 2
    VLI.COL 19, 0x4070
    DELAY 128
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
    # Case 059: csum || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDSUM.BF16 16, 2
    VLI.ONE 19, 0x4090
    DELAY 128
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
    # Case 060: cmin || mov
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VMOV 18, 4
    DELAY 128
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
    # Case 061: cmin || rcp
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VRECIP.BF16 18, 4
    DELAY 128
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
    # Case 062: cmin || exp
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VEXP 18, 4
    DELAY 128
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
    # Case 063: cmin || exp2
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VEXP2 18, 4
    DELAY 128
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
    # Case 064: cmin || square
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VSQUARE.BF16 18, 4
    DELAY 128
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
    # Case 065: cmin || cube
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VCUBE.BF16 18, 4
    DELAY 128
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
    # Case 066: cmin || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDMIN.BF16 16, 2
    VFP8PACK 18, 4, 2
    DELAY 128
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
    # Case 067: cmin || fp8unpack
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VFP8UNPACK 18, 9, 3
    DELAY 128
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
    # Case 068: cmin || relu
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VRELU 18, 4
    DELAY 128
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
    # Case 069: cmin || sin
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VSIN 18, 4
    DELAY 128
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
    # Case 070: cmin || cos
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VCOS 18, 4
    DELAY 128
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
    # Case 071: cmin || tanh
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VTANH 18, 4
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
    # Case 072: cmin || log
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VLOG2 18, 4
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
    # Case 073: cmin || sqrt
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VSQRT 18, 4
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
    # Case 074: cmin || csum
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VREDSUM.BF16 18, 6
    DELAY 128
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
    # Case 075: cmin || cmax
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VREDMAX.BF16 18, 6
    DELAY 128
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
    # Case 076: cmin || vliAll
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VLI.ALL 18, 0x3FA0
    DELAY 128
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
    # Case 077: cmin || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDMIN.BF16 16, 2
    VLI.ROW 18, 0x4020
    DELAY 128
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
    # Case 078: cmin || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDMIN.BF16 16, 2
    VLI.COL 19, 0x4070
    DELAY 128
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
    # Case 079: cmin || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDMIN.BF16 16, 2
    VLI.ONE 19, 0x4090
    DELAY 128
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
    # Case 080: cmax || mov
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VMOV 18, 4
    DELAY 128
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
    # Case 081: cmax || rcp
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VRECIP.BF16 18, 4
    DELAY 128
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
    # Case 082: cmax || exp
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VEXP 18, 4
    DELAY 128
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
    # Case 083: cmax || exp2
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VEXP2 18, 4
    DELAY 128
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
    # Case 084: cmax || square
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VSQUARE.BF16 18, 4
    DELAY 128
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
    # Case 085: cmax || cube
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VCUBE.BF16 18, 4
    DELAY 128
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
    # Case 086: cmax || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDMAX.BF16 16, 2
    VFP8PACK 18, 4, 2
    DELAY 128
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
    # Case 087: cmax || fp8unpack
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VFP8UNPACK 18, 9, 3
    DELAY 128
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
    # Case 088: cmax || relu
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VRELU 18, 4
    DELAY 128
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
    # Case 089: cmax || sin
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VSIN 18, 4
    DELAY 128
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
    # Case 090: cmax || cos
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VCOS 18, 4
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
    # Case 091: cmax || tanh
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VTANH 18, 4
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
    # Case 092: cmax || log
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VLOG2 18, 4
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
    # Case 093: cmax || sqrt
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VSQRT 18, 4
    DELAY 128
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
    # Case 094: cmax || csum
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VREDSUM.BF16 18, 6
    DELAY 128
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
    # Case 095: cmax || cmin
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VREDMIN.BF16 18, 6
    DELAY 128
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
    # Case 096: cmax || vliAll
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VLI.ALL 18, 0x3FA0
    DELAY 128
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
    # Case 097: cmax || vliRow
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDMAX.BF16 16, 2
    VLI.ROW 18, 0x4020
    DELAY 128
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
    # Case 098: cmax || vliCol
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDMAX.BF16 16, 2
    VLI.COL 19, 0x4070
    DELAY 128
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
    # Case 099: cmax || vliOne
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VREDMAX.BF16 16, 2
    VLI.ONE 19, 0x4090
    DELAY 128
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
    # Case 100: vliAll || mov
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VMOV 18, 4
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
    # Case 101: vliAll || rcp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VRECIP.BF16 18, 4
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
    # Case 102: vliAll || exp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VEXP 18, 4
    DELAY 64
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
    # Case 103: vliAll || exp2
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VEXP2 18, 4
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
    # Case 104: vliAll || square
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VSQUARE.BF16 18, 4
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
    # Case 105: vliAll || cube
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VCUBE.BF16 18, 4
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
    # Case 106: vliAll || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 18, 0x0000
    DELAY 64
    VLI.ALL 16, 0x3FA0
    VFP8PACK 18, 4, 2
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
    # Case 107: vliAll || fp8unpack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 108: vliAll || relu
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VRELU 18, 4
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
    # Case 109: vliAll || sin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VSIN 18, 4
    DELAY 64
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
    # Case 110: vliAll || cos
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VCOS 18, 4
    DELAY 64
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
    # Case 111: vliAll || tanh
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VTANH 18, 4
    DELAY 64
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
    # Case 112: vliAll || log
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VLOG2 18, 4
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
    # Case 113: vliAll || sqrt
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VSQRT 18, 4
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
    # Case 114: vliAll || csum
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VREDSUM.BF16 18, 6
    DELAY 128
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
    # Case 115: vliAll || cmin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VREDMIN.BF16 18, 6
    DELAY 128
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
    # Case 116: vliAll || cmax
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VREDMAX.BF16 18, 6
    DELAY 128
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
    # Case 117: vliRow || mov
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VMOV 18, 4
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
    # Case 118: vliRow || rcp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VRECIP.BF16 18, 4
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
    # Case 119: vliRow || exp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VEXP 18, 4
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
    # Case 120: vliRow || exp2
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VEXP2 18, 4
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
    # Case 121: vliRow || square
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VSQUARE.BF16 18, 4
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
    # Case 122: vliRow || cube
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VCUBE.BF16 18, 4
    DELAY 64
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
    # Case 123: vliRow || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ALL 18, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VFP8PACK 18, 4, 2
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
    # Case 124: vliRow || fp8unpack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 125: vliRow || relu
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VRELU 18, 4
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
    # Case 126: vliRow || sin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VSIN 18, 4
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
    # Case 127: vliRow || cos
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VCOS 18, 4
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
    # Case 128: vliRow || tanh
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VTANH 18, 4
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
    # Case 129: vliRow || log
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VLOG2 18, 4
    DELAY 64
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
    # Case 130: vliRow || sqrt
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VSQRT 18, 4
    DELAY 64
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
    # Case 131: vliRow || csum
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VREDSUM.BF16 18, 6
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
    # Case 132: vliRow || cmin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VREDMIN.BF16 18, 6
    DELAY 128
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
    # Case 133: vliRow || cmax
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ROW 16, 0x4020
    VREDMAX.BF16 18, 6
    DELAY 128
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
    # Case 134: vliCol || mov
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VMOV 18, 4
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
    # Case 135: vliCol || rcp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VRECIP.BF16 18, 4
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
    # Case 136: vliCol || exp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VEXP 18, 4
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
    # Case 137: vliCol || exp2
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VEXP2 18, 4
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
    # Case 138: vliCol || square
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VSQUARE.BF16 18, 4
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
    # Case 139: vliCol || cube
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VCUBE.BF16 18, 4
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
    # Case 140: vliCol || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ALL 18, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VFP8PACK 18, 4, 2
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
    # Case 141: vliCol || fp8unpack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 142: vliCol || relu
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VRELU 18, 4
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
    # Case 143: vliCol || sin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VSIN 18, 4
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
    # Case 144: vliCol || cos
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VCOS 18, 4
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
    # Case 145: vliCol || tanh
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VTANH 18, 4
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
    # Case 146: vliCol || log
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VLOG2 18, 4
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
    # Case 147: vliCol || sqrt
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VSQRT 18, 4
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
    # Case 148: vliCol || csum
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VREDSUM.BF16 18, 6
    DELAY 128
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
    # Case 149: vliCol || cmin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VREDMIN.BF16 18, 6
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
    # Case 150: vliCol || cmax
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.COL 17, 0x4070
    VREDMAX.BF16 18, 6
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
    # Case 151: vliOne || mov
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VMOV 18, 4
    DELAY 64
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
    # Case 152: vliOne || rcp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VRECIP.BF16 18, 4
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
    # Case 153: vliOne || exp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VEXP 18, 4
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
    # Case 154: vliOne || exp2
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VEXP2 18, 4
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
    # Case 155: vliOne || square
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VSQUARE.BF16 18, 4
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
    # Case 156: vliOne || cube
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VCUBE.BF16 18, 4
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
    # Case 157: vliOne || fp8pack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ALL 18, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VFP8PACK 18, 4, 2
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
    # Case 158: vliOne || fp8unpack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 159: vliOne || relu
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VRELU 18, 4
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
    # Case 160: vliOne || sin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VSIN 18, 4
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
    # Case 161: vliOne || cos
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VCOS 18, 4
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
    # Case 162: vliOne || tanh
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VTANH 18, 4
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
    # Case 163: vliOne || log
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VLOG2 18, 4
    DELAY 64
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
    # Case 164: vliOne || sqrt
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VSQRT 18, 4
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
    # Case 165: vliOne || csum
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VREDSUM.BF16 18, 6
    DELAY 128
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
    # Case 166: vliOne || cmin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VREDMIN.BF16 18, 6
    DELAY 128
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
    # Case 167: vliOne || cmax
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x0000
    DELAY 64
    VLI.ONE 17, 0x4090
    VREDMAX.BF16 18, 6
    DELAY 128
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

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL


# pw_vpu_overlap_pointwise_pack.S — pairwise VPU overlap with pointwise/fp8 first-op cases
# Exhaustive ordered overlap coverage for the current VPU single-input
# overlap window: each case issues op1 and op2 in consecutive cycles
# with no intervening DELAY. Double-port ops are intentionally excluded
# because the current FSM never accepts them into the overlap slot.
# Cases in this file: 234
# @TIMEOUT 600000
# @DRAM_BASE 0x90000000
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
    LI    x5, 0x00000000
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
    LI    x1, 0x90000000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 0x90000400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 0
    VLOAD 0, x6, 0
    DELAY 33
    VLOAD 1, x6, 8
    DELAY 33

    # Load signed slot1 pair -> m2,m3
    LI    x6, 512
    LI    x1, 0x90000800
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 768
    LI    x1, 0x90000C00
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 512
    VLOAD 2, x6, 0
    DELAY 33
    VLOAD 3, x6, 8
    DELAY 33

    # Load safe slot2 pair -> m4,m5
    LI    x6, 1024
    LI    x1, 0x90001000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1280
    LI    x1, 0x90001400
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1024
    VLOAD 4, x6, 0
    DELAY 33
    VLOAD 5, x6, 8
    DELAY 33

    # Load signed slot2 pair -> m6,m7
    LI    x6, 1536
    LI    x1, 0x90001800
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1792
    LI    x1, 0x90001C00
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 1536
    VLOAD 6, x6, 0
    DELAY 33
    VLOAD 7, x6, 8
    DELAY 33

    # Load packed slot1 bank -> m8
    LI    x6, 2048
    LI    x1, 0x90002000
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 2048
    VLOAD 8, x6, 0
    DELAY 33

    # Load packed slot2 bank -> m9
    LI    x6, 2304
    LI    x1, 0x90002400
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
    LI    x3, 0x90002800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900028C0
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
    LI    x3, 0x90002900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900029C0
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
    LI    x3, 0x90002A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002AC0
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
    LI    x3, 0x90002B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002BC0
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
    LI    x3, 0x90002C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002CC0
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
    LI    x3, 0x90002D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002DC0
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
    LI    x3, 0x90002E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002EC0
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
    LI    x3, 0x90002F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002FC0
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
    LI    x3, 0x90003000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900030C0
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
    LI    x3, 0x90003100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900031C0
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
    LI    x3, 0x90003200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900032C0
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
    LI    x3, 0x90003300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900033C0
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
    LI    x3, 0x90003400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900034C0
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
    LI    x3, 0x90003500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900035C0
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
    LI    x3, 0x90003600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900036C0
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
    LI    x3, 0x90003700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900037C0
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
    LI    x3, 0x90003800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900038C0
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
    LI    x3, 0x90003900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900039C0
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
    LI    x3, 0x90003A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003AC0
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
    LI    x3, 0x90003B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003BC0
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
    LI    x3, 0x90003C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003CC0
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
    LI    x3, 0x90003D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003DC0
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
    LI    x3, 0x90003E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003EC0
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
    LI    x3, 0x90003F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003FC0
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
    LI    x3, 0x90004000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900040C0
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
    LI    x3, 0x90004100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900041C0
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
    LI    x3, 0x90004200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900042C0
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
    LI    x3, 0x90004300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900043C0
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
    LI    x3, 0x90004400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900044C0
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
    LI    x3, 0x90004500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900045C0
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
    LI    x3, 0x90004600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900046C0
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
    LI    x3, 0x90004700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900047C0
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
    LI    x3, 0x90004800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900048C0
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
    LI    x3, 0x90004900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900049C0
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
    LI    x3, 0x90004A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004AC0
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
    LI    x3, 0x90004B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004BC0
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
    LI    x3, 0x90004C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004CC0
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
    LI    x3, 0x90004D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004DC0
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
    LI    x3, 0x90004E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004EC0
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
    LI    x3, 0x90004F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004FC0
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
    LI    x3, 0x90005000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900050C0
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
    LI    x3, 0x90005100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900051C0
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
    LI    x3, 0x90005200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900052C0
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
    LI    x3, 0x90005300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900053C0
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
    LI    x3, 0x90005400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900054C0
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
    LI    x3, 0x90005500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900055C0
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
    LI    x3, 0x90005600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900056C0
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
    LI    x3, 0x90005700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900057C0
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
    LI    x3, 0x90005800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900058C0
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
    LI    x3, 0x90005900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900059C0
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
    LI    x3, 0x90005A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005AC0
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
    LI    x3, 0x90005B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005BC0
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
    LI    x3, 0x90005C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005CC0
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
    LI    x3, 0x90005D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005DC0
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
    LI    x3, 0x90005E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005EC0
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
    LI    x3, 0x90005F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005FC0
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
    LI    x3, 0x90006000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900060C0
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
    LI    x3, 0x90006100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900061C0
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
    LI    x3, 0x90006200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900062C0
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
    LI    x3, 0x90006300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900063C0
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
    LI    x3, 0x90006400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900064C0
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
    LI    x3, 0x90006500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900065C0
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
    LI    x3, 0x90006600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900066C0
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
    LI    x3, 0x90006700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900067C0
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
    LI    x3, 0x90006800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900068C0
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
    LI    x3, 0x90006900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900069C0
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
    LI    x3, 0x90006A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006AC0
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
    LI    x3, 0x90006B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006BC0
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
    LI    x3, 0x90006C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006CC0
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
    LI    x3, 0x90006D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006DC0
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
    LI    x3, 0x90006E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006EC0
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
    LI    x3, 0x90006F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006FC0
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
    LI    x3, 0x90007000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900070C0
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
    LI    x3, 0x90007100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900071C0
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
    LI    x3, 0x90007200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900072C0
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
    LI    x3, 0x90007300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900073C0
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
    LI    x3, 0x90007400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900074C0
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
    LI    x3, 0x90007500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900075C0
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
    LI    x3, 0x90007600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900076C0
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
    LI    x3, 0x90007700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900077C0
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
    LI    x3, 0x90007800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900078C0
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
    LI    x3, 0x90007900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900079C0
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
    LI    x3, 0x90007A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007AC0
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
    LI    x3, 0x90007B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007BC0
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
    LI    x3, 0x90007C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007CC0
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
    LI    x3, 0x90007D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007DC0
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
    LI    x3, 0x90007E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007EC0
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
    LI    x3, 0x90007F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007FC0
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
    LI    x3, 0x90008000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900080C0
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
    LI    x3, 0x90008100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900081C0
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
    LI    x3, 0x90008200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900082C0
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
    LI    x3, 0x90008300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900083C0
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
    LI    x3, 0x90008400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900084C0
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
    LI    x3, 0x90008500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900085C0
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
    LI    x3, 0x90008600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900086C0
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
    LI    x3, 0x90008700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900087C0
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
    LI    x3, 0x90008800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900088C0
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
    LI    x3, 0x90008900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900089C0
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
    LI    x3, 0x90008A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008AC0
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
    LI    x3, 0x90008B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008BC0
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
    LI    x3, 0x90008C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008CC0
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
    LI    x3, 0x90008D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008DC0
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
    LI    x3, 0x90008E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008EC0
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
    LI    x3, 0x90008F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008FC0
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
    LI    x3, 0x90009000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900090C0
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
    LI    x3, 0x90009100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900091C0
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
    LI    x3, 0x90009200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900092C0
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
    LI    x3, 0x90009300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900093C0
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
    LI    x3, 0x90009400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900094C0
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
    LI    x3, 0x90009500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900095C0
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
    LI    x3, 0x90009600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900096C0
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
    LI    x3, 0x90009700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900097C0
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
    LI    x3, 0x90009800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900098C0
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
    LI    x3, 0x90009900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900099C0
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
    LI    x3, 0x90009A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009AC0
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
    LI    x3, 0x90009B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009BC0
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
    LI    x3, 0x90009C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009CC0
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
    LI    x3, 0x90009D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009DC0
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
    LI    x3, 0x90009E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009EC0
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
    LI    x3, 0x90009F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009FC0
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
    LI    x3, 0x9000A000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A0C0
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
    LI    x3, 0x9000A100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A1C0
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
    LI    x3, 0x9000A200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A2C0
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
    LI    x3, 0x9000A300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A3C0
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
    LI    x3, 0x9000A400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A4C0
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
    LI    x3, 0x9000A500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A5C0
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
    LI    x3, 0x9000A600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A6C0
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
    LI    x3, 0x9000A700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A7C0
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
    LI    x3, 0x9000A800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A8C0
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
    LI    x3, 0x9000A900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A9C0
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
    LI    x3, 0x9000AA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000AAC0
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
    LI    x3, 0x9000AB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000ABC0
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
    LI    x3, 0x9000AC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000ACC0
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
    LI    x3, 0x9000AD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000ADC0
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
    LI    x3, 0x9000AE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000AEC0
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
    LI    x3, 0x9000AF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000AFC0
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
    LI    x3, 0x9000B000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B0C0
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
    LI    x3, 0x9000B100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B1C0
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
    LI    x3, 0x9000B200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B2C0
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
    LI    x3, 0x9000B300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B3C0
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
    LI    x3, 0x9000B400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B4C0
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
    LI    x3, 0x9000B500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B5C0
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
    LI    x3, 0x9000B600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B6C0
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
    LI    x3, 0x9000B700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B7C0
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
    LI    x3, 0x9000B800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B8C0
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
    LI    x3, 0x9000B900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B9C0
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
    LI    x3, 0x9000BA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BAC0
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
    LI    x3, 0x9000BB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BBC0
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
    LI    x3, 0x9000BC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BCC0
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
    LI    x3, 0x9000BD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BDC0
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
    LI    x3, 0x9000BE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BEC0
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
    LI    x3, 0x9000BF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BFC0
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
    LI    x3, 0x9000C000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C0C0
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
    LI    x3, 0x9000C100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C1C0
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
    LI    x3, 0x9000C200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C2C0
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
    LI    x3, 0x9000C300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C3C0
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
    LI    x3, 0x9000C400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C4C0
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
    LI    x3, 0x9000C500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C5C0
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
    LI    x3, 0x9000C600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C6C0
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
    LI    x3, 0x9000C700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C7C0
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
    LI    x3, 0x9000C800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C8C0
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
    LI    x3, 0x9000C900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C9C0
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
    LI    x3, 0x9000CA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CAC0
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
    LI    x3, 0x9000CB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CBC0
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
    LI    x3, 0x9000CC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CCC0
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
    LI    x3, 0x9000CD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CDC0
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
    LI    x3, 0x9000CE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CEC0
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
    LI    x3, 0x9000CF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CFC0
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
    LI    x3, 0x9000D000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D0C0
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
    LI    x3, 0x9000D100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D1C0
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
    LI    x3, 0x9000D200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D2C0
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
    LI    x3, 0x9000D300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D3C0
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
    LI    x3, 0x9000D400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D4C0
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
    LI    x3, 0x9000D500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D5C0
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
    LI    x3, 0x9000D600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D6C0
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
    LI    x3, 0x9000D700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D7C0
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
    LI    x3, 0x9000D800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D8C0
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
    LI    x3, 0x9000D900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D9C0
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
    LI    x3, 0x9000DA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DAC0
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
    LI    x3, 0x9000DB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DBC0
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
    LI    x3, 0x9000DC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DCC0
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
    LI    x3, 0x9000DD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DDC0
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
    LI    x3, 0x9000DE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DEC0
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
    LI    x3, 0x9000DF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DFC0
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
    LI    x3, 0x9000E000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E0C0
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
    LI    x3, 0x9000E100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E1C0
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
    LI    x3, 0x9000E200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E2C0
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
    LI    x3, 0x9000E300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E3C0
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
    LI    x3, 0x9000E400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E4C0
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
    LI    x3, 0x9000E500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E5C0
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
    LI    x3, 0x9000E600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E6C0
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
    LI    x3, 0x9000E700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E7C0
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
    LI    x3, 0x9000E800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E8C0
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
    LI    x3, 0x9000E900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E9C0
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
    LI    x3, 0x9000EA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EAC0
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
    LI    x3, 0x9000EB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EBC0
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
    LI    x3, 0x9000EC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000ECC0
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
    LI    x3, 0x9000ED00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000ED40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000ED80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EDC0
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
    LI    x3, 0x9000EE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EEC0
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
    LI    x3, 0x9000EF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EFC0
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
    LI    x3, 0x9000F000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F0C0
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
    LI    x3, 0x9000F100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F1C0
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
    LI    x3, 0x9000F200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F2C0
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
    LI    x3, 0x9000F300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F3C0
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
    LI    x3, 0x9000F400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F4C0
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
    LI    x3, 0x9000F500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F5C0
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
    LI    x3, 0x9000F600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F6C0
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
    LI    x3, 0x9000F700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F7C0
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
    LI    x3, 0x9000F800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F8C0
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
    LI    x3, 0x9000F900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F9C0
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
    LI    x3, 0x9000FA00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FA40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FA80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FAC0
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
    LI    x3, 0x9000FB00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FB40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FB80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FBC0
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
    LI    x3, 0x9000FC00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FC40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FC80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FCC0
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
    LI    x3, 0x9000FD00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FD40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FD80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FDC0
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
    LI    x3, 0x9000FE00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FE40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FE80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FEC0
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
    LI    x3, 0x9000FF00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FF40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FF80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FFC0
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
    LI    x3, 0x90010000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900100C0
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
    LI    x3, 0x90010100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900101C0
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
    LI    x3, 0x90010200
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010240
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010280
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900102C0
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
    LI    x3, 0x90010300
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010340
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010380
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900103C0
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
    LI    x3, 0x90010400
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010440
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010480
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900104C0
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
    LI    x3, 0x90010500
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010540
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010580
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900105C0
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
    LI    x3, 0x90010600
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010640
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010680
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900106C0
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
    LI    x3, 0x90010700
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010740
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010780
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900107C0
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
    LI    x3, 0x90010800
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010840
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010880
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900108C0
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
    LI    x3, 0x90010900
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010940
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010980
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900109C0
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
    LI    x3, 0x90010A00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010A40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010A80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010AC0
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
    LI    x3, 0x90010B00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010B40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010B80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010BC0
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
    LI    x3, 0x90010C00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010C40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010C80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010CC0
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
    LI    x3, 0x90010D00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010D40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010D80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010DC0
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
    LI    x3, 0x90010E00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010E40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010E80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010EC0
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
    LI    x3, 0x90010F00
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010F40
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010F80
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010FC0
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
    LI    x3, 0x90011000
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90011040
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90011080
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900110C0
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
    LI    x3, 0x90011100
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1280
    VSTORE 17, x8, 0
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90011140
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1536
    VSTORE 18, x8, 0
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90011180
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1
    LI    x8, 1792
    VSTORE 19, x8, 0
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900111C0
    DMA.STORE x3, x8, x21, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL


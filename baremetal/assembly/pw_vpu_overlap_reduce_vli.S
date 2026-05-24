# pw_vpu_overlap_reduce_vli.S — pairwise VPU overlap with reduce/VLI first-op cases
# Exhaustive ordered overlap coverage for the current VPU single-input
# overlap window: each case issues op1 and op2 in consecutive cycles
# with no intervening DELAY. Double-port ops are intentionally excluded
# because the current FSM never accepts them into the overlap slot.
# Cases in this file: 168
# @TIMEOUT 600000
# @DRAM_BASE 0x90000000
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
    # Case 000: log || mov
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VMOV 18, 4
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
    # Case 001: log || rcp
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VRECIP.BF16 18, 4
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
    # Case 002: log || exp
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VEXP 18, 4
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
    # Case 003: log || exp2
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VEXP2 18, 4
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
    # Case 004: log || square
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VSQUARE.BF16 18, 4
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
    # Case 005: log || cube
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VCUBE.BF16 18, 4
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
    # Case 007: log || fp8unpack
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 008: log || relu
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VRELU 18, 4
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
    # Case 009: log || sin
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VSIN 18, 4
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
    # Case 010: log || cos
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VCOS 18, 4
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
    # Case 011: log || tanh
    # ----------------------------------------------------------------
    VLOG2 16, 0
    VTANH 18, 4
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
    # Case 012: log || sqrt
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 013: log || csum
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 014: log || cmin
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 015: log || cmax
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 016: log || vliAll
    # ----------------------------------------------------------------
    VLOG2 16, 0
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
    # Case 020: sqrt || mov
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 021: sqrt || rcp
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VRECIP.BF16 18, 4
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
    # Case 022: sqrt || exp
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VEXP 18, 4
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
    # Case 023: sqrt || exp2
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VEXP2 18, 4
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
    # Case 024: sqrt || square
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VSQUARE.BF16 18, 4
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
    # Case 025: sqrt || cube
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VCUBE.BF16 18, 4
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
    # Case 027: sqrt || fp8unpack
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 028: sqrt || relu
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VRELU 18, 4
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
    # Case 029: sqrt || sin
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VSIN 18, 4
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
    # Case 030: sqrt || cos
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VCOS 18, 4
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
    # Case 031: sqrt || tanh
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VTANH 18, 4
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
    # Case 032: sqrt || log
    # ----------------------------------------------------------------
    VSQRT 16, 0
    VLOG2 18, 4
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
    # Case 033: sqrt || csum
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 034: sqrt || cmin
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 035: sqrt || cmax
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 036: sqrt || vliAll
    # ----------------------------------------------------------------
    VSQRT 16, 0
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
    # Case 040: csum || mov
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VMOV 18, 4
    DELAY 128
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
    # Case 041: csum || rcp
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VRECIP.BF16 18, 4
    DELAY 128
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
    # Case 042: csum || exp
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VEXP 18, 4
    DELAY 128
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
    # Case 043: csum || exp2
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VEXP2 18, 4
    DELAY 128
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
    # Case 044: csum || square
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VSQUARE.BF16 18, 4
    DELAY 128
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
    # Case 045: csum || cube
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VCUBE.BF16 18, 4
    DELAY 128
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
    # Case 047: csum || fp8unpack
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VFP8UNPACK 18, 9, 3
    DELAY 128
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
    # Case 048: csum || relu
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VRELU 18, 4
    DELAY 128
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
    # Case 049: csum || sin
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VSIN 18, 4
    DELAY 128
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
    # Case 050: csum || cos
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VCOS 18, 4
    DELAY 128
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
    # Case 051: csum || tanh
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VTANH 18, 4
    DELAY 128
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
    # Case 052: csum || log
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VLOG2 18, 4
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
    # Case 053: csum || sqrt
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VSQRT 18, 4
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
    # Case 054: csum || cmin
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VREDMIN.BF16 18, 6
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
    # Case 055: csum || cmax
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VREDMAX.BF16 18, 6
    DELAY 128
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
    # Case 056: csum || vliAll
    # ----------------------------------------------------------------
    VREDSUM.BF16 16, 2
    VLI.ALL 18, 0x3FA0
    DELAY 128
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
    # Case 060: cmin || mov
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VMOV 18, 4
    DELAY 128
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
    # Case 061: cmin || rcp
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VRECIP.BF16 18, 4
    DELAY 128
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
    # Case 062: cmin || exp
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VEXP 18, 4
    DELAY 128
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
    # Case 063: cmin || exp2
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VEXP2 18, 4
    DELAY 128
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
    # Case 064: cmin || square
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VSQUARE.BF16 18, 4
    DELAY 128
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
    # Case 065: cmin || cube
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VCUBE.BF16 18, 4
    DELAY 128
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
    # Case 067: cmin || fp8unpack
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VFP8UNPACK 18, 9, 3
    DELAY 128
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
    # Case 068: cmin || relu
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VRELU 18, 4
    DELAY 128
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
    # Case 069: cmin || sin
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VSIN 18, 4
    DELAY 128
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
    # Case 070: cmin || cos
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VCOS 18, 4
    DELAY 128
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
    # Case 071: cmin || tanh
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VTANH 18, 4
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
    # Case 072: cmin || log
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VLOG2 18, 4
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
    # Case 073: cmin || sqrt
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VSQRT 18, 4
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
    # Case 074: cmin || csum
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VREDSUM.BF16 18, 6
    DELAY 128
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
    # Case 075: cmin || cmax
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VREDMAX.BF16 18, 6
    DELAY 128
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
    # Case 076: cmin || vliAll
    # ----------------------------------------------------------------
    VREDMIN.BF16 16, 2
    VLI.ALL 18, 0x3FA0
    DELAY 128
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
    # Case 080: cmax || mov
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VMOV 18, 4
    DELAY 128
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
    # Case 081: cmax || rcp
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VRECIP.BF16 18, 4
    DELAY 128
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
    # Case 082: cmax || exp
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VEXP 18, 4
    DELAY 128
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
    # Case 083: cmax || exp2
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VEXP2 18, 4
    DELAY 128
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
    # Case 084: cmax || square
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VSQUARE.BF16 18, 4
    DELAY 128
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
    # Case 085: cmax || cube
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VCUBE.BF16 18, 4
    DELAY 128
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
    # Case 087: cmax || fp8unpack
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VFP8UNPACK 18, 9, 3
    DELAY 128
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
    # Case 088: cmax || relu
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VRELU 18, 4
    DELAY 128
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
    # Case 089: cmax || sin
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VSIN 18, 4
    DELAY 128
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
    # Case 090: cmax || cos
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VCOS 18, 4
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
    # Case 091: cmax || tanh
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VTANH 18, 4
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
    # Case 092: cmax || log
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VLOG2 18, 4
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
    # Case 093: cmax || sqrt
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VSQRT 18, 4
    DELAY 128
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
    # Case 094: cmax || csum
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VREDSUM.BF16 18, 6
    DELAY 128
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
    # Case 095: cmax || cmin
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VREDMIN.BF16 18, 6
    DELAY 128
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
    # Case 096: cmax || vliAll
    # ----------------------------------------------------------------
    VREDMAX.BF16 16, 2
    VLI.ALL 18, 0x3FA0
    DELAY 128
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
    # Case 100: vliAll || mov
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VMOV 18, 4
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
    # Case 101: vliAll || rcp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VRECIP.BF16 18, 4
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
    # Case 102: vliAll || exp
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VEXP 18, 4
    DELAY 64
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
    # Case 103: vliAll || exp2
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VEXP2 18, 4
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
    # Case 104: vliAll || square
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VSQUARE.BF16 18, 4
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
    # Case 105: vliAll || cube
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VCUBE.BF16 18, 4
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
    # Case 107: vliAll || fp8unpack
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VFP8UNPACK 18, 9, 3
    DELAY 65
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
    # Case 108: vliAll || relu
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VRELU 18, 4
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
    # Case 109: vliAll || sin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VSIN 18, 4
    DELAY 64
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
    # Case 110: vliAll || cos
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VCOS 18, 4
    DELAY 64
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
    # Case 111: vliAll || tanh
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VTANH 18, 4
    DELAY 64
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
    # Case 112: vliAll || log
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VLOG2 18, 4
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
    # Case 113: vliAll || sqrt
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VSQRT 18, 4
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
    # Case 114: vliAll || csum
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VREDSUM.BF16 18, 6
    DELAY 128
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
    # Case 115: vliAll || cmin
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VREDMIN.BF16 18, 6
    DELAY 128
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
    # Case 116: vliAll || cmax
    # ----------------------------------------------------------------
    VLI.ALL 16, 0x3FA0
    VREDMAX.BF16 18, 6
    DELAY 128
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

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL


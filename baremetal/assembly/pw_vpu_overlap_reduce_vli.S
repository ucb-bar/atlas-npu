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
    dma.config x5, ch0
    LI    x2, 1024
    LI    x21, 64

    # Scale registers for fp8pack / fp8unpack cases
    seli e0, 0x80
    seli e1, 0x81
    seli e2, 0x82
    seli e3, 0x7F

    # Load safe slot1 pair -> m0,m1
    LI    x6, 0
    LI    x1, 0x90000000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 256
    LI    x1, 0x90000400
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # Load signed slot1 pair -> m2,m3
    LI    x6, 512
    LI    x1, 0x90000800
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 768
    LI    x1, 0x90000C00
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 512
    vload m2, 0(x6)
    DELAY 33
    vload m3, 8(x6)
    DELAY 33

    # Load safe slot2 pair -> m4,m5
    LI    x6, 1024
    LI    x1, 0x90001000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 1280
    LI    x1, 0x90001400
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 1024
    vload m4, 0(x6)
    DELAY 33
    vload m5, 8(x6)
    DELAY 33

    # Load signed slot2 pair -> m6,m7
    LI    x6, 1536
    LI    x1, 0x90001800
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 1792
    LI    x1, 0x90001C00
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 1536
    vload m6, 0(x6)
    DELAY 33
    vload m7, 8(x6)
    DELAY 33

    # Load packed slot1 bank -> m8
    LI    x6, 2048
    LI    x1, 0x90002000
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 2048
    vload m8, 0(x6)
    DELAY 33

    # Load packed slot2 bank -> m9
    LI    x6, 2304
    LI    x1, 0x90002400
    dma.load x6, x1, x2, ch0
    dma.wait ch0
    LI    x6, 2304
    vload m9, 0(x6)
    DELAY 33

    # ----------------------------------------------------------------
    # Case 000: log || mov
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900028C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 001: log || rcp
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900029C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 002: log || exp
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 003: log || exp2
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 004: log || square
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 005: log || cube
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 006: log || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vlog2.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 007: log || fp8unpack
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90002F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90002F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90002F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90002FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 008: log || relu
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900030C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 009: log || sin
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vsin.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900031C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 010: log || cos
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vcos.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900032C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 011: log || tanh
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vtanh.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900033C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 012: log || sqrt
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900034C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 013: log || csum
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900035C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 014: log || cmin
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900036C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 015: log || cmax
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900037C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 016: log || vliAll
    # ----------------------------------------------------------------
    vlog2.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900038C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 017: log || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vlog2.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900039C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 018: log || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vlog2.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 019: log || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vlog2.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 020: sqrt || mov
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 021: sqrt || rcp
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 022: sqrt || exp
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 023: sqrt || exp2
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90003F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90003F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90003F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90003FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 024: sqrt || square
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900040C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 025: sqrt || cube
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900041C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 026: sqrt || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsqrt.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900042C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 027: sqrt || fp8unpack
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900043C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 028: sqrt || relu
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900044C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 029: sqrt || sin
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vsin.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900045C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 030: sqrt || cos
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vcos.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900046C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 031: sqrt || tanh
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vtanh.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900047C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 032: sqrt || log
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900048C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 033: sqrt || csum
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900049C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 034: sqrt || cmin
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 035: sqrt || cmax
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 036: sqrt || vliAll
    # ----------------------------------------------------------------
    vsqrt.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 037: sqrt || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsqrt.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 038: sqrt || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsqrt.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 039: sqrt || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsqrt.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90004F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90004F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90004F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90004FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 040: csum || mov
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vmov m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900050C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 041: csum || rcp
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vrecip.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900051C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 042: csum || exp
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vexp.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900052C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 043: csum || exp2
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vexp2.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900053C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 044: csum || square
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vsquare.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900054C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 045: csum || cube
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vcube.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900055C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 046: csum || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredsum.bf16 m16, m2
    vpack.bf16.fp8 m18, m4, e2
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900056C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 047: csum || fp8unpack
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900057C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 048: csum || relu
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vrelu.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900058C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 049: csum || sin
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vsin.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900059C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 050: csum || cos
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vcos.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 051: csum || tanh
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vtanh.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 052: csum || log
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vlog2.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 053: csum || sqrt
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vsqrt.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 054: csum || cmin
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 055: csum || cmax
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90005F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90005F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90005F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90005FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 056: csum || vliAll
    # ----------------------------------------------------------------
    vredsum.bf16 m16, m2
    vli.all m18, 0x3FA0
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900060C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 057: csum || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredsum.bf16 m16, m2
    vli.row m18, 0x4020
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900061C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 058: csum || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredsum.bf16 m16, m2
    vli.col m19, 0x4070
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900062C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 059: csum || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredsum.bf16 m16, m2
    vli.one m19, 0x4090
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900063C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 060: cmin || mov
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vmov m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900064C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 061: cmin || rcp
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vrecip.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900065C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 062: cmin || exp
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vexp.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900066C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 063: cmin || exp2
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vexp2.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900067C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 064: cmin || square
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vsquare.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900068C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 065: cmin || cube
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vcube.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900069C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 066: cmin || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredmin.bf16 m16, m2
    vpack.bf16.fp8 m18, m4, e2
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 067: cmin || fp8unpack
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 068: cmin || relu
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vrelu.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 069: cmin || sin
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vsin.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 070: cmin || cos
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vcos.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 071: cmin || tanh
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vtanh.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90006F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90006F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90006F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90006FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 072: cmin || log
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vlog2.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900070C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 073: cmin || sqrt
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vsqrt.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900071C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 074: cmin || csum
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900072C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 075: cmin || cmax
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900073C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 076: cmin || vliAll
    # ----------------------------------------------------------------
    vredmin.bf16 m16, m2
    vli.all m18, 0x3FA0
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900074C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 077: cmin || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredmin.bf16 m16, m2
    vli.row m18, 0x4020
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900075C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 078: cmin || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredmin.bf16 m16, m2
    vli.col m19, 0x4070
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900076C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 079: cmin || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredmin.bf16 m16, m2
    vli.one m19, 0x4090
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900077C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 080: cmax || mov
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vmov m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900078C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 081: cmax || rcp
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vrecip.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900079C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 082: cmax || exp
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vexp.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 083: cmax || exp2
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vexp2.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 084: cmax || square
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vsquare.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 085: cmax || cube
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vcube.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 086: cmax || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredmax.bf16 m16, m2
    vpack.bf16.fp8 m18, m4, e2
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 087: cmax || fp8unpack
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90007F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90007F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90007F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90007FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 088: cmax || relu
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vrelu.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900080C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 089: cmax || sin
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vsin.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900081C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 090: cmax || cos
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vcos.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900082C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 091: cmax || tanh
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vtanh.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900083C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 092: cmax || log
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vlog2.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900084C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 093: cmax || sqrt
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vsqrt.bf16 m18, m4
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900085C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 094: cmax || csum
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900086C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 095: cmax || cmin
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900087C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 096: cmax || vliAll
    # ----------------------------------------------------------------
    vredmax.bf16 m16, m2
    vli.all m18, 0x3FA0
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900088C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 097: cmax || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredmax.bf16 m16, m2
    vli.row m18, 0x4020
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900089C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 098: cmax || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredmax.bf16 m16, m2
    vli.col m19, 0x4070
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 099: cmax || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vredmax.bf16 m16, m2
    vli.one m19, 0x4090
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 100: vliAll || mov
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 101: vliAll || rcp
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 102: vliAll || exp
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 103: vliAll || exp2
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90008F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90008F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90008F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90008FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 104: vliAll || square
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900090C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 105: vliAll || cube
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900091C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 106: vliAll || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vli.all m16, 0x3FA0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900092C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 107: vliAll || fp8unpack
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900093C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 108: vliAll || relu
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900094C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 109: vliAll || sin
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vsin.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900095C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 110: vliAll || cos
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vcos.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900096C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 111: vliAll || tanh
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vtanh.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900097C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 112: vliAll || log
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vlog2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900098C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 113: vliAll || sqrt
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900099C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 114: vliAll || csum
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 115: vliAll || cmin
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 116: vliAll || cmax
    # ----------------------------------------------------------------
    vli.all m16, 0x3FA0
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 117: vliRow || mov
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 118: vliRow || rcp
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 119: vliRow || exp
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90009F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90009F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90009F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90009FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 120: vliRow || exp2
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A0C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 121: vliRow || square
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A1C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 122: vliRow || cube
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A2C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 123: vliRow || fp8pack
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.all m18, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A3C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 124: vliRow || fp8unpack
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A4C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 125: vliRow || relu
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A5C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 126: vliRow || sin
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vsin.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A6C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 127: vliRow || cos
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vcos.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A7C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 128: vliRow || tanh
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vtanh.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A8C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 129: vliRow || log
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vlog2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000A900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000A940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000A980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000A9C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 130: vliRow || sqrt
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000AA00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AA40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AA80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000AAC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 131: vliRow || csum
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000AB00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AB40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AB80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000ABC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 132: vliRow || cmin
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000AC00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AC40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AC80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000ACC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 133: vliRow || cmax
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.row m16, 0x4020
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000AD00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AD40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AD80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000ADC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 134: vliCol || mov
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000AE00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AE40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AE80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000AEC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 135: vliCol || rcp
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000AF00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000AF40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000AF80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000AFC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 136: vliCol || exp
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B0C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 137: vliCol || exp2
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B1C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 138: vliCol || square
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B2C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 139: vliCol || cube
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B3C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 140: vliCol || fp8pack
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.all m18, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B4C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 141: vliCol || fp8unpack
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B5C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 142: vliCol || relu
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B6C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 143: vliCol || sin
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vsin.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B7C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 144: vliCol || cos
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vcos.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B8C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 145: vliCol || tanh
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vtanh.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000B900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000B940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000B980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000B9C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 146: vliCol || log
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vlog2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000BA00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BA40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BA80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BAC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 147: vliCol || sqrt
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000BB00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BB40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BB80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BBC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 148: vliCol || csum
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000BC00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BC40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BC80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BCC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 149: vliCol || cmin
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000BD00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BD40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BD80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BDC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 150: vliCol || cmax
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.col m17, 0x4070
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000BE00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BE40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BE80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BEC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 151: vliOne || mov
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000BF00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000BF40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000BF80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000BFC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 152: vliOne || rcp
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C0C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 153: vliOne || exp
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C1C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 154: vliOne || exp2
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C2C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 155: vliOne || square
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C3C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 156: vliOne || cube
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C4C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 157: vliOne || fp8pack
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.all m18, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C5C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 158: vliOne || fp8unpack
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C6C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 159: vliOne || relu
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C7C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 160: vliOne || sin
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vsin.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C8C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 161: vliOne || cos
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vcos.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000C900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000C940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000C980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000C9C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 162: vliOne || tanh
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vtanh.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000CA00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CA40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CA80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CAC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 163: vliOne || log
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vlog2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000CB00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CB40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CB80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CBC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 164: vliOne || sqrt
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000CC00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CC40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CC80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CCC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 165: vliOne || csum
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000CD00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CD40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CD80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CDC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 166: vliOne || cmin
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000CE00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CE40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CE80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CEC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 167: vliOne || cmax
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.one m17, 0x4090
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000CF00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000CF40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000CF80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000CFC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL


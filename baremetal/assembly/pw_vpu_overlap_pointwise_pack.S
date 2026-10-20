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
    # Case 000: mov || rcp
    # ----------------------------------------------------------------
    vmov m16, m0
    vrecip.bf16 m18, m4
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
    # Case 001: mov || exp
    # ----------------------------------------------------------------
    vmov m16, m0
    vexp.bf16 m18, m4
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
    # Case 002: mov || exp2
    # ----------------------------------------------------------------
    vmov m16, m0
    vexp2.bf16 m18, m4
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
    # Case 003: mov || square
    # ----------------------------------------------------------------
    vmov m16, m0
    vsquare.bf16 m18, m4
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
    # Case 004: mov || cube
    # ----------------------------------------------------------------
    vmov m16, m0
    vcube.bf16 m18, m4
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
    # Case 005: mov || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vmov m16, m0
    vpack.bf16.fp8 m18, m4, e2
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
    # Case 006: mov || fp8unpack
    # ----------------------------------------------------------------
    vmov m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
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
    # Case 007: mov || relu
    # ----------------------------------------------------------------
    vmov m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
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
    # Case 008: mov || sin
    # ----------------------------------------------------------------
    vmov m16, m0
    vsin.bf16 m18, m4
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
    # Case 009: mov || cos
    # ----------------------------------------------------------------
    vmov m16, m0
    vcos.bf16 m18, m4
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
    # Case 010: mov || tanh
    # ----------------------------------------------------------------
    vmov m16, m0
    vtanh.bf16 m18, m4
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
    # Case 011: mov || log
    # ----------------------------------------------------------------
    vmov m16, m0
    vlog2.bf16 m18, m4
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
    # Case 012: mov || sqrt
    # ----------------------------------------------------------------
    vmov m16, m0
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
    # Case 013: mov || csum
    # ----------------------------------------------------------------
    vmov m16, m0
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
    # Case 014: mov || cmin
    # ----------------------------------------------------------------
    vmov m16, m0
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
    # Case 015: mov || cmax
    # ----------------------------------------------------------------
    vmov m16, m0
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
    # Case 016: mov || vliAll
    # ----------------------------------------------------------------
    vmov m16, m0
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
    # Case 017: mov || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vmov m16, m0
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
    # Case 018: mov || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vmov m16, m0
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
    # Case 019: mov || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vmov m16, m0
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
    # Case 020: rcp || mov
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
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
    # Case 021: rcp || exp
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vexp.bf16 m18, m4
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
    # Case 022: rcp || exp2
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vexp2.bf16 m18, m4
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
    # Case 023: rcp || square
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vsquare.bf16 m18, m4
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
    # Case 024: rcp || cube
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vcube.bf16 m18, m4
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
    # Case 025: rcp || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vrecip.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
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
    # Case 026: rcp || fp8unpack
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
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
    # Case 027: rcp || relu
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
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
    # Case 028: rcp || sin
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vsin.bf16 m18, m4
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
    # Case 029: rcp || cos
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vcos.bf16 m18, m4
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
    # Case 030: rcp || tanh
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vtanh.bf16 m18, m4
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
    # Case 031: rcp || log
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vlog2.bf16 m18, m4
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
    # Case 032: rcp || sqrt
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
    vsqrt.bf16 m18, m4
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
    # Case 033: rcp || csum
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
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
    # Case 034: rcp || cmin
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
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
    # Case 035: rcp || cmax
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
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
    # Case 036: rcp || vliAll
    # ----------------------------------------------------------------
    vrecip.bf16 m16, m0
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
    # Case 037: rcp || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vrecip.bf16 m16, m0
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
    # Case 038: rcp || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vrecip.bf16 m16, m0
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
    # Case 039: rcp || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vrecip.bf16 m16, m0
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
    # Case 040: exp || mov
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vmov m18, m4
    DELAY 64
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
    # Case 041: exp || rcp
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
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
    # Case 042: exp || square
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vsquare.bf16 m18, m4
    DELAY 64
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
    # Case 043: exp || cube
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vcube.bf16 m18, m4
    DELAY 64
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
    # Case 044: exp || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vexp.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
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
    # Case 045: exp || fp8unpack
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
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
    # Case 046: exp || relu
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
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
    # Case 047: exp || sin
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vsin.bf16 m18, m4
    DELAY 64
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
    # Case 048: exp || cos
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vcos.bf16 m18, m4
    DELAY 64
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
    # Case 049: exp || tanh
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vtanh.bf16 m18, m4
    DELAY 64
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
    # Case 050: exp || log
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
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
    # Case 051: exp || sqrt
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vsqrt.bf16 m18, m4
    DELAY 64
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
    # Case 052: exp || csum
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vredsum.bf16 m18, m6
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
    # Case 053: exp || cmin
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vredmin.bf16 m18, m6
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
    # Case 054: exp || cmax
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vredmax.bf16 m18, m6
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
    # Case 055: exp || vliAll
    # ----------------------------------------------------------------
    vexp.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
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
    # Case 056: exp || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vexp.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
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
    # Case 057: exp || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vexp.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
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
    # Case 058: exp || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vexp.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
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
    # Case 059: exp2 || mov
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vmov m18, m4
    DELAY 64
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
    # Case 060: exp2 || rcp
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
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
    # Case 061: exp2 || square
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vsquare.bf16 m18, m4
    DELAY 64
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
    # Case 062: exp2 || cube
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vcube.bf16 m18, m4
    DELAY 64
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
    # Case 063: exp2 || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vexp2.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
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
    # Case 064: exp2 || fp8unpack
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
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
    # Case 065: exp2 || relu
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
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
    # Case 066: exp2 || sin
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vsin.bf16 m18, m4
    DELAY 64
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
    # Case 067: exp2 || cos
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vcos.bf16 m18, m4
    DELAY 64
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
    # Case 068: exp2 || tanh
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vtanh.bf16 m18, m4
    DELAY 64
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
    # Case 069: exp2 || log
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
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
    # Case 070: exp2 || sqrt
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vsqrt.bf16 m18, m4
    DELAY 64
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
    # Case 071: exp2 || csum
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vredsum.bf16 m18, m6
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
    # Case 072: exp2 || cmin
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vredmin.bf16 m18, m6
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
    # Case 073: exp2 || cmax
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vredmax.bf16 m18, m6
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
    # Case 074: exp2 || vliAll
    # ----------------------------------------------------------------
    vexp2.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
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
    # Case 075: exp2 || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vexp2.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
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
    # Case 076: exp2 || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vexp2.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
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
    # Case 077: exp2 || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vexp2.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
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
    # Case 078: square || mov
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vmov m18, m4
    DELAY 64
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
    # Case 079: square || rcp
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
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
    # Case 080: square || exp
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vexp.bf16 m18, m4
    DELAY 64
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
    # Case 081: square || exp2
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vexp2.bf16 m18, m4
    DELAY 64
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
    # Case 082: square || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsquare.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
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
    # Case 083: square || fp8unpack
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
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
    # Case 084: square || relu
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
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
    # Case 085: square || sin
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vsin.bf16 m18, m4
    DELAY 64
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
    # Case 086: square || cos
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vcos.bf16 m18, m4
    DELAY 64
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
    # Case 087: square || tanh
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vtanh.bf16 m18, m4
    DELAY 64
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
    # Case 088: square || log
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
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
    # Case 089: square || sqrt
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vsqrt.bf16 m18, m4
    DELAY 64
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
    # Case 090: square || csum
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vredsum.bf16 m18, m6
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
    # Case 091: square || cmin
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vredmin.bf16 m18, m6
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
    # Case 092: square || cmax
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vredmax.bf16 m18, m6
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
    # Case 093: square || vliAll
    # ----------------------------------------------------------------
    vsquare.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
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
    # Case 094: square || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsquare.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
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
    # Case 095: square || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsquare.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
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
    # Case 096: square || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsquare.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
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
    # Case 097: cube || mov
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vmov m18, m4
    DELAY 64
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
    # Case 098: cube || rcp
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
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
    # Case 099: cube || exp
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vexp.bf16 m18, m4
    DELAY 64
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
    # Case 100: cube || exp2
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vexp2.bf16 m18, m4
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
    # Case 101: cube || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vcube.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
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
    # Case 102: cube || fp8unpack
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
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
    # Case 103: cube || relu
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vrelu.bf16 m18, m4
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
    # Case 104: cube || sin
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vsin.bf16 m18, m4
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
    # Case 105: cube || cos
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vcos.bf16 m18, m4
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
    # Case 106: cube || tanh
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vtanh.bf16 m18, m4
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
    # Case 107: cube || log
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
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
    # Case 108: cube || sqrt
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vsqrt.bf16 m18, m4
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
    # Case 109: cube || csum
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vredsum.bf16 m18, m6
    DELAY 128
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
    # Case 110: cube || cmin
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vredmin.bf16 m18, m6
    DELAY 128
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
    # Case 111: cube || cmax
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vredmax.bf16 m18, m6
    DELAY 128
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
    # Case 112: cube || vliAll
    # ----------------------------------------------------------------
    vcube.bf16 m16, m0
    vli.all m18, 0x3FA0
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
    # Case 113: cube || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vcube.bf16 m16, m0
    vli.row m18, 0x4020
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
    # Case 114: cube || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vcube.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
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
    # Case 115: cube || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vcube.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
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
    # Case 116: fp8pack || mov
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vmov m18, m4
    DELAY 64
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
    # Case 117: fp8pack || rcp
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vrecip.bf16 m18, m4
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
    # Case 118: fp8pack || exp
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vexp.bf16 m18, m4
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
    # Case 119: fp8pack || exp2
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vexp2.bf16 m18, m4
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
    # Case 120: fp8pack || square
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vsquare.bf16 m18, m4
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
    # Case 121: fp8pack || cube
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vcube.bf16 m18, m4
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
    # Case 122: fp8pack || fp8unpack
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
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
    # Case 123: fp8pack || relu
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vrelu.bf16 m18, m4
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
    # Case 124: fp8pack || sin
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vsin.bf16 m18, m4
    DELAY 64
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
    # Case 125: fp8pack || cos
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vcos.bf16 m18, m4
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
    # Case 126: fp8pack || tanh
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vtanh.bf16 m18, m4
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
    # Case 127: fp8pack || log
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vlog2.bf16 m18, m4
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
    # Case 128: fp8pack || sqrt
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vsqrt.bf16 m18, m4
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
    # Case 129: fp8pack || csum
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vredsum.bf16 m18, m6
    DELAY 128
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
    # Case 130: fp8pack || cmin
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vredmin.bf16 m18, m6
    DELAY 128
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
    # Case 131: fp8pack || cmax
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vredmax.bf16 m18, m6
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
    # Case 132: fp8pack || vliAll
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vli.all m18, 0x3FA0
    DELAY 64
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
    # Case 133: fp8pack || vliRow
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.all m18, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vli.row m18, 0x4020
    DELAY 64
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
    # Case 134: fp8pack || vliCol
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.all m18, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vli.col m19, 0x4070
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
    # Case 135: fp8pack || vliOne
    # ----------------------------------------------------------------
    vli.all m16, 0x0000
    DELAY 64
    vli.all m18, 0x0000
    DELAY 64
    vpack.bf16.fp8 m16, m0, e0
    vli.one m19, 0x4090
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
    # Case 136: fp8unpack || mov
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vmov m18, m4
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
    # Case 137: fp8unpack || rcp
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vrecip.bf16 m18, m4
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
    # Case 138: fp8unpack || exp
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vexp.bf16 m18, m4
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
    # Case 139: fp8unpack || exp2
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vexp2.bf16 m18, m4
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
    # Case 140: fp8unpack || square
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vsquare.bf16 m18, m4
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
    # Case 141: fp8unpack || cube
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vcube.bf16 m18, m4
    DELAY 64
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
    # Case 142: fp8unpack || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vunpack.fp8.bf16 m16, m8, e1
    vpack.bf16.fp8 m18, m4, e2
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
    # Case 143: fp8unpack || relu
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vrelu.bf16 m18, m4
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
    # Case 144: fp8unpack || sin
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vsin.bf16 m18, m4
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
    # Case 145: fp8unpack || cos
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vcos.bf16 m18, m4
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
    # Case 146: fp8unpack || tanh
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vtanh.bf16 m18, m4
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
    # Case 147: fp8unpack || log
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vlog2.bf16 m18, m4
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
    # Case 148: fp8unpack || sqrt
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vsqrt.bf16 m18, m4
    DELAY 64
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
    # Case 149: fp8unpack || csum
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vredsum.bf16 m18, m6
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
    # Case 150: fp8unpack || cmin
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vredmin.bf16 m18, m6
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
    # Case 151: fp8unpack || cmax
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vredmax.bf16 m18, m6
    DELAY 128
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
    # Case 152: fp8unpack || vliAll
    # ----------------------------------------------------------------
    vunpack.fp8.bf16 m16, m8, e1
    vli.all m18, 0x3FA0
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
    # Case 153: fp8unpack || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vunpack.fp8.bf16 m16, m8, e1
    vli.row m18, 0x4020
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
    # Case 154: fp8unpack || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vunpack.fp8.bf16 m16, m8, e1
    vli.col m19, 0x4070
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
    # Case 155: fp8unpack || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vunpack.fp8.bf16 m16, m8, e1
    vli.one m19, 0x4090
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
    # Case 156: relu || mov
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vmov m18, m4
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
    # Case 157: relu || rcp
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vrecip.bf16 m18, m4
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
    # Case 158: relu || exp
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vexp.bf16 m18, m4
    DELAY 64
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
    # Case 159: relu || exp2
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vexp2.bf16 m18, m4
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
    # Case 160: relu || square
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vsquare.bf16 m18, m4
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
    # Case 161: relu || cube
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vcube.bf16 m18, m4
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
    # Case 162: relu || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vrelu.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
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
    # Case 163: relu || fp8unpack
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
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
    # Case 164: relu || sin
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vsin.bf16 m18, m4
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
    # Case 165: relu || cos
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vcos.bf16 m18, m4
    DELAY 64
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
    # Case 166: relu || tanh
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vtanh.bf16 m18, m4
    DELAY 64
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
    # Case 167: relu || log
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
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

    # ----------------------------------------------------------------
    # Case 168: relu || sqrt
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D0C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 169: relu || csum
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D1C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 170: relu || cmin
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D2C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 171: relu || cmax
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D3C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 172: relu || vliAll
    # ----------------------------------------------------------------
    vrelu.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D4C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 173: relu || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vrelu.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D5C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 174: relu || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vrelu.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D6C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 175: relu || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vrelu.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D7C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 176: sin || mov
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D8C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 177: sin || rcp
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000D900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000D940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000D980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000D9C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 178: sin || exp
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000DA00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DA40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DA80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DAC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 179: sin || exp2
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000DB00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DB40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DB80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DBC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 180: sin || square
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000DC00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DC40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DC80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DCC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 181: sin || cube
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000DD00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DD40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DD80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DDC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 182: sin || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsin.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000DE00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DE40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DE80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DEC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 183: sin || fp8unpack
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000DF00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000DF40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000DF80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000DFC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 184: sin || relu
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E0C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 185: sin || tanh
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vtanh.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E1C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 186: sin || log
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E2C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 187: sin || sqrt
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E3C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 188: sin || csum
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E4C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 189: sin || cmin
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E5C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 190: sin || cmax
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E6C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 191: sin || vliAll
    # ----------------------------------------------------------------
    vsin.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E7C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 192: sin || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsin.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E8C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 193: sin || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsin.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000E900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000E940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000E980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000E9C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 194: sin || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vsin.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000EA00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EA40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EA80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EAC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 195: cos || mov
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000EB00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EB40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EB80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EBC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 196: cos || rcp
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000EC00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EC40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EC80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000ECC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 197: cos || exp
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000ED00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000ED40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000ED80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EDC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 198: cos || exp2
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000EE00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EE40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EE80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EEC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 199: cos || square
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000EF00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000EF40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000EF80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000EFC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 200: cos || cube
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F0C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 201: cos || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vcos.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F1C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 202: cos || fp8unpack
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F2C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 203: cos || relu
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F3C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 204: cos || tanh
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vtanh.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F4C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 205: cos || log
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F5C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 206: cos || sqrt
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F6C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 207: cos || csum
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F7C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 208: cos || cmin
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F8C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 209: cos || cmax
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000F900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000F940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000F980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000F9C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 210: cos || vliAll
    # ----------------------------------------------------------------
    vcos.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000FA00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FA40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FA80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FAC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 211: cos || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vcos.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000FB00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FB40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FB80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FBC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 212: cos || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vcos.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000FC00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FC40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FC80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FCC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 213: cos || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vcos.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000FD00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FD40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FD80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FDC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 214: tanh || mov
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vmov m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000FE00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FE40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FE80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FEC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 215: tanh || rcp
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vrecip.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x9000FF00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x9000FF40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x9000FF80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x9000FFC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 216: tanh || exp
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vexp.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900100C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 217: tanh || exp2
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vexp2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900101C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 218: tanh || square
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vsquare.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010200
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010240
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010280
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900102C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 219: tanh || cube
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vcube.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010300
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010340
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010380
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900103C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 220: tanh || fp8pack
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vtanh.bf16 m16, m0
    vpack.bf16.fp8 m18, m4, e2
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010400
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010440
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010480
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900104C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 221: tanh || fp8unpack
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vunpack.fp8.bf16 m18, m9, e3
    DELAY 65
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010500
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010540
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010580
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900105C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 222: tanh || relu
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vrelu.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010600
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010640
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010680
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900106C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 223: tanh || sin
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vsin.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010700
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010740
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010780
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900107C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 224: tanh || cos
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vcos.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010800
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010840
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010880
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900108C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 225: tanh || log
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vlog2.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010900
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010940
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010980
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900109C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 226: tanh || sqrt
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vsqrt.bf16 m18, m4
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010A00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010A40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010A80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010AC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 227: tanh || csum
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vredsum.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010B00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010B40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010B80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010BC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 228: tanh || cmin
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vredmin.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010C00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010C40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010C80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010CC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 229: tanh || cmax
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vredmax.bf16 m18, m6
    DELAY 128
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010D00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010D40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010D80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010DC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 230: tanh || vliAll
    # ----------------------------------------------------------------
    vtanh.bf16 m16, m0
    vli.all m18, 0x3FA0
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010E00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010E40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010E80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010EC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 231: tanh || vliRow
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vtanh.bf16 m16, m0
    vli.row m18, 0x4020
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90010F00
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90010F40
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90010F80
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x90010FC0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 232: tanh || vliCol
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vtanh.bf16 m16, m0
    vli.col m19, 0x4070
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90011000
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90011040
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90011080
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900110C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

    # ----------------------------------------------------------------
    # Case 233: tanh || vliOne
    # ----------------------------------------------------------------
    vli.all m18, 0x0000
    DELAY 64
    vtanh.bf16 m16, m0
    vli.one m19, 0x4090
    DELAY 64
    LI    x8, 1024
    vstore m16, 0(x8)
    DELAY 33
    LI    x8, 1024
    LI    x3, 0x90011100
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1280
    vstore m17, 0(x8)
    DELAY 33
    LI    x8, 1280
    LI    x3, 0x90011140
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1536
    vstore m18, 0(x8)
    DELAY 33
    LI    x8, 1536
    LI    x3, 0x90011180
    dma.store x3, x8, x21, ch1
    dma.wait ch1
    LI    x8, 1792
    vstore m19, 0(x8)
    DELAY 33
    LI    x8, 1792
    LI    x3, 0x900111C0
    dma.store x3, x8, x21, ch1
    dma.wait ch1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL


# vpu_reduction.S — VPU reduction operations integration test
#
# Tensor reg allocation: 0,2 = A (input)    8,10 = result
# VMEM_BASE:      0x2000_0000
# ATLAS_CSR_BASE: 0x0004_0000
#
# Each test times only the VPU op itself (op + DELAY).
# VSTORE, LHU, and verify branches are outside the timed window.
# The cumulative cycle count across all tests is accumulated in x26.
#
# Total theoretical        : 576
# actual = 627
# utilization = 91.8%

    ADDI  x28, x0, 0           # x28 tracks the current test number

    # ── Setup Input A (4.0 -> BF16: 0x4080) ──
    CSRRS x24, 0xC00, x0
    VLI.ALL 0, 0x4080
    DELAY 63
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x0,  x9         # x26 = cumulative cycles (start fresh)

    # ── Setup Input B (2.0 -> BF16: 0x4000) ──
    CSRRS x24, 0xC00, x0
    VLI.ALL 2, 0x4000
    DELAY 63
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    # ================================================================
    # Test 1: VREDSUM.BF16  8, 0  (col-wise sum: 64 x 4.0 = 256.0 -> 0x4380)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VREDSUM.BF16  8, 0
    DELAY 128
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4380
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 2: VREDMIN.BF16  8, 0  (col-wise min: min(4.0) = 4.0 -> 0x4080)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VREDMIN.BF16  8, 0
    DELAY 128
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4080
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 3: VREDMAX.BF16  8, 0  (col-wise max: max(4.0) = 4.0 -> 0x4080)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VREDMAX.BF16  8, 0
    DELAY 128
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4080
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 4: VREDSUM.ROW.BF16  8, 0  (row-wise sum: 32 x 4.0 = 128.0 -> 0x4300)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VREDSUM.ROW.BF16  8, 0
    DELAY 37
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4300
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 5: VREDMIN.ROW.BF16  8, 0  (row-wise min: min(4.0) = 4.0 -> 0x4080)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VREDMIN.ROW.BF16  8, 0
    DELAY 32
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4080
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 6: VREDMAX.ROW.BF16  8, 0  (row-wise max: max(4.0) = 4.0 -> 0x4080)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VREDMAX.ROW.BF16  8, 0
    DELAY 32
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4080
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

pass:
    CSRRW x0, 0xC11, x26       # write cumulative timed cycles to CSR_DBG1
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
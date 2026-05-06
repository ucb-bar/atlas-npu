# vpu_binary.S — VPU binary operations integration test
#
# Tensor reg allocation: 0,2 = A    4,6 = B    8,10 = result
# VMEM_BASE:      0x2000_0000
# ATLAS_CSR_BASE: 0x0004_0000
#
# Each test times only the VPU op itself (op + DELAY).
# VSTORE, LHU, and verify branches are outside the timed window.
# The cumulative cycle count across all tests is accumulated in x26.
#
# total theoretical = 64*6 = 384
# actual = 460
# utilization = 83.4%

    ADDI  x28, x0, 1           # x28 tracks the current test number

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
    # Test 1: vadd.bf16  8, 0, 2  (4.0 + 2.0 = 6.0 -> 0x40C0)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VADD.BF16  8, 0, 2
    DELAY 64
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x40C0
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 2: vsub.bf16  8, 0, 2  (4.0 - 2.0 = 2.0 -> 0x4000)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VSUB.BF16  8, 0, 2
    DELAY 64
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4000
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 3: vmul.bf16  8, 0, 2  (4.0 * 2.0 = 8.0 -> 0x4100)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VMUL.BF16  8, 0, 2
    DELAY 64
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4100
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 4: vmin.bf16  8, 0, 2  (min(4.0, 2.0) = 2.0 -> 0x4000)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VMIN.BF16  8, 0, 2
    DELAY 64
    CSRRS x25, 0xC00, x0
    SUB   x9,  x25, x24
    ADD   x26, x26, x9

    LI    x8, 0x20000000
    VSTORE 8, x8, 0
    DELAY 33
    LHU   x10, x8, 0
    DELAY 2
    LI    x11, 0x4000
    BNE   x10, x11, fail
    ADDI  x28, x28, 1

    # ================================================================
    # Test 5: vmax.bf16  8, 0, 2  (max(4.0, 2.0) = 4.0 -> 0x4080)
    # ================================================================
    CSRRS x24, 0xC00, x0
    VMAX.BF16  8, 0, 2
    DELAY 64
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
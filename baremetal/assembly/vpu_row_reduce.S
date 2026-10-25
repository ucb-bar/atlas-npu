# vpu_row_reduce.S — VPU row (lane) reduction ops
# Tests: vredsum.row.bf16, vredmin.row.bf16, vredmax.row.bf16
# @TIMEOUT 30000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_row_reduce.py
#
# Row reductions reduce over the full BF16 tensor-register pair for each row:
#   vredsum.row.bf16: m[vd][i,j] = sum(m[vs1][i,:] ++ m[vs1+1][i,:])
#   vredmin.row.bf16: m[vd][i,j] = min(m[vs1][i,:] ++ m[vs1+1][i,:])
#   vredmax.row.bf16: m[vd][i,j] = max(m[vs1][i,:] ++ m[vs1+1][i,:])
#
# Result is a tensor where every column in a given row has the same
# reduction value from that row.
#
# DRAM layout:
#   [0x0000..0x07FF]  Input A                 (2048 B)
#   [0x0800..0x0FFF]  vredsum.row output      (2048 B)
#   [0x1000..0x17FF]  vredmin.row output      (2048 B)
#   [0x1800..0x1FFF]  vredmax.row output      (2048 B)

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

    # ================================================================
    # Test 1: vredsum.row.bf16  m4, m0
    # ================================================================
    vredsum.row.bf16 m4, m0
    DELAY 37

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
    # Test 2: vredmin.row.bf16  m4, m0
    # ================================================================
    vredmin.row.bf16 m4, m0
    DELAY 32

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
    # Test 3: vredmax.row.bf16  m4, m0
    # ================================================================
    vredmax.row.bf16 m4, m0
    DELAY 32

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

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0x800, x1
    ECALL

fail:
    CSRRW x0, 0x800, x28
    ECALL

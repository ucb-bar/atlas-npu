# vpu_col_reduce.S — VPU column (sublane) reduction ops
# Tests: vredsum.bf16, vredmin.bf16, vredmax.bf16
# @TIMEOUT 40000
# @DRAM_BASE 0x90000000
# @PYTHON_GEN gen_vpu_col_reduce.py
#
# Column reductions reduce over rows for each column:
#   vredsum.bf16: m[vd][i,j] = sum(m[vs1][:,j])
#   vredmin.bf16: m[vd][i,j] = min(m[vs1][:,j])
#   vredmax.bf16: m[vd][i,j] = max(m[vs1][:,j])
#
# The FSM reads 128 rows (counter 0–127) from vs1,vs1+1 and writes
# 64 rows to vd,vd+1 so needs a longer delay (~250 cycles).
#
# DRAM layout:
#   [0x0000..0x07FF]  Input A            (2048 B)
#   [0x0800..0x0FFF]  vredsum output     (2048 B)
#   [0x1000..0x17FF]  vredmin output     (2048 B)
#   [0x1800..0x1FFF]  vredmax output     (2048 B)

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
    # Test 1: vredsum.bf16  m4, m0
    # ================================================================
    vredsum.bf16 m4, m0
    DELAY 128

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

    # Reload A (VPU may have consumed it)
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ================================================================
    # Test 2: vredmin.bf16  m4, m0
    # ================================================================
    vredmin.bf16 m4, m0
    DELAY 128

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

    # Reload A
    ADDI  x6, x0, 0
    vload m0, 0(x6)
    DELAY 33
    vload m1, 8(x6)
    DELAY 33

    # ================================================================
    # Test 3: vredmax.bf16  m4, m0
    # ================================================================
    vredmax.bf16 m4, m0
    DELAY 128

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

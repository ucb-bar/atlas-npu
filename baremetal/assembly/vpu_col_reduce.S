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
    LI    x5, 0x90000000
    DMA.CONFIG x5, 0
    LI    x2, 1024

    # ── Load Input A → m0,m1 ──
    ADDI  x6, x0, 0
    ADDI  x1, x0, 0
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    LI    x6, 256
    LI    x1, 1024
    DMA.LOAD  x6, x1, x2, 0
    DMA.WAIT  0
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ================================================================
    # Test 1: vredsum.bf16  m4, m0
    # ================================================================
    VREDSUM.BF16 4, 0
    DELAY 250

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x0800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x0C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # Reload A (VPU may have consumed it)
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ================================================================
    # Test 2: vredmin.bf16  m4, m0
    # ================================================================
    VREDMIN.BF16 4, 0
    DELAY 250

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x1000
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x1400
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

    # Reload A
    ADDI  x6, x0, 0
    VLOAD 0, x6, 0
    DELAY 40
    VLOAD 1, x6, 8
    DELAY 40

    # ================================================================
    # Test 3: vredmax.bf16  m4, m0
    # ================================================================
    VREDMAX.BF16 4, 0
    DELAY 250

    LI    x8, 1024
    VSTORE 4, x8, 0
    DELAY 40
    VSTORE 5, x8, 8
    DELAY 40
    LI    x8, 1024
    LI    x3, 0x1800
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1
    LI    x8, 1280
    LI    x3, 0x1C00
    DMA.STORE x3, x8, x2, 1
    DMA.WAIT  1

pass:
    ADDI  x1, x0, 1
    CSRRW x0, 0xC10, x1
    ECALL

fail:
    CSRRW x0, 0xC10, x28
    ECALL
